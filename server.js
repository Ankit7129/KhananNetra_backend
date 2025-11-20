import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import morgan from 'morgan';
import cookieParser from 'cookie-parser'; // ✅ ADD THIS
import fs from 'fs';
import { spawn, spawnSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

import connectDB from './config/database.js';

// Route imports
import authRoutes from './routes/auth.js';
import userRoutes from './routes/users.js';
import adminRoutes from './routes/adminRoutes.js';
import pythonProxyRoutes from './routes/pythonProxy.js';
import historyRoutes from './routes/historyRoutes.js';

dotenv.config();
// Load env vars
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const shouldAutoStartPython = process.env.AUTO_START_PYTHON !== 'false';
const autoInstallPythonRequirements = process.env.AUTO_INSTALL_PYTHON_REQUIREMENTS !== 'false';
const pythonExecutableFallback = process.platform === 'win32' ? 'python' : 'python3';
const pythonHealthUrl = (() => {
  if (process.env.PYTHON_BACKEND_HEALTH_URL) {
    return process.env.PYTHON_BACKEND_HEALTH_URL;
  }
  const base = process.env.PYTHON_BACKEND_URL || process.env.PYTHON_API_URL || 'http://localhost:8000';
  return `${base.replace(/\/$/, '')}/health`;
})();
const autoCreatePythonVenv = process.env.AUTO_CREATE_PYTHON_VENV !== 'false';
const forcePythonRequirementsInstall = process.env.FORCE_PYTHON_REQUIREMENTS_INSTALL === 'true';

const parseCsvList = (value) =>
  (value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

const DEFAULT_PYTHON_IMPORT_CHECKS = ['fastapi', 'uvicorn', 'rasterio', 'numpy'];

const pythonImportChecks = (() => {
  const overrides = parseCsvList(process.env.PYTHON_BACKEND_IMPORT_CHECKS);
  return overrides.length > 0 ? overrides : DEFAULT_PYTHON_IMPORT_CHECKS;
})();

const runCommand = (command, args, options = {}) =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      ...options,
      stdio: options.stdio || 'inherit',
    });

    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        const error = new Error(`${command} ${args.join(' ')} exited with code ${code}`);
        error.code = code;
        reject(error);
      }
    });
  });

const detectPythonExecutable = () => {
  const preferred = process.env.PYTHON_BACKEND_EXECUTABLE
    || process.env.PYTHON_BACKEND_PYTHON
    || process.env.PYTHON;

  const candidates = [...new Set(
    [
      preferred,
      'python3.12',
      'python3.11',
      'python3.10',
      'python3.9',
      pythonExecutableFallback,
      'python3',
      'python',
    ].filter(Boolean),
  )];

  for (const command of candidates) {
    try {
      const result = spawnSync(command, ['--version'], {
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      if (result.error || result.status !== 0) {
        continue;
      }

      const versionOutput = (result.stdout || result.stderr || '').trim();
      return { command, version: versionOutput };
    } catch (error) {
      // Ignore and try the next candidate
    }
  }

  throw new Error(`Unable to locate a working Python interpreter. Tried: ${candidates.join(', ')}`);
};

const getVenvLayout = (pythonCwd) => {
  const venvPath = path.join(pythonCwd, '.venv');
  if (process.platform === 'win32') {
    const scriptsDir = path.join(venvPath, 'Scripts');
    return {
      venvPath,
      binPath: scriptsDir,
      pythonPath: path.join(scriptsDir, 'python.exe'),
    };
  }

  const binDir = path.join(venvPath, 'bin');
  return {
    venvPath,
    binPath: binDir,
    pythonPath: path.join(binDir, 'python'),
  };
};

const ensureVirtualEnvironment = async (basePython, pythonCwd) => {
  const { venvPath, pythonPath } = getVenvLayout(pythonCwd);
  const resolvedVenvPath = path.resolve(venvPath);
  const resolvedPythonPath = path.resolve(pythonPath);

  if (fs.existsSync(resolvedPythonPath)) {
    return resolvedPythonPath;
  }

  console.log(`🐍 Creating isolated virtual environment at ${resolvedVenvPath}`);
  await runCommand(basePython, ['-m', 'venv', resolvedVenvPath], {
    cwd: pythonCwd,
  });

  if (!fs.existsSync(resolvedPythonPath)) {
    throw new Error(`Virtual environment created but interpreter missing at ${resolvedPythonPath}`);
  }

  return resolvedPythonPath;
};

const isVenvPython = (pythonExecutable, pythonCwd) => {
  const { venvPath, binPath } = getVenvLayout(pythonCwd);
  const normalizedExec = path.resolve(pythonExecutable);
  const normalizedVenv = path.resolve(venvPath);
  return normalizedExec.startsWith(normalizedVenv);
};

const detectMissingPythonModules = (pythonExecutable, pythonCwd, modules) => {
  if (!modules || modules.length === 0) {
    return [];
  }

  const script = [
    'import importlib.util, json',
    `modules = ${JSON.stringify(modules)}`,
    'missing = [m for m in modules if importlib.util.find_spec(m) is None]',
    'print(json.dumps(missing))',
  ].join('; ');

  try {
    const result = spawnSync(pythonExecutable, ['-c', script], {
      cwd: pythonCwd,
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    if (result.status === 0) {
      try {
        const parsed = JSON.parse((result.stdout || '').trim() || '[]');
        return Array.isArray(parsed) ? parsed : modules;
      } catch (parseError) {
        console.warn('⚠️ Unable to parse Python dependency check result:', parseError.message);
        return modules;
      }
    }

    if (result.error) {
      console.warn(`⚠️ Dependency check failed for ${pythonExecutable}: ${result.error.message}`);
    } else if (result.stderr) {
      console.warn(`⚠️ Dependency check stderr: ${result.stderr.trim()}`);
    }
  } catch (error) {
    console.warn(`⚠️ Dependency check raised an exception: ${error.message}`);
  }

  return modules;
};

const ensurePythonDependencies = async (pythonExecutable, pythonCwd, options = {}) => {
  const {
    allowSystemPipFallbacks = false,
    importCheckModules = pythonImportChecks,
  } = options;

  if (!autoInstallPythonRequirements) {
    console.log('📦 AUTO_INSTALL_PYTHON_REQUIREMENTS disabled – skipping pip install.');
    return;
  }

  try {
    console.log('📦 Ensuring Python requirements are installed...');

    if (!forcePythonRequirementsInstall && importCheckModules.length > 0) {
      const missingModules = detectMissingPythonModules(pythonExecutable, pythonCwd, importCheckModules);
      if (missingModules.length === 0) {
        console.log('📦 Required Python modules already present – skipping pip install.');
        return;
      }
      console.log(`📦 Missing Python modules detected: ${missingModules.join(', ')}`);
    } else if (forcePythonRequirementsInstall) {
      console.log('📦 FORCE_PYTHON_REQUIREMENTS_INSTALL enabled – skipping dependency check.');
    }

    try {
      await runCommand(pythonExecutable, ['-m', 'pip', 'install', '--upgrade', 'pip'], {
        cwd: pythonCwd,
        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' },
      });
    } catch (upgradeError) {
      console.warn('⚠️ Unable to upgrade pip automatically:', upgradeError.message);
    }

    const installVariants = [{ label: 'default', args: [] }];

    if (allowSystemPipFallbacks) {
      installVariants.push({ label: '--user', args: ['--user'] });
      if (process.platform !== 'win32') {
        installVariants.push({ label: '--break-system-packages', args: ['--break-system-packages'] });
      }
    }

    let installed = false;
    let lastError = null;
    for (const variant of installVariants) {
      try {
        await runCommand(
          pythonExecutable,
          ['-m', 'pip', 'install', '-r', 'requirements.txt', ...variant.args],
          {
            cwd: pythonCwd,
            env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' },
          },
        );
        console.log(`📦 Python dependencies ready (${variant.label}).`);
        installed = true;
        break;
      } catch (installError) {
        lastError = installError;
        console.warn(`⚠️ pip install failed (${variant.label}): ${installError.message}`);
      }
    }

    if (!installed) {
      throw lastError || new Error('pip install failed for all variants');
    }
  } catch (error) {
    console.error('⚠️ Unable to install Python requirements automatically:', error.message);
    console.error('   Tip: install manually using `python -m pip install -r python-backend/requirements.txt` inside a virtual environment (e.g. python3.12).');
  }
};

const startPythonBackend = async () => {
  if (!shouldAutoStartPython) {
    console.log('🐍 AUTO_START_PYTHON disabled – skipping embedded Python backend bootstrap.');
    return null;
  }

  const pythonCwd = process.env.PYTHON_BACKEND_CWD
    ? path.resolve(process.env.PYTHON_BACKEND_CWD)
    : path.join(__dirname, 'python-backend');

  let pythonInfo;
  try {
    pythonInfo = detectPythonExecutable();
  } catch (error) {
    console.error('🐍 Unable to find a usable Python interpreter:', error.message);
    console.error('   Tip: install Python 3.12 (e.g. `brew install python@3.12`) or set PYTHON_BACKEND_EXECUTABLE to a compatible interpreter.');
    return null;
  }

  let allowCreatingVenv = autoCreatePythonVenv;

  if (pythonInfo.version) {
    console.log(`🐍 Selected Python interpreter: ${pythonInfo.command} (${pythonInfo.version})`);
    const versionMatch = pythonInfo.version.match(/Python\s+(\d+)\.(\d+)/i);
    if (versionMatch) {
      const major = Number.parseInt(versionMatch[1], 10);
      const minor = Number.parseInt(versionMatch[2], 10);
      if (major === 3 && minor >= 13) {
        console.warn('⚠️ Detected Python version ≥ 3.13. Some geospatial wheels may be unavailable. Installing python3.12 is recommended.');
        allowCreatingVenv = false;
      }
    }
  } else {
    console.log(`🐍 Selected Python interpreter: ${pythonInfo.command}`);
  }

  const overrideExecutable = process.env.PYTHON_BACKEND_EXECUTABLE;
  const preferExternalInterpreter = Boolean(overrideExecutable && pythonInfo.command === overrideExecutable);
  if (preferExternalInterpreter) {
    console.log('🐍 Using user-specified interpreter from PYTHON_BACKEND_EXECUTABLE.');
    allowCreatingVenv = false;
  }

  if (!forcePythonRequirementsInstall && pythonImportChecks.length > 0 && allowCreatingVenv) {
    const baseMissingModules = detectMissingPythonModules(pythonInfo.command, pythonCwd, pythonImportChecks);
    if (baseMissingModules.length === 0) {
      console.log('📦 Base Python interpreter already satisfies required modules – reusing existing environment.');
      allowCreatingVenv = false;
    }
  }

  const { venvPath, binPath } = getVenvLayout(pythonCwd);
  let runtimePython = pythonInfo.command;
  let usingEmbeddedVenv = false;

  if (allowCreatingVenv) {
    try {
      runtimePython = await ensureVirtualEnvironment(pythonInfo.command, pythonCwd);
      usingEmbeddedVenv = true;
      console.log(`🐍 Using virtual environment interpreter: ${runtimePython}`);
    } catch (venvError) {
      console.warn(`⚠️ Unable to prepare virtual environment automatically: ${venvError.message}`);
      runtimePython = pythonInfo.command;
    }
  } else {
    usingEmbeddedVenv = isVenvPython(runtimePython, pythonCwd);
  }

  const rawArgs = process.env.PYTHON_BACKEND_ARGS;
  const pythonArgs = rawArgs && rawArgs.trim().length > 0
    ? rawArgs.match(/[^\s"']+|"([^"]*)"|'([^']*)'/g)?.map((token) => token.replace(/^['"]|['"]$/g, '')) ?? []
    : ['main.py'];

  const argsPreview = pythonArgs.length > 0 ? pythonArgs.join(' ') : '(no args)';
  console.log(`🐍 Launching Python backend via ${runtimePython} ${argsPreview} (cwd: ${pythonCwd})`);

  await ensurePythonDependencies(runtimePython, pythonCwd, {
    allowSystemPipFallbacks: !usingEmbeddedVenv,
    importCheckModules: pythonImportChecks,
  });

  const child = spawn(runtimePython, pythonArgs, {
    cwd: pythonCwd,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      ...(usingEmbeddedVenv
        ? {
            VIRTUAL_ENV: path.resolve(venvPath),
            PATH: `${path.resolve(binPath)}${path.delimiter}${process.env.PATH || ''}`,
          }
        : {}),
    },
    stdio: ['inherit', 'pipe', 'pipe'],
  });

  child.stdout?.on('data', (data) => {
    process.stdout.write(`🐍 [python] ${data}`);
  });

  child.stderr?.on('data', (data) => {
    process.stderr.write(`🐍 [python][stderr] ${data}`);
  });

  child.on('exit', (code, signal) => {
    console.log(`🐍 Python backend exited with code ${code}${signal ? ` (signal ${signal})` : ''}`);
  });

  child.on('error', (error) => {
    console.error('🐍 Failed to start Python backend:', error);
  });

  return child;
};

const waitForPython = async (url, attempts = 12, delayMs = 1000) => {
  if (!shouldAutoStartPython) {
    return;
  }

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);
      if (response.ok) {
        console.log(`✅ Python backend is reachable at ${url}`);
        return;
      }
      console.warn(`⚠️ Python health check returned ${response.status}. Retrying (${attempt}/${attempts})...`);
    } catch (error) {
      console.warn(`⚠️ Python backend not ready yet (${attempt}/${attempts}):`, error instanceof Error ? error.message : error);
    }
    await delay(delayMs);
  }

  console.warn(`⚠️ Proceeding without Python health confirmation after ${attempts} attempts (${url}).`);
};

// Connect to database
connectDB();

const app = express();

// ==================== MORGAN REQUEST LOGGING ====================
// Custom token for user ID (if available)
morgan.token('user-id', (req) => {
  return req.user?.id || 'anonymous';
});

// Custom token for response time color
morgan.token('colored-status', (req, res) => {
  const status = res.statusCode;
  let color = '\x1b[32m'; // Green for 2xx
  
  if (status >= 500) color = '\x1b[31m'; // Red for 5xx
  else if (status >= 400) color = '\x1b[33m'; // Yellow for 4xx
  else if (status >= 300) color = '\x1b[36m'; // Cyan for 3xx
  
  return `${color}${status}\x1b[0m`;
});

// Development format - detailed logging
const devFormat = ':method :url :colored-status :res[content-length] - :response-time ms - user::user-id';

// Production format - concise logging  
const prodFormat = ':remote-addr - :method :url :colored-status :res[content-length] - :response-time ms';

// Use different formats based on environment
app.use(morgan(process.env.NODE_ENV === 'production' ? prodFormat : devFormat, {
  skip: (req) => req.url === '/api/health' // Skip health checks to reduce noise
}));

// Security Middleware
app.use(helmet());

// CORS - Update to handle credentials properly
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:3000',
  credentials: true, // ✅ Important for cookies
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Cookie']
}));

// Rate limiting - Applied before auth, so it's IP-based
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Increased from 100 to 1000 per window
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true, // Return rate limit info in `RateLimit-*` headers
  legacyHeaders: false, // Disable `X-RateLimit-*` headers
  skip: (req) => {
    // Skip rate limiting for health checks
    return req.url === '/api/health';
  }
});

// Apply rate limiting to all /api routes EXCEPT auth (which has its own limiting)
app.use('/api', limiter);

// Optional: Stricter rate limiting for auth endpoints to prevent brute force
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Only 5 attempts per 15 minutes for login/register
  message: 'Too many login attempts, please try again later.',
  skipSuccessfulRequests: true, // Don't count successful requests
  skipFailedRequests: false // Count failed requests
});

// Apply stricter limiting to auth routes
app.use('/api/auth/login', authLimiter);
app.use('/api/auth/register', authLimiter);

// Body parser middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ✅ ADD COOKIE PARSER MIDDLEWARE (CRITICAL FIX)
app.use(cookieParser());

// Debug middleware to verify cookies are being parsed
// app.use((req, res, next) => {
//   console.log('=== COOKIE DEBUG ===');
//   console.log('Cookies parsed:', req.cookies);
//   console.log('Raw cookie header:', req.headers.cookie);
//   console.log('Path:', req.path);
//   console.log('====================');
//   next();
// });

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/python', pythonProxyRoutes); // Python backend proxy
app.use('/api/history', historyRoutes); // Analysis history

// Health check
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'success',
    message: '🚀 KhananNetra API is running!',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV,
    cookiesEnabled: true // ✅ Confirm cookies are working
  });
});

// 404 handler
app.use((req, res, next) => {
  res.status(404).json({
    status: 'fail',
    message: `Route ${req.originalUrl} not found on this server`
  });
});

// Global error handler
app.use((err, req, res, next) => {
  err.statusCode = err.statusCode || 500;
  err.status = err.status || 'error';

  res.status(err.statusCode).json({
    status: err.status,
    message: err.message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

const PORT = process.env.PORT || 5000;

const bootstrap = async () => {
  const pythonProcess = await startPythonBackend();

  if (pythonProcess) {
    await waitForPython(pythonHealthUrl);
  } else if (shouldAutoStartPython) {
    console.warn('⚠️ Skipping Python health check because the backend failed to start.');
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🎯 KhananNetra Backend Server Started!`);
    console.log(`📍 Port: ${PORT}`);
    console.log(`🌍 Environment: ${process.env.NODE_ENV}`);
    console.log(`🔗 API URL: http://0.0.0.0:${PORT}/api`);
    console.log(`❤️  Health Check: http://0.0.0.0:${PORT}/api/health`);
    console.log(`👤 Super Admin: superadmin@khanannetra.gov.in / Admin@123`);
    console.log(`📊 Morgan Logging: ACTIVE`);
    console.log(`🍪 Cookie Parser: ENABLED\n`);
  });

  const shutdown = async (signal) => {
    console.log(`\n🛑 Received ${signal}. Shutting down gracefully...`);
    server.close(() => {
      console.log('🧹 Express server closed.');
      process.exit(0);
    });

    if (pythonProcess && !pythonProcess.killed) {
      console.log('🐍 Stopping embedded Python backend...');
      pythonProcess.kill('SIGTERM');
    }
  };

  ['SIGINT', 'SIGTERM', 'SIGQUIT'].forEach((signal) => {
    process.on(signal, () => shutdown(signal));
  });

  process.on('exit', () => {
    if (pythonProcess && !pythonProcess.killed) {
      pythonProcess.kill('SIGTERM');
    }
  });
};

bootstrap().catch((error) => {
  console.error('🚨 Failed to bootstrap backend:', error);
  process.exit(1);
});