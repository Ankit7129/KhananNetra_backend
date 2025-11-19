import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import morgan from 'morgan';
import cookieParser from 'cookie-parser'; // ✅ ADD THIS

import connectDB from './config/database.js';

// Route imports
import authRoutes from './routes/auth.js';
import userRoutes from './routes/users.js';
import adminRoutes from './routes/adminRoutes.js';
import pythonProxyRoutes from './routes/pythonProxy.js';
import historyRoutes from './routes/historyRoutes.js';

// Load env vars
dotenv.config();

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

const PORT = process.env.PORT || 8000;  // ✅ Change to 8000

app.listen(PORT, '0.0.0.0', () => {  // ✅ Add '0.0.0.0' for Docker/Cloud Run
    console.log(`\n🎯 KhananNetra Backend Server Started!`);
    console.log(`📍 Port: ${PORT}`);
    console.log(`🌍 Environment: ${process.env.NODE_ENV}`);
    console.log(`🔗 API URL: http://0.0.0.0:${PORT}/api`);
    console.log(`❤️  Health Check: http://0.0.0.0:${PORT}/api/health`);
    console.log(`👤 Super Admin: superadmin@khanannetra.gov.in / Admin@123`);
    console.log(`📊 Morgan Logging: ACTIVE`);
    console.log(`🍪 Cookie Parser: ENABLED\n`);
});