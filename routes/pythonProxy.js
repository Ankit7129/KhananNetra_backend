/**
 * Python Backend Proxy Routes
 * Bridges Node.js Express with Python FastAPI for geospatial analysis
 */

import express from 'express';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import AnalysisHistory from '../models/AnalysisHistory.js';
import { protect } from '../middleware/auth.js';
const shouldBypassAuth = () => (
  process.env.NODE_ENV !== 'production' && process.env.DEV_ALLOW_OPEN_PYTHON_PROXY === 'true'
);

const optionalProtect = (req, res, next) => {
  if (shouldBypassAuth()) {
    console.warn('⚠️  DEV_ALLOW_OPEN_PYTHON_PROXY enabled - bypassing auth for python proxy route');
    return next();
  }
  return protect(req, res, next);
};

const router = express.Router();

// Store uploads in memory so we can forward buffers to Python backend
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50 MB safety limit
  },
});

// Python backend URL - uses environment variable with fallback
// For local development: http://localhost:8000 (FastAPI default)
// For Docker: http://python-backend:8001 (service name)
const getPythonBackendURL = () => {
  // Explicit environment variable takes precedence
  if (process.env.PYTHON_BACKEND_URL) {
    return process.env.PYTHON_BACKEND_URL;
  }
  if (process.env.PYTHON_API_URL) {
    return process.env.PYTHON_API_URL;
  }
  
  // Default based on environment
  const isDocker = process.env.DOCKER_ENV === 'true' || process.env.NODE_ENV === 'production';
  return isDocker ? 'http://python-backend:8001' : 'http://localhost:8000';
};

const PYTHON_API_URL = getPythonBackendURL();

console.log(`🐍 Python Backend URL configured: ${PYTHON_API_URL}`);
console.log(`🔍 Environment: ${process.env.NODE_ENV || 'development'}`);

/**
 * Health check for Python backend
 */
router.get('/health', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/health`);
    res.json({
      status: 'ok',
      pythonBackend: response.data,
      message: 'Python backend is reachable'
    });
  } catch (error) {
    res.status(503).json({
      status: 'error',
      message: 'Python backend is not available',
      error: error.message
    });
  }
});

/**
 * Create AOI from geometry
 * POST /api/python/aoi/create
 */
router.post('/aoi/create', async (req, res) => {
  try {
    const { geometry, properties } = req.body;

    if (!geometry || !geometry.type || !geometry.coordinates) {
      return res.status(400).json({
        error: 'Invalid geometry. Must include type and coordinates'
      });
    }

    console.log('📍 Creating AOI:', { 
      type: geometry.type, 
      points: geometry.coordinates[0]?.length 
    });

    const response = await axios.post(
      `${PYTHON_API_URL}/api/v1/aoi/create`,
      { geometry, properties },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 30000 // 30 seconds
      }
    );

    console.log('✅ AOI created successfully:', response.data.id);

    res.json(response.data);
  } catch (error) {
    console.error('❌ AOI creation error:', error.message);
    
    if (error.response) {
      console.error('Python backend error details:', error.response.data);
      res.status(error.response.status).json({
        error: error.response.data.detail || 'Failed to create AOI',
        details: error.response.data
      });
    } else {
      res.status(500).json({
        error: 'Failed to communicate with Python backend',
        message: error.message
      });
    }
  }
});

/**
 * Upload AOI file (KML, GeoJSON, Shapefile)
 * POST /api/python/aoi/upload
 */
router.post('/aoi/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'File upload is required. Expected field name "file".',
      });
    }

    // Forward multipart form data to Python backend
    const formData = new FormData();

    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // Include any additional form fields
    Object.entries(req.body || {}).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((item) => formData.append(key, item));
      } else if (typeof value === 'object' && value !== null) {
        formData.append(key, JSON.stringify(value));
      } else if (value !== undefined) {
        formData.append(key, value);
      }
    });

    const response = await axios.post(
      `${PYTHON_API_URL}/api/v1/aoi/upload`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 60000 // 60 seconds for file upload
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('❌ File upload error:', error.message);
    
    if (error.response) {
      res.status(error.response.status).json({
        error: error.response.data.detail || 'Failed to upload file',
        details: error.response.data
      });
    } else {
      res.status(500).json({
        error: 'Failed to upload file',
        message: error.message
      });
    }
  }
});

/**
 * Start analysis
 * POST /api/python/analysis/start
 */
router.post('/analysis/start', optionalProtect, async (req, res) => {
  try {
    const { aoi_id, geometry } = req.body;
    const userId = req.user?._id || null; // From auth middleware when available

    if (!aoi_id) {
      return res.status(400).json({
        error: 'aoi_id is required'
      });
    }

    console.log('🚀 Starting analysis for AOI:', aoi_id, 'User:', userId);

    const response = await axios.post(
      `${PYTHON_API_URL}/api/v1/analysis/start`,
      { aoi_id },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 120000 // 2 minutes for analysis start
      }
    );

    console.log('✅ Analysis started:', response.data.analysis_id);

    // Create analysis history record
    if (userId) {
      try {
        const historyRecord = new AnalysisHistory({
          analysisId: response.data.analysis_id,
          userId,
          aoiId: aoi_id,
          aoiGeometry: geometry || {
            type: 'Polygon',
            coordinates: []
          },
          status: 'processing',
          startTime: new Date()
        });

        await historyRecord.save();
        console.log('📝 Analysis history record created:', response.data.analysis_id);
      } catch (historyError) {
        console.error('⚠️ Failed to create history record:', historyError.message);
        // Don't fail the request if history saving fails
      }
    }

    res.json(response.data);
  } catch (error) {
    console.error('❌ Analysis start error:', error.message);
    
    if (error.response) {
      res.status(error.response.status).json({
        error: error.response.data.detail || 'Failed to start analysis',
        details: error.response.data
      });
    } else {
      res.status(500).json({
        error: 'Failed to start analysis',
        message: error.message
      });
    }
  }
});

/**
 * Get analysis status
 * GET /api/python/analysis/:analysisId
 */
router.get('/analysis/:analysisId', optionalProtect, async (req, res) => {
  try {
    const { analysisId } = req.params;
    const userId = req.user?._id;
    
    console.log(`📊 Fetching analysis status for: ${analysisId}`);
    console.log(`🔗 Python API URL: ${PYTHON_API_URL}/api/v1/analysis/${analysisId}`);

    const response = await axios.get(
      `${PYTHON_API_URL}/api/v1/analysis/${analysisId}`,
      {
        timeout: 5000, // 5 seconds - status check should be fast
        validateStatus: function (status) {
          return status < 500; // Resolve only if status < 500
        }
      }
    );

    console.log(`✅ Analysis status retrieved: ${response.status}`);
    
    // Update history record with latest status
    if (userId) {
      try {
        const historyRecord = await AnalysisHistory.findOne({ analysisId, userId });
        
        if (historyRecord && response.data) {
          const data = response.data;
          
          if (data.status && data.status !== historyRecord.status) {
            historyRecord.status = data.status;
          }
          
          if (data.message) {
            await historyRecord.addLog(
              data.status || 'processing',
              data.message,
              data.progress || 0,
              'info'
            );
          }
          
          if (data.status === 'completed' && data.results) {
            await historyRecord.markCompleted(data.results);
          } else if (data.status === 'failed' && data.error) {
            await historyRecord.markFailed(data.error);
          } else {
            await historyRecord.save();
          }
          
          console.log('📝 History record updated:', analysisId);
        }
      } catch (historyError) {
        console.error('⚠️ Failed to update history record:', historyError.message);
        // Don't fail the request if history update fails
      }
    }
    
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error('❌ Analysis status error:', error.message);
    console.error('   Python Backend URL:', PYTHON_API_URL);
    console.error('   Error code:', error.code);
    
    if (error.code === 'ECONNABORTED') {
      console.error('   Timeout - Analysis endpoint took too long to respond');
      res.status(504).json({
        error: 'Gateway timeout',
        message: 'Analysis status check timed out. The Python backend may be overloaded.',
        pythonBackendUrl: PYTHON_API_URL
      });
    } else if (error.code === 'ECONNREFUSED') {
      console.error(`   ❌ Connection refused - Python backend not running at ${PYTHON_API_URL}`);
      res.status(503).json({
        error: 'Python backend is not running',
        message: `Failed to connect to Python backend at ${PYTHON_API_URL}. Please ensure:
          1. Python backend is running (python main.py)
          2. For local dev: http://localhost:8000 should be accessible
          3. For Docker: python-backend service should be running
          4. Firewall/network allows connection`,
        pythonBackendUrl: PYTHON_API_URL
      });
    } else if (error.code === 'ECONNRESET') {
      console.error('   ❌ Connection reset - Python backend crashed or connection interrupted');
      res.status(503).json({
        error: 'Python backend connection reset',
        message: `Connection to Python backend at ${PYTHON_API_URL} was reset. The backend may have crashed or restarted.`,
        pythonBackendUrl: PYTHON_API_URL
      });
    } else if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', error.response.data);
      res.status(error.response.status).json({
        error: error.response.data.detail || 'Failed to get analysis status',
        details: error.response.data,
        pythonBackendUrl: PYTHON_API_URL
      });
    } else {
      res.status(500).json({
        error: 'Failed to get analysis status',
        message: error.message,
        pythonBackendUrl: PYTHON_API_URL
      });
    }
  }
});

/**
 * Get imagery for AOI
 * POST /api/python/imagery/fetch
 */
router.post('/imagery/fetch', async (req, res) => {
  try {
    const { aoi_id, start_date, end_date } = req.body;

    console.log('🛰️ Fetching imagery for AOI:', aoi_id);

    const response = await axios.post(
      `${PYTHON_API_URL}/api/v1/imagery/fetch`,
      { aoi_id, start_date, end_date },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 120000 // 2 minutes
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('❌ Imagery fetch error:', error.message);
    
    if (error.response) {
      res.status(error.response.status).json({
        error: error.response.data.detail || 'Failed to fetch imagery',
        details: error.response.data
      });
    } else {
      res.status(500).json({
        error: 'Failed to fetch imagery',
        message: error.message
      });
    }
  }
});

/**
 * Stop/Cancel an ongoing analysis
 * POST /api/python/analysis/:analysisId/stop
 */
router.post('/analysis/:analysisId/stop', optionalProtect, async (req, res) => {
  try {
    const { analysisId } = req.params;
    const userId = req.user?._id;

    console.log(`🛑 Stopping analysis: ${analysisId}`);

    // Update history record to mark as cancelled
    let historyRecord = null;
    try {
      historyRecord = await AnalysisHistory.findOne({ analysisId, userId });
      if (!historyRecord || !userId) {
        historyRecord.status = 'cancelled';
        historyRecord.endTime = new Date();
        historyRecord.duration = Math.floor((historyRecord.endTime - historyRecord.startTime) / 1000);
        await historyRecord.addLog('cancelled', 'Analysis stopped by user', 0, 'warning');
        await historyRecord.save();
        console.log(`✅ Analysis marked as cancelled in database: ${analysisId}`);
      }
    } catch (historyError) {
      console.error('⚠️ Failed to update history record:', historyError.message);
    }

    // Try to call Python backend to stop analysis (optional, may not be supported)
    try {
      await axios.post(
        `${PYTHON_API_URL}/api/v1/analysis/${analysisId}/stop`,
        {},
        {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 5000
        }
      );
      console.log(`✅ Python backend stop signal sent: ${analysisId}`);
    } catch (pythonError) {
      // Python backend may not support stop endpoint - that's okay, we've already marked it as cancelled
      if (pythonError.response?.status === 404) {
        console.log(`⚠️ Python backend doesn't have stop endpoint - analysis marked as cancelled in DB`);
      } else {
        console.warn('⚠️ Warning calling Python stop endpoint:', pythonError.message);
      }
    }

    // Return success since we've updated the history
    res.json({ 
      success: true, 
      message: 'Analysis stopped successfully',
      analysisId,
      status: 'cancelled'
    });
  } catch (error) {
    console.error('❌ Stop analysis error:', error.message);
    res.status(500).json({
      error: 'Failed to stop analysis',
      message: error.message
    });
  }
});

export default router;
