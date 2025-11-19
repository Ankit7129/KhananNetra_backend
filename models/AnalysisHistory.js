/**
 * Analysis History Model
 * Stores complete analysis records with logs, timestamps, and results
 */

import mongoose from 'mongoose';

const TileSchema = new mongoose.Schema({
  tile_id: Number,
  bounds: [[Number]], // Array of coordinate pairs
  mining_detected: Boolean,
  mining_percentage: Number,
  confidence: Number,
  num_mine_blocks: Number,
  total_area_m2: Number,
  image_base64: String,
  probability_map_base64: String,
  mine_blocks: [{
    id: String,
    coordinates: [[Number]],
    area_m2: Number,
    confidence: Number
  }]
}, { _id: false });

const ProcessingLogSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  step: String,
  message: String,
  progress: Number,
  status: String,
  details: mongoose.Schema.Types.Mixed
}, { _id: false });

const AnalysisHistorySchema = new mongoose.Schema({
  // Analysis Identification
  analysisId: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  
  // User Information
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  
  // AOI Information
  aoiId: {
    type: String,
    required: false
  },
  aoiGeometry: {
    type: {
      type: String,
      enum: ['Polygon'],
      required: false
    },
    coordinates: {
      type: [[[Number]]],
      required: false
    }
  },
  aoiArea: {
    km2: Number,
    hectares: Number
  },
  aoiBounds: {
    north: Number,
    south: Number,
    east: Number,
    west: Number
  },
  
  // Analysis Status
  status: {
    type: String,
    enum: ['processing', 'completed', 'failed', 'cancelled'],
    default: 'processing',
    index: true
  },
  
  // Timestamps
  startTime: {
    type: Date,
    required: true,
    default: Date.now,
    index: true
  },
  endTime: Date,
  duration: Number, // Duration in seconds
  
  // Progress Tracking
  currentStep: String,
  progress: {
    type: Number,
    default: 0,
    min: 0,
    max: 100
  },
  
  // Processing Logs
  logs: [ProcessingLogSchema],
  
  // Results
  results: {
    totalTiles: Number,
    tilesProcessed: Number,
    tilesWithMining: Number,
    
    detections: [{
      id: String,
      latitude: Number,
      longitude: Number,
      confidence: Number,
      area_m2: Number,
      bounds: [[Number]]
    }],
    
    detectionCount: Number,
    totalMiningArea: {
      m2: Number,
      hectares: Number,
      km2: Number
    },
    
    // Merged blocks (if polygon merging was performed)
    mergedBlocks: {
      type: mongoose.Schema.Types.Mixed
    },
    
    // Individual tiles data
    tiles: [TileSchema],
    
    // Statistics
    statistics: {
      avgConfidence: Number,
      maxConfidence: Number,
      minConfidence: Number,
      coveragePercentage: Number
    }
  },
  
  // Error Information (if failed)
  error: {
    message: String,
    stack: String,
    timestamp: Date
  },
  
  // Metadata
  metadata: {
    pythonBackendVersion: String,
    mlModelVersion: String,
    earthEngineVersion: String,
    processingServer: String,
    notes: String
  },
  
  // User Actions
  isArchived: {
    type: Boolean,
    default: false
  },
  tags: [String],
  userNotes: String

}, {
  timestamps: true // Adds createdAt and updatedAt automatically
});

// Indexes for efficient querying
AnalysisHistorySchema.index({ userId: 1, startTime: -1 });
AnalysisHistorySchema.index({ status: 1, startTime: -1 });
AnalysisHistorySchema.index({ 'results.detectionCount': -1 });

// Virtual for analysis URL
AnalysisHistorySchema.virtual('viewUrl').get(function() {
  return `/geoanalyst-dashboard/history/${this.analysisId}`;
});

// Method to add log entry
AnalysisHistorySchema.methods.addLog = function(step, message, progress, status = 'info', details = {}) {
  this.logs.push({
    timestamp: new Date(),
    step,
    message,
    progress,
    status,
    details
  });
  return this.save();
};

// Method to mark as completed
AnalysisHistorySchema.methods.markCompleted = function(results) {
  this.status = 'completed';
  this.endTime = new Date();
  this.duration = Math.floor((this.endTime - this.startTime) / 1000);
  this.progress = 100;
  if (results) {
    this.results = { ...this.results, ...results };
  }
  return this.save();
};

// Method to mark as failed
AnalysisHistorySchema.methods.markFailed = function(error) {
  this.status = 'failed';
  this.endTime = new Date();
  this.duration = Math.floor((this.endTime - this.startTime) / 1000);
  this.error = {
    message: error.message || String(error),
    stack: error.stack,
    timestamp: new Date()
  };
  return this.save();
};

// Static method to get user's analysis history
AnalysisHistorySchema.statics.getUserHistory = function(userId, options = {}) {
  const {
    limit = 50,
    skip = 0,
    status,
    sortBy = '-startTime'
  } = options;
  
  const query = { userId };
  if (status) query.status = status;
  
  return this.find(query)
    .sort(sortBy)
    .limit(limit)
    .skip(skip)
    .select('-tiles.image_base64 -tiles.probability_map_base64') // Exclude large data
    .lean();
};

// Static method to get analysis statistics
AnalysisHistorySchema.statics.getUserStats = async function(userId) {
  const result = await this.aggregate([
    { $match: { userId: new mongoose.Types.ObjectId(userId) } },
    {
      $group: {
        _id: null,
        totalAnalyses: { $sum: 1 },
        completedAnalyses: {
          $sum: { $cond: [{ $eq: ['$status', 'completed'] }, 1, 0] }
        },
        failedAnalyses: {
          $sum: { $cond: [{ $eq: ['$status', 'failed'] }, 1, 0] }
        },
        processingAnalyses: {
          $sum: { $cond: [{ $eq: ['$status', 'processing'] }, 1, 0] }
        },
        totalDetections: { $sum: '$results.detectionCount' },
        averageDuration: { $avg: '$duration' }
      }
    }
  ]);
  
  return result[0] || {
    totalAnalyses: 0,
    completedAnalyses: 0,
    failedAnalyses: 0,
    processingAnalyses: 0,
    totalDetections: 0,
    averageDuration: 0
  };
};

const AnalysisHistory = mongoose.model('AnalysisHistory', AnalysisHistorySchema);

export default AnalysisHistory;
