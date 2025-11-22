/**
 * Analysis History Routes
 * Manage analysis records - create, read, update, delete
 */

import express from 'express';
import axios from 'axios';
import AnalysisHistory from '../models/AnalysisHistory.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

const shouldBypassAuth = () => (
  process.env.NODE_ENV !== 'production' && process.env.DEV_ALLOW_OPEN_PYTHON_PROXY === 'true'
);

const getPythonBackendURL = () => {
  if (process.env.PYTHON_BACKEND_URL) return process.env.PYTHON_BACKEND_URL;
  if (process.env.PYTHON_API_URL) return process.env.PYTHON_API_URL;
  const isDocker = process.env.DOCKER_ENV === 'true' || process.env.NODE_ENV === 'production';
  return isDocker ? 'http://python-backend:8001' : 'http://localhost:8000';
};

const PYTHON_API_URL = getPythonBackendURL();

const optionalProtect = (req, res, next) => {
  if (shouldBypassAuth()) {
    console.warn('⚠️  DEV_ALLOW_OPEN_PYTHON_PROXY enabled - bypassing auth for history route');
    return next();
  }
  return protect(req, res, next);
};

router.use(optionalProtect);

const DEV_HISTORY_LIMIT = Number.parseInt(process.env.DEV_HISTORY_LIMIT || '20', 10);
const devHistoryStore = new Map();

const getDevHistoryArray = () => Array.from(devHistoryStore.values()).sort((a, b) => {
  const aTime = new Date(a.createdAt || a.startTime || 0).getTime();
  const bTime = new Date(b.createdAt || b.startTime || 0).getTime();
  return bTime - aTime;
});

const saveDevHistoryRecord = (record) => {
  if (!record?.analysisId) {
    return;
  }

  const nowIso = new Date().toISOString();
  const normalized = {
    ...record,
    createdAt: record.createdAt || nowIso,
    updatedAt: nowIso,
    startTime: record.startTime || record.metadata?.startTime || nowIso
  };

  devHistoryStore.set(record.analysisId, normalized);

  const maxEntries = Number.isFinite(DEV_HISTORY_LIMIT) && DEV_HISTORY_LIMIT > 0 ? DEV_HISTORY_LIMIT : 20;
  while (devHistoryStore.size > maxEntries) {
    const oldest = getDevHistoryArray().pop();
    if (!oldest) break;
    devHistoryStore.delete(oldest.analysisId);
  }
};

const parseTimestampValue = (value) => {
  if (value === null || value === undefined) {
    return null;
  }

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    const milliseconds = value > 1e12 ? value : value * 1000;
    const dateFromNumber = new Date(milliseconds);
    return Number.isNaN(dateFromNumber.getTime()) ? null : dateFromNumber;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }

    const numericCandidate = Number(trimmed);
    if (Number.isFinite(numericCandidate)) {
      const milliseconds = numericCandidate > 1e12 ? numericCandidate : numericCandidate * 1000;
      const numericDate = new Date(milliseconds);
      if (!Number.isNaN(numericDate.getTime())) {
        return numericDate;
      }
    }

    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  return null;
};

const coerceSecondsValue = (value, divisor = 1) => {
  if (value === null || value === undefined) {
    return null;
  }

  const normalize = (candidate) => {
    if (!Number.isFinite(candidate)) {
      return null;
    }
    const scaled = divisor !== 0 ? candidate / divisor : candidate;
    return Number.isFinite(scaled) && scaled >= 0 ? scaled : null;
  };

  if (typeof value === 'number') {
    return normalize(value);
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = Number.parseFloat(trimmed.replace(/,/g, ''));
    return normalize(parsed);
  }

  return null;
};

const getValueByPath = (source, path) => {
  if (!source || typeof source !== 'object') {
    return undefined;
  }

  return path.split('.').reduce((acc, key) => {
    if (acc && typeof acc === 'object' && Object.prototype.hasOwnProperty.call(acc, key)) {
      return acc[key];
    }
    return undefined;
  }, source);
};

const deriveTimingInfo = (resultsPayload = {}, metadataPayload = {}) => {
  const results = resultsPayload && typeof resultsPayload === 'object' ? resultsPayload : {};
  const metadata = metadataPayload && typeof metadataPayload === 'object' ? metadataPayload : {};
  const summary = results.summary && typeof results.summary === 'object' ? results.summary : {};
  const statistics = results.statistics && typeof results.statistics === 'object' ? results.statistics : {};

  const context = { metadata, results, summary, statistics };

  const pickFirstTimestamp = (paths) => {
    for (const path of paths) {
      const candidateDate = parseTimestampValue(getValueByPath(context, path));
      if (candidateDate) {
        return candidateDate;
      }
    }
    return null;
  };

  const pickFirstDuration = (specs) => {
    for (const [path, divisor = 1] of specs) {
      const candidate = coerceSecondsValue(getValueByPath(context, path), divisor);
      if (candidate !== null && candidate !== undefined) {
        return candidate;
      }
    }
    return null;
  };

  const startPaths = [
    'metadata.startTime',
    'metadata.start_time',
    'metadata.startedAt',
    'metadata.started_at',
    'metadata.analysisStart',
    'metadata.analysis_start',
    'metadata.analysisStartedAt',
    'metadata.analysis_started_at',
    'metadata.timing.startTime',
    'metadata.timing.start_time',
    'metadata.timing.startedAt',
    'metadata.timing.started_at',
    'metadata.timeline.start',
    'metadata.timeline.startedAt',
    'metadata.timeline.started_at',
    'metadata.runtime.start',
    'results.startTime',
    'results.start_time',
    'results.startedAt',
    'results.started_at',
    'results.createdAt',
    'results.created_at',
    'summary.startTime',
    'summary.start_time',
    'summary.startedAt',
    'summary.started_at',
    'summary.analysisStartTime',
    'statistics.startTime',
    'statistics.start_time',
    'statistics.startedAt',
    'statistics.started_at',
    'statistics.processingStartedAt',
    'statistics.processing_started_at'
  ];

  const endPaths = [
    'metadata.completedAt',
    'metadata.completed_at',
    'metadata.endTime',
    'metadata.end_time',
    'metadata.finishedAt',
    'metadata.finished_at',
    'metadata.analysisCompletedAt',
    'metadata.analysis_completed_at',
    'metadata.analysisEnd',
    'metadata.analysis_end',
    'metadata.timing.endTime',
    'metadata.timing.end_time',
    'metadata.timing.completedAt',
    'metadata.timing.completed_at',
    'metadata.timeline.end',
    'metadata.timeline.completedAt',
    'metadata.timeline.completed_at',
    'metadata.runtime.end',
    'results.endTime',
    'results.end_time',
    'results.completedAt',
    'results.completed_at',
    'results.finishedAt',
    'results.finished_at',
    'summary.endTime',
    'summary.end_time',
    'summary.completedAt',
    'summary.completed_at',
    'summary.finishedAt',
    'summary.finished_at',
    'summary.analysisCompletedAt',
    'statistics.endTime',
    'statistics.end_time',
    'statistics.completedAt',
    'statistics.completed_at',
    'statistics.finishedAt',
    'statistics.finished_at'
  ];

  const durationSpecs = [
    ['metadata.processingTimeSeconds'],
    ['metadata.processing_time_seconds'],
    ['metadata.processingTimeMs', 1000],
    ['metadata.processing_time_ms', 1000],
    ['metadata.runtimeSeconds'],
    ['metadata.runtime_seconds'],
    ['metadata.runtimeMs', 1000],
    ['metadata.runtime_ms', 1000],
    ['metadata.analysisDurationSeconds'],
    ['metadata.analysis_duration_seconds'],
    ['metadata.analysisDurationMs', 1000],
    ['metadata.analysis_duration_ms', 1000],
    ['metadata.durationSeconds'],
    ['metadata.duration_seconds'],
    ['metadata.durationMs', 1000],
    ['metadata.duration_ms', 1000],
    ['results.durationSeconds'],
    ['results.duration_seconds'],
    ['results.runtimeSeconds'],
    ['results.runtime_seconds'],
    ['results.processingTimeSeconds'],
    ['results.processing_time_seconds'],
    ['results.processingTimeMs', 1000],
    ['results.processing_time_ms', 1000],
    ['summary.durationSeconds'],
    ['summary.duration_seconds'],
    ['summary.runtimeSeconds'],
    ['summary.runtime_seconds'],
    ['summary.runtimeMs', 1000],
    ['summary.runtime_ms', 1000],
    ['summary.processingSeconds'],
    ['summary.processing_seconds'],
    ['summary.processingMs', 1000],
    ['summary.processing_ms', 1000],
    ['statistics.durationSeconds'],
    ['statistics.duration_seconds'],
    ['statistics.runtimeSeconds'],
    ['statistics.runtime_seconds'],
    ['statistics.runtimeMs', 1000],
    ['statistics.runtime_ms', 1000],
    ['statistics.processingTimeSeconds'],
    ['statistics.processing_time_seconds'],
    ['statistics.processingTimeMs', 1000],
    ['statistics.processing_time_ms', 1000]
  ];

  let startTime = pickFirstTimestamp(startPaths);
  let endTime = pickFirstTimestamp(endPaths);
  let durationSeconds = pickFirstDuration(durationSpecs);

  if (startTime && endTime) {
    const diffSeconds = (endTime.getTime() - startTime.getTime()) / 1000;
    if (Number.isFinite(diffSeconds) && diffSeconds >= 0.5) {
      durationSeconds = Math.round(diffSeconds);
    } else if (durationSeconds && diffSeconds < 0) {
      endTime = new Date(startTime.getTime() + Math.round(durationSeconds) * 1000);
    }
  }

  if (!startTime && endTime && durationSeconds && durationSeconds > 0) {
    startTime = new Date(endTime.getTime() - Math.round(durationSeconds) * 1000);
  }

  if (startTime && !endTime && durationSeconds && durationSeconds > 0) {
    endTime = new Date(startTime.getTime() + Math.round(durationSeconds) * 1000);
  }

  if (!startTime) {
    startTime = new Date();
  }

  if (!endTime) {
    endTime = durationSeconds && durationSeconds > 0
      ? new Date(startTime.getTime() + Math.round(durationSeconds) * 1000)
      : new Date(startTime.getTime());
  }

  if (durationSeconds === null || durationSeconds === undefined || durationSeconds < 0) {
    const diffSeconds = (endTime.getTime() - startTime.getTime()) / 1000;
    durationSeconds = Number.isFinite(diffSeconds) && diffSeconds >= 0 ? Math.round(diffSeconds) : 0;
  }

  return {
    startTime,
    endTime,
    durationSeconds: Number.isFinite(durationSeconds) && durationSeconds >= 0 ? Math.round(durationSeconds) : 0
  };
};

const sanitizeTileImages = (tiles = [], includeImages = false) => {
  if (includeImages) {
    return tiles;
  }
  return tiles.map((tile) => {
    const {
      image_base64,
      probability_map_base64,
      thumbnail,
      ...rest
    } = tile;
    return rest;
  });
};

/**
 * GET /api/history
 * Get user's analysis history with pagination and filtering
 */
router.get('/', async (req, res) => {
  try {
    const {
      page = 1,
      limit = 20,
      status,
      sortBy = 'startTime',
      sortOrder = 'desc',
      search
    } = req.query;

    const pageInt = Number.parseInt(page, 10) || 1;
    const limitInt = Number.parseInt(limit, 10) || 20;
    const sortDirection = sortOrder === 'asc' ? 1 : -1;

    if (!req.user && shouldBypassAuth()) {
      console.warn('⚠️  History list requested without auth - using in-memory dev store');
      let records = getDevHistoryArray();

      if (status) {
        records = records.filter((rec) => rec.status === status);
      }
      if (search) {
        const regex = new RegExp(search, 'i');
        records = records.filter((rec) => regex.test(rec.analysisId) || regex.test(rec.aoiId || ''));
      }

      const deriveSortValue = (rec) => {
        switch (sortBy) {
          case 'duration':
            return rec.duration ?? rec.metadata?.processingTimeMs ?? 0;
          case 'detectionCount':
            return rec.results?.detectionCount ?? rec.results?.mine_block_count ?? 0;
          case 'startTime':
          default:
            return new Date(rec.startTime || rec.createdAt || 0).getTime();
        }
      };

      records.sort((a, b) => {
        const aValue = deriveSortValue(a);
        const bValue = deriveSortValue(b);
        if (aValue === bValue) return 0;
        return aValue > bValue ? sortDirection : -sortDirection;
      });

      const total = records.length;
      const pages = Math.ceil(total / limitInt);
      const offset = (pageInt - 1) * limitInt;
      const paged = records
        .slice(offset, offset + limitInt)
        .map((rec) => ({
          ...rec,
          results: rec.results ? {
            ...rec.results,
            tiles: sanitizeTileImages(rec.results.tiles || [], false)
          } : rec.results
        }));

      return res.json({
        analyses: paged,
        pagination: {
          page: pageInt,
          limit: limitInt,
          total,
          pages
        }
      });
    }

    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const skip = (pageInt - 1) * limitInt;
    const query = { userId: req.user._id };

    if (status) query.status = status;
    if (search) {
      query.$or = [
        { analysisId: { $regex: search, $options: 'i' } },
        { aoiId: { $regex: search, $options: 'i' } },
        { tags: { $in: [new RegExp(search, 'i')] } }
      ];
    }

    const sortMap = {
      startTime: 'startTime',
      duration: 'duration',
      detectionCount: 'results.detectionCount'
    };

    const sortField = sortMap[sortBy] || 'startTime';
    const sortObj = { [sortField]: sortDirection };

    const total = await AnalysisHistory.countDocuments(query);

    const analyses = await AnalysisHistory.find(query)
      .sort(sortObj)
      .limit(limitInt)
      .skip(skip)
      .select('-tiles.image_base64 -tiles.probability_map_base64 -logs')
      .lean();

    res.json({
      analyses,
      pagination: {
        page: pageInt,
        limit: limitInt,
        total,
        pages: Math.ceil(total / limitInt)
      }
    });
  } catch (error) {
    console.error('❌ Error fetching analysis history:', error);
    res.status(500).json({
      error: 'Failed to fetch analysis history',
      message: error.message
    });
  }
});

/**
 * GET /api/history/stats
 * Get user's analysis statistics
 */
router.get('/stats', async (req, res) => {
  try {
    if (!req.user && shouldBypassAuth()) {
      console.warn('⚠️  History stats requested without auth - calculating from in-memory dev store');
      const records = getDevHistoryArray();
      const totals = records.reduce((acc, rec) => {
        acc.totalAnalyses += 1;
        if (rec.status === 'completed') acc.completedAnalyses += 1;
        if (rec.status === 'failed') acc.failedAnalyses += 1;
        if (rec.status === 'processing') acc.processingAnalyses += 1;

        const detectionCount = rec.results?.detectionCount
          ?? rec.results?.mine_block_count
          ?? rec.results?.totalMiningArea?.detections
          ?? 0;
        acc.totalDetections += detectionCount;

        const miningHa = rec.results?.totalMiningArea?.hectares
          ?? rec.results?.mining_area_ha
          ?? 0;
        acc.totalMiningAreaHa += miningHa;

        const startTimestamp = new Date(rec.startTime || rec.createdAt || 0).getTime();
        if (startTimestamp > acc.lastAnalysisTimestamp) {
          acc.lastAnalysisTimestamp = startTimestamp;
        }

        const durationSeconds = (() => {
          if (typeof rec.duration === 'number' && Number.isFinite(rec.duration) && rec.duration > 0) {
            return rec.duration;
          }

          const coerceSeconds = (value, divisor = 1) => {
            if (value === undefined || value === null) return null;
            const numeric = typeof value === 'string' ? Number.parseFloat(value) : Number(value);
            if (!Number.isFinite(numeric) || numeric <= 0) return null;
            return numeric / divisor;
          };

          const fallback = [
            coerceSeconds(rec.metadata?.processingTimeSeconds),
            coerceSeconds(rec.metadata?.processingTimeMs, 1000),
            coerceSeconds(rec.metadata?.runtimeSeconds),
            coerceSeconds(rec.metadata?.runtimeMs, 1000),
            coerceSeconds(rec.results?.processing_time_seconds),
            coerceSeconds(rec.results?.processing_time_ms, 1000),
            coerceSeconds(rec.results?.runtime_seconds),
            coerceSeconds(rec.results?.runtime_ms, 1000),
            coerceSeconds(rec.results?.summary?.runtime_seconds),
            coerceSeconds(rec.results?.summary?.processing_seconds),
            coerceSeconds(rec.results?.summary?.runtime_ms, 1000),
            coerceSeconds(rec.results?.statistics?.runtimeSeconds),
            coerceSeconds(rec.results?.statistics?.processingTimeSeconds),
            coerceSeconds(rec.results?.statistics?.runtimeMs, 1000)
          ].find((seconds) => seconds !== null);

          if (fallback) {
            return fallback;
          }

          const start = new Date(rec.startTime || rec.createdAt || 0);
          const end = new Date(rec.endTime || rec.completedAt || rec.updatedAt || start);
          const diff = (end.getTime() - start.getTime()) / 1000;
          return Number.isFinite(diff) && diff >= 0 ? diff : 0;
        })();

        if (durationSeconds > 0) {
          acc.durationSum += durationSeconds;
          acc.durationCount += 1;
        }

        return acc;
      }, {
        totalAnalyses: 0,
        completedAnalyses: 0,
        failedAnalyses: 0,
        processingAnalyses: 0,
        totalDetections: 0,
        totalMiningAreaHa: 0,
        lastAnalysisTimestamp: 0,
        durationSum: 0,
        durationCount: 0
      });

      const averageDuration = totals.durationCount > 0
        ? totals.durationSum / totals.durationCount
        : 0;

      return res.json({
        totalAnalyses: totals.totalAnalyses,
        completedAnalyses: totals.completedAnalyses,
        failedAnalyses: totals.failedAnalyses,
        processingAnalyses: totals.processingAnalyses,
        totalDetections: totals.totalDetections,
        averageDuration,
        lastAnalysisDate: totals.lastAnalysisTimestamp
          ? new Date(totals.lastAnalysisTimestamp).toISOString()
          : null
      });
    }

    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const stats = await AnalysisHistory.getUserStats(req.user._id);
    res.json(stats);
  } catch (error) {
    console.error('❌ Error fetching statistics:', error);
    res.status(500).json({
      error: 'Failed to fetch statistics',
      message: error.message
    });
  }
});

/**
 * GET /api/history/:analysisId
 * Get detailed analysis record including logs
 */
router.get('/:analysisId', async (req, res) => {
  try {
    const { analysisId } = req.params;
    const { includeTileImages = 'false' } = req.query;

    if (!req.user && shouldBypassAuth()) {
      console.warn(`⚠️  History detail requested without auth - serving dev data for ${analysisId}`);
      const stored = devHistoryStore.get(analysisId);
      if (stored) {
        return res.json({
          ...stored,
          results: stored.results ? {
            ...stored.results,
            tiles: sanitizeTileImages(
              stored.results.tiles || [],
              includeTileImages === 'true'
            )
          } : stored.results
        });
      }

      try {
        const response = await axios.get(`${PYTHON_API_URL}/api/v1/analysis/${analysisId}`, {
          params: { includeTileImages }
        });
        return res.json(response.data);
      } catch (proxyError) {
        console.error('❌ Failed to proxy analysis detail:', proxyError.message);
        const status = proxyError.response?.status || 404;
        return res.status(status).json({
          error: proxyError.response?.data || 'Analysis not found'
        });
      }
    }

    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    console.log(`\n🔍 Fetching analysis from database: ${analysisId}`);
    console.log(`   └─ User ID: ${req.user._id}`);
    console.log(`   └─ Include tile images: ${includeTileImages}`);

    let selectFields = '-tiles.image_base64 -tiles.probability_map_base64';
    if (includeTileImages === 'true') {
      selectFields = '';
    }

    const analysis = await AnalysisHistory.findOne({
      analysisId,
      userId: req.user._id
    }).select(selectFields);

    if (!analysis) {
      console.log(`❌ Analysis not found: ${analysisId}`);
      return res.status(404).json({
        error: 'Analysis not found'
      });
    }

    console.log(`✅ Analysis found in database`);
    console.log(`   └─ Status: ${analysis.status}`);
    console.log(`   └─ Total tiles: ${analysis.results?.totalTiles || 0}`);
    console.log(`   └─ Detection count: ${analysis.results?.detectionCount || 0}`);
    console.log(`   └─ Created: ${analysis.createdAt}`);

    res.json(analysis);
  } catch (error) {
    console.error('❌ Error fetching analysis:', error);
    res.status(500).json({
      error: 'Failed to fetch analysis',
      message: error.message
    });
  }
});

/**
 * POST /api/history
 * Save a new analysis record to database
 */
router.post('/', async (req, res) => {
  try {
    const {
      analysisId,
      aoiGeometry,
      aoiBounds,
      results,
      logs,
      metadata,
      force = false  // If true, allows overwriting existing analysis
    } = req.body;

    if (!req.user && shouldBypassAuth()) {
      console.warn('⚠️  Save history requested without auth - storing in in-memory dev history store');
      const rawResults = results?.results && typeof results.results === 'object'
        ? results.results
        : results || {};

      const { startTime, endTime, durationSeconds } = deriveTimingInfo(rawResults, metadata);

      const devRecord = {
        analysisId,
        aoiGeometry,
        aoiBounds,
        results,
        logs,
        metadata,
        status: rawResults?.status || results?.status || 'completed',
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
        completedAt: metadata?.completedAt
          || rawResults?.completed_at
          || rawResults?.completedAt
          || endTime.toISOString(),
        duration: durationSeconds,
        progress: typeof rawResults?.progress === 'number'
          ? rawResults.progress
          : typeof results?.progress === 'number'
            ? results.progress
            : 100,
        currentStep: rawResults?.current_step
          || rawResults?.currentStep
          || results?.current_step
          || results?.currentStep
          || 'completed',
        userId: 'dev-user',
        source: 'dev-memory-store'
      };

      saveDevHistoryRecord(devRecord);

      return res.status(201).json({
        status: 'stored',
        message: 'Analysis saved to in-memory dev history store',
        analysis: devRecord
      });
    }

    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    console.log('\n📝 ==================== SAVE ANALYSIS REQUEST ====================');
    console.log(`📋 Analysis ID: ${analysisId}`);
    console.log(`👤 User ID: ${req.user._id}`);
    
    // Normalize result structure
    const rawResults = results?.results && typeof results.results === 'object'
      ? results.results
      : results || {};

    const summary = rawResults.summary && typeof rawResults.summary === 'object'
      ? rawResults.summary
      : results?.summary && typeof results.summary === 'object'
        ? results.summary
        : {};

    const tiles = Array.isArray(rawResults.tiles)
      ? rawResults.tiles
      : Array.isArray(results?.tiles)
        ? results.tiles
        : [];

    const detections = Array.isArray(rawResults.detections)
      ? rawResults.detections
      : Array.isArray(results?.detections)
        ? results.detections
        : [];

    const totalTiles = Number.isFinite(rawResults.totalTiles)
      ? Number(rawResults.totalTiles)
      : Number.isFinite(summary.total_tiles)
        ? Number(summary.total_tiles)
        : tiles.length;

    const tilesWithMining = Number.isFinite(rawResults.tilesWithMining)
      ? Number(rawResults.tilesWithMining)
      : Number.isFinite(summary.tiles_with_detections)
        ? Number(summary.tiles_with_detections)
        : tiles.filter((tile) => tile.mining_detected || tile.miningDetected).length;

    const detectionCount = Number.isFinite(rawResults.detectionCount)
      ? Number(rawResults.detectionCount)
      : Number.isFinite(summary.mine_block_count)
        ? Number(summary.mine_block_count)
        : detections.length;

    const mergedBlocks = rawResults.mergedBlocks
      ?? results?.mergedBlocks
      ?? results?.merged_blocks
      ?? null;

    const totalMiningAreaM2 = (() => {
      if (rawResults.totalMiningArea && typeof rawResults.totalMiningArea === 'object') {
        const candidate = Number(rawResults.totalMiningArea.m2);
        if (Number.isFinite(candidate) && candidate > 0) {
          return candidate;
        }
      }
      if (typeof summary.mining_area_m2 === 'number' && Number.isFinite(summary.mining_area_m2)) {
        return Number(summary.mining_area_m2);
      }
      if (mergedBlocks?.metadata?.total_area_m2 && Number.isFinite(mergedBlocks.metadata.total_area_m2)) {
        return Number(mergedBlocks.metadata.total_area_m2);
      }
      return 0;
    })();

    const statistics = (() => {
      const base = rawResults.statistics && typeof rawResults.statistics === 'object'
        ? { ...rawResults.statistics }
        : {};
      if (base.avgConfidence === undefined && typeof summary.confidence === 'number') {
        base.avgConfidence = summary.confidence * 100;
      }
      if (base.coveragePercentage === undefined && typeof summary.mining_percentage === 'number') {
        base.coveragePercentage = summary.mining_percentage;
      }
      return {
        avgConfidence: Number(base.avgConfidence) || 0,
        maxConfidence: Number(base.maxConfidence) || 0,
        minConfidence: Number(base.minConfidence) || 0,
        coveragePercentage: Number(base.coveragePercentage) || 0
      };
    })();

    console.log(`📊 Results summary:`, {
      status: rawResults.status ?? results?.status,
      totalTiles,
      tilesWithMining,
      detectionCount,
      mergedBlockFeatures: mergedBlocks?.features?.length,
      hasMergedBlocks: Boolean(mergedBlocks),
      hasResultsObject: !!results,
      resultKeys: results ? Object.keys(results).slice(0, 15) : []
    });

    if (!analysisId) {
      console.log('❌ Validation failed: Missing analysisId');
      return res.status(400).json({
        error: 'Invalid request',
        message: 'analysisId is required'
      });
    }

    // Check if analysis already exists
    const existing = await AnalysisHistory.findOne({ analysisId });
    if (existing) {
      console.log('⚠️  Analysis already exists in database');
      console.log(`📅 Originally saved: ${existing.createdAt}`);
      console.log(`👤 Owner: ${existing.userId}`);
      console.log(`📊 Existing data:`, {
        status: existing.status,
        totalTiles: existing.results?.totalTiles,
        detectionCount: existing.results?.detectionCount,
        totalMiningArea: existing.results?.totalMiningArea
      });
      
      // Check if this is the same user
      if (existing.userId.toString() === req.user._id.toString()) {
        // If force=true, allow overwriting
        if (force) {
          console.log('🔄 Force flag set - updating existing analysis');
        } else {
          console.log('✅ Same user - returning existing record (use force=true to update)');
          return res.status(409).json({
            error: 'Analysis already exists',
            message: 'This analysis has already been saved. Pass force=true to overwrite.',
            existingAnalysis: {
              analysisId: existing.analysisId,
              savedAt: existing.createdAt,
              status: existing.status,
              detectionCount: existing.results?.detectionCount || 0,
              totalTiles: existing.results?.totalTiles || 0,
              totalMiningArea: existing.results?.totalMiningArea || { m2: 0, hectares: 0, km2: 0 }
            }
          });
        }
      } else {
        console.log('❌ Different user attempting to save same analysis ID');
        return res.status(403).json({
          error: 'Forbidden',
          message: 'This analysis belongs to another user'
        });
      }
    }

    console.log('✅ Analysis ID is unique, proceeding with save...');

    // Calculate AOI area from geometry if provided
    let aoiArea = null;
    if (aoiGeometry && aoiGeometry.coordinates) {
      try {
        // Simple bounding box area calculation (for display purposes)
        const coords = aoiGeometry.coordinates[0];
        if (coords && coords.length > 0) {
          const lons = coords.map(c => c[0]);
          const lats = coords.map(c => c[1]);
          const width = Math.max(...lons) - Math.min(...lons);
          const height = Math.max(...lats) - Math.min(...lats);
          
          // Approximate area in km² (rough calculation)
          const approxKm2 = width * height * 111 * 111 * Math.cos((Math.max(...lats) + Math.min(...lats)) / 2 * Math.PI / 180);
          aoiArea = {
            km2: approxKm2,
            hectares: approxKm2 * 100
          };
          console.log(`📏 Calculated AOI area: ${aoiArea.hectares.toFixed(2)} ha (${aoiArea.km2.toFixed(4)} km²)`);
        }
      } catch (areaError) {
        console.warn('⚠️  Could not calculate AOI area:', areaError);
      }
    }

    const summaryCoveragePct = typeof summary.mining_percentage === 'number'
      ? summary.mining_percentage
      : (typeof statistics.coveragePercentage === 'number' ? statistics.coveragePercentage : null);

    const summaryConfidence = typeof summary.confidence === 'number'
      ? summary.confidence * 100
      : (typeof statistics.avgConfidence === 'number' ? statistics.avgConfidence : null);

    if (Object.keys(summary).length > 0) {
      console.log('📌 Summary snapshot:', {
        totalTiles,
        tilesWithDetections: tilesWithMining,
        mineBlockCount: detectionCount,
        miningAreaHa: totalMiningAreaM2 / 10000,
        miningCoveragePct: summaryCoveragePct,
        confidencePct: summaryConfidence
      });
    }

    // Process tiles to ensure mine_blocks have GeoJSON format
    const processedTiles = tiles.map(tile => {
      const rawTileId = tile.tile_id ?? tile.id ?? tile.index;
      const tileId = rawTileId !== undefined && rawTileId !== null ? String(rawTileId) : undefined;

      const tileIndex = typeof tile.index === 'number'
        ? tile.index
        : typeof tile.tile_index === 'number'
          ? tile.tile_index
          : (() => {
              if (rawTileId === undefined || rawTileId === null) return undefined;
              const numericCandidate = Number(rawTileId);
              return Number.isFinite(numericCandidate) ? numericCandidate : undefined;
            })();

      const tileLabel = tile.tile_label
        ?? (typeof rawTileId === 'string' ? rawTileId : undefined)
        ?? (tileIndex !== undefined ? `tile_${tileIndex}` : undefined);

      const mineBlocks = Array.isArray(tile.mine_blocks)
        ? tile.mine_blocks
        : Array.isArray(tile.mineBlocks)
          ? tile.mineBlocks
          : [];

      return {
        ...tile,
        tile_id: tileId,
        tile_index: tileIndex,
        tile_label: tileLabel,
        mine_blocks: mineBlocks,
        metadata: {
          ...(tile.metadata || {}),
          sourceTileIndex: tileIndex ?? tile.index,
          sourceBands: tile.bands_used || tile.bands,
          dimensions: tile.mask_shape || (tile.size ? tile.size.split('x').map(Number) : undefined),
          isMosaic: tile.status === 'mosaic'
        }
      };
    });

    console.log(`🗂️  Processing ${processedTiles.length} tiles`);
    const tilesWithBlocks = processedTiles.filter(t => t.mine_blocks && t.mine_blocks.length > 0);
    console.log(`   └─ ${tilesWithBlocks.length} tiles have mine blocks`);
    
    // Log mine block structure
    if (tilesWithBlocks.length > 0) {
      const firstTileWithBlocks = tilesWithBlocks[0];
      console.log(`   └─ First tile mine_blocks type: ${Array.isArray(firstTileWithBlocks.mine_blocks) ? 'Array' : typeof firstTileWithBlocks.mine_blocks}`);
      if (firstTileWithBlocks.mine_blocks.length > 0) {
        const firstBlock = firstTileWithBlocks.mine_blocks[0];
        console.log(`   └─ First block structure:`, {
          hasProperties: !!firstBlock.properties,
          hasGeometry: !!firstBlock.geometry,
          blockId: firstBlock.properties?.block_id,
          name: firstBlock.properties?.name,
          area_m2: firstBlock.properties?.area_m2
        });
      }
    }

    // Log merged blocks structure
    if (mergedBlocks) {
      console.log(`📦 Merged blocks:`, {
        type: mergedBlocks.type,
        featuresCount: mergedBlocks.features?.length || 0,
        metadata: mergedBlocks.metadata
      });
      
      if (mergedBlocks.features && mergedBlocks.features.length > 0) {
        const firstMerged = mergedBlocks.features[0];
        console.log(`   └─ First merged block:`, {
          blockId: firstMerged.properties?.block_id,
          name: firstMerged.properties?.name,
          area_m2: firstMerged.properties?.area_m2,
          is_merged: firstMerged.properties?.is_merged,
          hasGeometry: !!firstMerged.geometry,
          geometryType: firstMerged.geometry?.type,
          coordinatesLength: firstMerged.geometry?.coordinates?.length
        });
        
        // Log total area from metadata
        const metadataArea = mergedBlocks.metadata?.total_area_m2;
        if (metadataArea) {
          console.log(`   └─ Metadata total area: ${(metadataArea / 10000).toFixed(2)} ha (${metadataArea} m²)`);
        }
      }
    }

    const fallbackTrackedBlocks = processedTiles.flatMap(tile => {
      if (!Array.isArray(tile.mine_blocks) || tile.mine_blocks.length === 0) {
        return [];
      }

      return tile.mine_blocks.map(block => {
        const props = block?.properties || {};
        const persistentId = props.persistent_id || props.block_id || null;
        const boundsArray = Array.isArray(props.bbox) && props.bbox.length === 4 ? props.bbox : null;
        const centroidArray = Array.isArray(props.label_position) && props.label_position.length >= 2
          ? props.label_position
          : null;

        return {
          persistentId,
          blockId: props.block_id || null,
          sequence: typeof props.block_index === 'number' ? props.block_index : null,
          tileId: props.tile_id || tile.tile_id || tile.tile_label || null,
          name: props.name || null,
          areaM2: typeof props.area_m2 === 'number' ? props.area_m2 : null,
          areaHa: typeof props.area_m2 === 'number' ? props.area_m2 / 10000 : null,
          avgConfidence: typeof props.avg_confidence === 'number' ? props.avg_confidence : null,
          centroid: centroidArray,
          bounds: boundsArray,
          analysisId,
          updatedAt: new Date()
        };
      });
    });

    fallbackTrackedBlocks.sort((a, b) => {
      if (a.sequence !== null && b.sequence !== null) {
        return a.sequence - b.sequence;
      }
      if (a.areaHa !== null && b.areaHa !== null) {
        return b.areaHa - a.areaHa;
      }
      return 0;
    });

    const canonicalBlockTracking = rawResults.blockTracking && typeof rawResults.blockTracking === 'object'
      ? rawResults.blockTracking
      : null;

    const blockTrackingSummary = canonicalBlockTracking?.summary ?? {
      total: fallbackTrackedBlocks.length,
      withPersistentIds: fallbackTrackedBlocks.filter(block => !!block.persistentId).length
    };

    const blockTracking = canonicalBlockTracking?.blocks && Array.isArray(canonicalBlockTracking.blocks)
      ? canonicalBlockTracking.blocks
      : fallbackTrackedBlocks;

    // Calculate duration
    const { startTime, endTime, durationSeconds } = deriveTimingInfo(rawResults, metadata);

    // Create analysis record
    const analysisData = {
      analysisId,
      userId: req.user._id,
      aoiGeometry,
      aoiBounds,
      aoiArea,
      status: 'completed',
      startTime,
      endTime,
      duration: durationSeconds,
      currentStep: 'completed',
      progress: 100,
      logs: logs || [],
      results: {
        totalTiles: totalTiles || processedTiles.length || 0,
        tilesProcessed: totalTiles || processedTiles.length || 0,
        tilesWithMining: tilesWithMining || 0,
        detectionCount,
        totalMiningArea: (() => {
          let totalAreaM2 = totalMiningAreaM2;

          if ((!totalAreaM2 || totalAreaM2 <= 0) && Array.isArray(mergedBlocks?.features)) {
            totalAreaM2 = mergedBlocks.features.reduce((sum, feature) => (
              sum + (feature.properties?.area_m2 || 0)
            ), 0);
          }

          if ((!totalAreaM2 || totalAreaM2 <= 0) && processedTiles.length > 0) {
            totalAreaM2 = processedTiles.reduce((sum, tile) => {
              if (!Array.isArray(tile.mine_blocks)) {
                return sum;
              }
              return sum + tile.mine_blocks.reduce((innerSum, block) => (
                innerSum + (block.properties?.area_m2 || 0)
              ), 0);
            }, 0);
          }

          return {
            m2: totalAreaM2,
            hectares: totalAreaM2 / 10000,
            km2: totalAreaM2 / 1000000
          };
        })(),
        mergedBlocks,
        tiles: processedTiles,
        statistics,
        blockTracking: {
          summary: blockTrackingSummary,
          blocks: blockTracking
        },
        detections,
        summary: Object.keys(summary).length > 0 ? summary : undefined
      },
      metadata: metadata || {}
    };

    console.log('💾 Creating/updating database record...');
    console.log(`   └─ Total tiles: ${analysisData.results.totalTiles}`);
    console.log(`   └─ Tiles with mining: ${analysisData.results.tilesWithMining}`);
    console.log(`   └─ Detection count (mine blocks): ${analysisData.results.detectionCount}`);
    console.log(`   └─ Duration: ${analysisData.duration} seconds`);
    console.log(`   └─ Total mining area: ${analysisData.results.totalMiningArea.hectares.toFixed(2)} ha (${analysisData.results.totalMiningArea.m2.toFixed(0)} m²)`);
    if (analysisData.results.summary) {
      console.log('   └─ Summary payload stored');
    }

    let analysis;
    let isUpdate = false;

    // If force=true and record exists, update it
    if (force && existing) {
      console.log(`🔄 Force update: replacing existing analysis`);
      isUpdate = true;
      // Update all fields in existing record
      Object.assign(existing, analysisData);
      // Reset timestamps for updated record
      existing.endTime = new Date();
      await existing.save();
      analysis = existing;
    } else {
      // Create new record
      analysis = new AnalysisHistory(analysisData);
      await analysis.save();
    }

    console.log(`✅ Analysis ${isUpdate ? 'updated' : 'saved'} successfully to MongoDB!`);
    console.log(`   └─ Document ID: ${analysis._id}`);
    console.log(`   └─ Created/Updated at: ${isUpdate ? analysis.endTime : analysis.createdAt}`);
    console.log('================================================================\n');

    res.status(isUpdate ? 200 : 201).json({
      message: `Analysis ${isUpdate ? 'updated' : 'saved'} successfully`,
      analysisId: analysis.analysisId,
      analysis
    });
  } catch (error) {
    console.error('❌ ==================== SAVE ANALYSIS ERROR ====================');
    console.error('Error saving analysis:', error);
    console.error('Stack trace:', error.stack);
    console.error('================================================================\n');
    res.status(500).json({
      error: 'Failed to save analysis',
      message: error.message
    });
  }
});

/**
 * PUT /api/history/:analysisId
 * Update analysis metadata (notes, tags, etc.)
 */
router.put('/:analysisId', async (req, res) => {
  try {
    const { analysisId } = req.params;
    const { userNotes, tags, isArchived } = req.body;

    const analysis = await AnalysisHistory.findOne({
      analysisId,
      userId: req.user._id
    });

    if (!analysis) {
      return res.status(404).json({
        error: 'Analysis not found'
      });
    }

    // Update allowed fields
    if (userNotes !== undefined) analysis.userNotes = userNotes;
    if (tags !== undefined) analysis.tags = tags;
    if (isArchived !== undefined) analysis.isArchived = isArchived;

    await analysis.save();

    res.json({
      message: 'Analysis updated successfully',
      analysis
    });
  } catch (error) {
    console.error('❌ Error updating analysis:', error);
    res.status(500).json({
      error: 'Failed to update analysis',
      message: error.message
    });
  }
});

/**
 * PUT /api/history/:analysisId/quantitative
 * Persist quantitative volumetric analysis output, including logs and visualizations.
 */
router.put('/:analysisId/quantitative', async (req, res) => {
  try {
    const { analysisId } = req.params;
    const payload = req.body || {};

    console.log(`\n📦 Persisting quantitative analysis for ${analysisId}`);
    const bypassAuth = shouldBypassAuth();
    const persistedBy = req.user?._id ?? (bypassAuth ? 'dev-user' : null);

    if (!persistedBy) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const coerceNumber = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
    const clampArray = (arr, maxLength = 180) => Array.isArray(arr) ? arr.slice(0, maxLength).map((item) => coerceNumber(item) ?? item) : [];
    const clampMatrix = (matrix, maxRows = 160, maxCols = 160) => {
      if (!Array.isArray(matrix)) return [];
      return matrix.slice(0, maxRows).map((row) => {
        if (!Array.isArray(row)) return [];
        return row.slice(0, maxCols).map((value) => {
          if (value === null || value === undefined) return null;
          return coerceNumber(value);
        });
      });
    };

    const sanitizeBounds = (bounds) => {
      if (!Array.isArray(bounds)) {
        return undefined;
      }
      const flatBounds = bounds.length === 4 && bounds.every((value) => typeof value === 'number' || typeof value === 'string')
        ? bounds
        : null;
      if (!flatBounds) {
        return undefined;
      }
      const numeric = flatBounds.map((value) => {
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      });
      return numeric.every((value) => value !== null) ? numeric : undefined;
    };

    const sanitizeTransform = (transform) => {
      if (!Array.isArray(transform)) {
        return undefined;
      }
      const numeric = transform.slice(0, 6).map((value) => {
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      });
      return numeric.every((value) => value !== null) ? numeric : undefined;
    };

    const sanitizeImagery = (imagery = {}) => {
      if (!imagery || typeof imagery !== 'object') {
        return undefined;
      }

      const imageBase64 = typeof imagery.imageBase64 === 'string' && imagery.imageBase64.trim().length > 0
        ? imagery.imageBase64
        : undefined;
      const probabilityBase64 = typeof imagery.probabilityBase64 === 'string' && imagery.probabilityBase64.trim().length > 0
        ? imagery.probabilityBase64
        : undefined;
      const tileLabel = typeof imagery.tileLabel === 'string' ? imagery.tileLabel : undefined;
      const tileBounds = sanitizeBounds(imagery.tileBounds);
      const blockBounds = sanitizeBounds(imagery.blockBounds);
      const transform = sanitizeTransform(imagery.transform);
      const crs = typeof imagery.crs === 'string' && imagery.crs.trim().length > 0 ? imagery.crs : undefined;

      const cleaned = {};
      if (imageBase64) cleaned.imageBase64 = imageBase64;
      if (probabilityBase64) cleaned.probabilityBase64 = probabilityBase64;
      if (tileBounds) cleaned.tileBounds = tileBounds;
      if (blockBounds) cleaned.blockBounds = blockBounds;
      if (tileLabel) cleaned.tileLabel = tileLabel;
      if (transform) cleaned.transform = transform;
      if (crs) cleaned.crs = crs;

      return Object.keys(cleaned).length ? cleaned : undefined;
    };

    const sanitizeGrid = (grid = {}) => {
      if (!grid || typeof grid !== 'object') return undefined;
      return {
        x: clampArray(grid.x, 180),
        y: clampArray(grid.y, 180),
        elevation: clampMatrix(grid.elevation),
        depth: clampMatrix(grid.depth),
        rimElevation: coerceNumber(grid.rimElevation),
        resolutionX: coerceNumber(grid.resolutionX),
        resolutionY: coerceNumber(grid.resolutionY),
        unit: grid.unit || 'meters'
      };
    };

    const sanitizeBlock = (block = {}) => {
      if (!block || typeof block !== 'object') return null;

      const imagery = sanitizeImagery(block.visualization?.imagery || block.imagery);

      const base = {
        blockId: block.blockId,
        blockLabel: block.blockLabel,
        persistentId: block.persistentId,
        source: block.source,
        areaSquareMeters: coerceNumber(block.areaSquareMeters),
        areaHectares: coerceNumber(block.areaHectares),
        rimElevationMeters: coerceNumber(block.rimElevationMeters),
        maxDepthMeters: coerceNumber(block.maxDepthMeters),
        meanDepthMeters: coerceNumber(block.meanDepthMeters),
        medianDepthMeters: coerceNumber(block.medianDepthMeters),
        volumeCubicMeters: coerceNumber(block.volumeCubicMeters),
        volumeTrapezoidalCubicMeters: coerceNumber(block.volumeTrapezoidalCubicMeters),
        pixelCount: typeof block.pixelCount === 'number' ? Math.round(block.pixelCount) : undefined,
        centroid: block.centroid,
        computedAt: block.computedAt ? new Date(block.computedAt) : undefined,
        notes: block.notes
      };

      let visualizationPayload = block.visualization ? {
        grid: sanitizeGrid(block.visualization.grid),
        stats: block.visualization.stats,
        extentUTM: block.visualization.extentUTM,
        metadata: block.visualization.metadata
      } : undefined;

      if (imagery) {
        if (visualizationPayload) {
          visualizationPayload.imagery = imagery;
        } else {
          visualizationPayload = { imagery };
        }
      }

      return {
        ...base,
        visualization: visualizationPayload
      };
    };

    const sanitizedBlocks = Array.isArray(payload.blocks)
      ? payload.blocks.map(sanitizeBlock).filter(Boolean)
      : [];

    const sanitizedSteps = Array.isArray(payload.steps)
      ? payload.steps.map((step) => ({
          name: step.name,
          status: step.status,
          durationMs: coerceNumber(step.durationMs),
          details: Array.isArray(step.details) ? step.details.slice(0, 25) : []
        }))
      : [];

    const buildQuantitativeRecord = (persistedById) => ({
      status: payload.status || 'completed',
      executedAt: payload.executedAt ? new Date(payload.executedAt) : new Date(),
      steps: sanitizedSteps,
      summary: payload.summary || {},
      executiveSummary: payload.executiveSummary || {},
      blocks: sanitizedBlocks,
      dem: payload.dem,
      source: payload.source,
      metadata: {
        ...(payload.metadata || {}),
        persistedAt: new Date().toISOString(),
        persistedBy: persistedById,
      }
    });

    const quantitativeRecord = buildQuantitativeRecord(persistedBy);

    if (!req.user && bypassAuth) {
      console.warn('⚠️  DEV_ALLOW_OPEN_PYTHON_PROXY enabled - storing quantitative snapshot in dev memory store');
      const existing = devHistoryStore.get(analysisId);

      if (!existing) {
        console.log('❌ Analysis not found in dev store');
        return res.status(404).json({ error: 'Analysis not found' });
      }

      const updatedRecord = {
        ...existing,
        quantitativeAnalysis: quantitativeRecord,
        updatedAt: new Date().toISOString()
      };

      devHistoryStore.set(analysisId, updatedRecord);

      console.log(`✅ Quantitative analysis stored for ${analysisId} (dev memory store)`);

      return res.json({
        message: 'Quantitative analysis saved',
        analysisId,
        quantitativeAnalysis: quantitativeRecord
      });
    }

    const analysis = await AnalysisHistory.findOne({
      analysisId,
      userId: req.user._id
    });

    if (!analysis) {
      console.log('❌ Analysis not found or not owned by user');
      return res.status(404).json({
        error: 'Analysis not found'
      });
    }

    analysis.quantitativeAnalysis = quantitativeRecord;
    analysis.markModified('quantitativeAnalysis');
    await analysis.save();

    console.log(`✅ Quantitative analysis stored for ${analysisId}`);

    res.json({
      message: 'Quantitative analysis saved',
      analysisId,
      quantitativeAnalysis: analysis.quantitativeAnalysis
    });
  } catch (error) {
    console.error('❌ Error saving quantitative analysis:', error);
    res.status(500).json({
      error: 'Failed to save quantitative analysis',
      message: error.message
    });
  }
});

/**
 * DELETE /api/history/:analysisId
 * Delete analysis record
 */
router.delete('/:analysisId', async (req, res) => {
  try {
    const { analysisId } = req.params;

    const analysis = await AnalysisHistory.findOneAndDelete({
      analysisId,
      userId: req.user._id
    });

    if (!analysis) {
      return res.status(404).json({
        error: 'Analysis not found'
      });
    }

    res.json({
      message: 'Analysis deleted successfully',
      analysisId
    });
  } catch (error) {
    console.error('❌ Error deleting analysis:', error);
    res.status(500).json({
      error: 'Failed to delete analysis',
      message: error.message
    });
  }
});

/**
 * DELETE /api/history/:analysisId
 * Delete a single analysis record by ID (allows re-analysis)
 */
router.delete('/:analysisId', async (req, res) => {
  try {
    const { analysisId } = req.params;

    console.log(`\n🗑️  ==================== DELETE ANALYSIS REQUEST ====================`);
    console.log(`📋 Analysis ID: ${analysisId}`);
    console.log(`👤 User ID: ${req.user._id}`);

    const deleted = await AnalysisHistory.findOneAndDelete({
      analysisId,
      userId: req.user._id
    });

    if (!deleted) {
      console.log('❌ Analysis not found or user not authorized');
      return res.status(404).json({
        error: 'Not found',
        message: 'Analysis not found or you do not have permission to delete it'
      });
    }

    console.log('✅ Analysis deleted successfully');
    console.log(`   └─ Deleted: ${deleted.analysisId}`);
    console.log(`   └─ Status was: ${deleted.status}`);
    console.log('================================================================\n');

    res.json({
      message: 'Analysis deleted successfully',
      analysisId: deleted.analysisId,
      deletedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Error deleting analysis:', error);
    console.log('================================================================\n');
    res.status(500).json({
      error: 'Failed to delete analysis',
      message: error.message
    });
  }
});

/**
 * POST /api/history/bulk-delete
 * Delete multiple analysis records
 */
router.post('/bulk-delete', async (req, res) => {
  try {
    const { analysisIds } = req.body;

    if (!Array.isArray(analysisIds) || analysisIds.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'analysisIds must be a non-empty array'
      });
    }

    const result = await AnalysisHistory.deleteMany({
      analysisId: { $in: analysisIds },
      userId: req.user._id
    });

    res.json({
      message: 'Analyses deleted successfully',
      deletedCount: result.deletedCount
    });
  } catch (error) {
    console.error('❌ Error bulk deleting analyses:', error);
    res.status(500).json({
      error: 'Failed to delete analyses',
      message: error.message
    });
  }
});

export default router;
