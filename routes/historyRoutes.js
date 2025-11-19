/**
 * Analysis History Routes
 * Manage analysis records - create, read, update, delete
 */

import express from 'express';
import AnalysisHistory from '../models/AnalysisHistory.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

// All routes require authentication
router.use(protect);

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

    const skip = (parseInt(page) - 1) * parseInt(limit);
    const query = { userId: req.user._id };

    // Apply filters
    if (status) query.status = status;
    if (search) {
      query.$or = [
        { analysisId: { $regex: search, $options: 'i' } },
        { aoiId: { $regex: search, $options: 'i' } },
        { tags: { $in: [new RegExp(search, 'i')] } }
      ];
    }

    // Build sort object based on sortBy field
    const sortMap = {
      'startTime': 'startTime',
      'duration': 'duration',
      'detectionCount': 'results.detectionCount'
    };
    
    const sortField = sortMap[sortBy] || 'startTime';
    const sortDirection = sortOrder === 'asc' ? 1 : -1;
    const sortObj = { [sortField]: sortDirection };

    // Get total count for pagination
    const total = await AnalysisHistory.countDocuments(query);

    // Get paginated results
    const analyses = await AnalysisHistory.find(query)
      .sort(sortObj)
      .limit(parseInt(limit))
      .skip(skip)
      .select('-tiles.image_base64 -tiles.probability_map_base64 -logs')
      .lean();

    res.json({
      analyses,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / parseInt(limit))
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

    console.log('\n📝 ==================== SAVE ANALYSIS REQUEST ====================');
    console.log(`📋 Analysis ID: ${analysisId}`);
    console.log(`👤 User ID: ${req.user._id}`);
    
    // More robust field extraction
    const tiles = results?.tiles || [];
    const tilesWithMining = tiles.filter(t => t.mining_detected || t.miningDetected).length;
    
    // Calculate total mine blocks from multiple sources
    let totalMineBlocks = 0;
    let blockCountSource = 'unknown';
    
    // Try 1: Check merged_block_count (from Python merge_polygons)
    if (results?.merged_block_count && results.merged_block_count > 0) {
      totalMineBlocks = results.merged_block_count;
      blockCountSource = 'merged_block_count';
    }
    // Try 2: Check merged_blocks.metadata.merged_block_count
    else if (results?.merged_blocks?.metadata?.merged_block_count && results.merged_blocks.metadata.merged_block_count > 0) {
      totalMineBlocks = results.merged_blocks.metadata.merged_block_count;
      blockCountSource = 'metadata.merged_block_count';
    }
    // Try 3: Check merged_blocks.features (GeoJSON format)
    else if (results?.merged_blocks?.features && Array.isArray(results.merged_blocks.features) && results.merged_blocks.features.length > 0) {
      totalMineBlocks = results.merged_blocks.features.length;
      blockCountSource = 'merged_blocks.features.length';
    }
    // Try 4: Check total_mine_blocks (old field name)
    else if (results?.total_mine_blocks && results.total_mine_blocks > 0) {
      totalMineBlocks = results.total_mine_blocks;
      blockCountSource = 'total_mine_blocks';
    }
    
    // Try 5: If merged blocks have 0 count, fall back to counting individual tile blocks
    // This handles the case where some blocks were skipped due to low confidence
    if (totalMineBlocks === 0 && tiles.length > 0) {
      tiles.forEach(tile => {
        if (tile.mine_blocks && Array.isArray(tile.mine_blocks)) {
          totalMineBlocks += tile.mine_blocks.length;
        }
      });
      if (totalMineBlocks > 0) {
        blockCountSource = 'individual_tile_blocks (merged=0)';
      }
    }
    
    console.log(`📊 Results summary:`, {
      status: results?.status,
      totalTiles: results?.total_tiles || tiles.length,
      tilesWithMining,
      totalMineBlocks: totalMineBlocks,
      blockCountSource: blockCountSource,
      mergedBlockCount: results?.merged_block_count,
      mergedBlocksFeatures: results?.merged_blocks?.features?.length,
      hasMergedBlocks: !!results?.merged_blocks,
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

    // Process tiles to ensure mine_blocks have GeoJSON format
    const processedTiles = (results?.tiles || []).map(tile => ({
      ...tile,
      mine_blocks: tile.mine_blocks || [],
      tile_id: tile.tile_id || tile.id || tile.index
    }));

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
    if (results?.merged_blocks) {
      console.log(`📦 Merged blocks:`, {
        type: results.merged_blocks.type,
        featuresCount: results.merged_blocks.features?.length || 0,
        metadata: results.merged_blocks.metadata
      });
      
      if (results.merged_blocks.features && results.merged_blocks.features.length > 0) {
        const firstMerged = results.merged_blocks.features[0];
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
        const metadataArea = results.merged_blocks.metadata?.total_area_m2;
        if (metadataArea) {
          console.log(`   └─ Metadata total area: ${(metadataArea / 10000).toFixed(2)} ha (${metadataArea} m²)`);
        }
      }
    }

    // Calculate duration
    const startTime = new Date(results?.start_time || results?.created_at || Date.now());
    const endTime = new Date();
    const duration = Math.round((endTime.getTime() - startTime.getTime()) / 1000); // Duration in seconds

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
      duration,
      currentStep: 'completed',
      progress: 100,
      logs: logs || [],
      results: {
        totalTiles: results?.total_tiles || results?.tiles?.length || processedTiles.length || 0,
        tilesProcessed: results?.tiles_processed || processedTiles.length || 0,
        tilesWithMining: tilesWithMining || 0,
        detectionCount: totalMineBlocks || 0,  // Use the robust totalMineBlocks calculation above
        totalMiningArea: (() => {
          // Calculate total mining area from multiple sources
          let totalAreaM2 = 0;
          
          // Try 1: Use total_mining_area_ha from Python (converted)
          if (results?.total_mining_area_ha && results.total_mining_area_ha > 0) {
            totalAreaM2 = results.total_mining_area_ha * 10000; // ha to m²
          }
          // Try 2: Use total_mining_area_m2 from Python
          else if (results?.total_mining_area_m2 && results.total_mining_area_m2 > 0) {
            totalAreaM2 = results.total_mining_area_m2;
          }
          // Try 3: Calculate from merged_blocks metadata
          else if (results?.merged_blocks?.metadata?.total_area_m2 && results.merged_blocks.metadata.total_area_m2 > 0) {
            totalAreaM2 = results.merged_blocks.metadata.total_area_m2;
          }
          // Try 4: Sum individual mine blocks from merged_blocks.features (if any)
          else if (results?.merged_blocks?.features && Array.isArray(results.merged_blocks.features) && results.merged_blocks.features.length > 0) {
            totalAreaM2 = results.merged_blocks.features.reduce((sum, feature) => {
              return sum + (feature.properties?.area_m2 || 0);
            }, 0);
          }
          // Try 5: If merged blocks are empty, sum from individual tile blocks
          // This handles case where blocks were skipped due to low confidence
          else if (tiles.length > 0) {
            tiles.forEach(tile => {
              if (tile.mine_blocks && Array.isArray(tile.mine_blocks)) {
                tile.mine_blocks.forEach(block => {
                  totalAreaM2 += block.properties?.area_m2 || 0;
                });
              }
            });
          }
          
          return {
            m2: totalAreaM2,
            hectares: totalAreaM2 / 10000,
            km2: totalAreaM2 / 1000000
          };
        })(),
        mergedBlocks: results?.merged_blocks || null,
        tiles: processedTiles,
        statistics: {
          avgConfidence: results?.avg_confidence || 0,
          maxConfidence: results?.max_confidence || 0,
          minConfidence: results?.min_confidence || 0,
          coveragePercentage: results?.mining_coverage_percentage || 0
        }
      },
      metadata: metadata || {}
    };

    console.log('💾 Creating/updating database record...');
    console.log(`   └─ Total tiles: ${analysisData.results.totalTiles}`);
    console.log(`   └─ Tiles with mining: ${analysisData.results.tilesWithMining}`);
    console.log(`   └─ Detection count (mine blocks): ${analysisData.results.detectionCount}`);
    console.log(`   └─ Duration: ${analysisData.duration} seconds`);
    console.log(`   └─ Total mining area: ${analysisData.results.totalMiningArea.hectares.toFixed(2)} ha (${analysisData.results.totalMiningArea.m2.toFixed(0)} m²)`);

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
