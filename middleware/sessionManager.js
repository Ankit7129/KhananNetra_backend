import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import User from '../models/User.js';
import BlockedIP from '../models/BlockedIP.js';
import FailedAttempt from '../models/FailedAttempt.js';
import VerifierRegistry from '../models/VerifierRegistry.js';

// Security Configuration
const SECURITY_CONFIG = {
  login_failed: {
    maxAttempts: 5,
    windowMinutes: 15,
    blockDurationMinutes: 1440
  },
  invalid_session: {
    maxAttempts: 10,
    windowMinutes: 60,
    blockDurationMinutes: 1440
  }
};

const RISK_THRESHOLDS = {
  LOW: 30,
  MEDIUM: 50,
  HIGH: 70,
  CRITICAL: 90
};

class SessionManager {
  constructor() {
    this.sessionTimeout = 30 * 24 * 60 * 60 * 1000; // 30 days
    this.accessTokenExpiry = 60 * 60 * 1000; // 1 hour
    this.refreshTokenExpiry = 90 * 24 * 60 * 60 * 1000; // 90 days
  }

  // Generate device fingerprint for web
  generateDeviceFingerprint(deviceInfo) {
    const components = [
      deviceInfo.userAgent,
      deviceInfo.deviceType,
      deviceInfo.os,
      deviceInfo.browser
    ].filter(Boolean).join('|');

    return crypto.createHash('sha256').update(components).digest('hex');
  }

  // Check if IP is blocked
  async isIPBlocked(ip) {
    return await BlockedIP.isIPBlocked(ip);
  }

  // Record security event
  async recordSecurityEvent(ip, email, eventType, userAgent = null, details = {}) {
    try {
      await FailedAttempt.recordSecurityEvent(ip, email, eventType, userAgent, details);

      // Check if we should block IP
      const eventConfig = SECURITY_CONFIG[eventType];
      if (eventConfig) {
        const recentAttempts = await FailedAttempt.getRecentAttemptsCount(
          ip, 
          eventConfig.windowMinutes, 
          eventType
        );
        
        if (recentAttempts >= eventConfig.maxAttempts) {
          await BlockedIP.blockIP(ip, `Multiple ${eventType} attempts`, email, eventConfig.blockDurationMinutes);
          return true;
        }
      }

      return false;
    } catch (error) {
      console.error('Error recording security event:', error);
      return false;
    }
  }

  // Clear failed attempts for successful login
  async clearFailedAttempts(ip, email) {
    try {
      await FailedAttempt.deleteMany({ 
        $or: [
          { ip, eventType: 'login_failed' },
          { email, eventType: 'login_failed' }
        ]
      });
    } catch (error) {
      console.error('Error clearing failed attempts:', error);
    }
  }

  // Calculate risk score for web sessions
  async calculateRiskScore(user, ip, deviceInfo) {
    let riskScore = 0;

    // Check if IP is blocked
    if (await this.isIPBlocked(ip)) {
      return 100;
    }

    // Check device trust
    const isTrustedDevice = user.securityPatterns?.knownDevices?.some(
      device => device.deviceFingerprint === deviceInfo.deviceFingerprint && device.isTrusted
    );

    if (!isTrustedDevice) {
      riskScore += 25;
    }

    // Check location patterns
    const commonCountries = user.securityPatterns?.commonCountries || [];
    const countryMatch = commonCountries.find(c => c.country === deviceInfo.country);
    if (!countryMatch) {
      riskScore += 30;
    }

    // Check recent security events
    const recentEvents = await FailedAttempt.getRecentAttemptsCount(ip, 60);
    riskScore += Math.min(recentEvents * 5, 20);

    return Math.min(100, riskScore);
  }

  // Create web session and set cookies
  async createWebSession(res, user, deviceInfo) {
    try {
      // Check IP blocking
      if (await this.isIPBlocked(deviceInfo.ip)) {
        throw new Error('IP address is temporarily blocked');
      }

      const sessionId = crypto.randomBytes(32).toString('hex');
      
      // Generate refresh token
      const refreshToken = jwt.sign(
        { 
          userId: user._id, 
          sessionId,
          type: 'refresh'
        }, 
        process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET + '_refresh', 
        { expiresIn: '90d' }
      );

      // Generate device fingerprint
      const deviceFingerprint = this.generateDeviceFingerprint(deviceInfo);

      // Calculate risk score
      const riskScore = await this.calculateRiskScore(user, deviceInfo.ip, {
        ...deviceInfo,
        deviceFingerprint
      });

      // Create session object
      const newSession = {
        sessionId,
        refreshToken,
        deviceFingerprint,
        deviceId: deviceInfo.deviceId || `web_${crypto.randomBytes(16).toString('hex')}`,
        deviceType: 'web',
        deviceName: deviceInfo.deviceName || 'Web Browser',
        os: deviceInfo.os || 'Unknown',
        browser: deviceInfo.browser || 'Unknown',
        userAgent: deviceInfo.userAgent,
        ip: deviceInfo.ip,
        country: deviceInfo.country || 'Unknown',
        city: deviceInfo.city || 'Unknown',
        authType: 'web',
        isActive: true,
        isTrusted: riskScore < RISK_THRESHOLDS.MEDIUM,
        mfaVerified: false,
        riskScore,
        createdAt: new Date(),
        lastUsedAt: new Date(),
        expiresAt: new Date(Date.now() + this.sessionTimeout),
        flags: {
          requiresReauth: riskScore > RISK_THRESHOLDS.HIGH,
          suspiciousActivity: riskScore > RISK_THRESHOLDS.MEDIUM,
          unusualLocation: riskScore > RISK_THRESHOLDS.LOW
        }
      };

      // Add session to user and cleanup old sessions
      user.loginSessions.unshift(newSession);
      user.cleanupOldSessions();
      user.lastLoginAt = new Date();
      user.lastLoginIP = deviceInfo.ip;

      await user.save();

      // Generate access token
      const accessToken = jwt.sign(
        { 
          userId: user._id, 
          sessionId,
          type: 'access'
        }, 
        process.env.JWT_SECRET, 
        { expiresIn: '1h' }
      );

      // Set HTTP-only cookies
      res.cookie('accessToken', accessToken, {
        httpOnly: true,
        secure: true,
        sameSite: 'none',
        maxAge: this.accessTokenExpiry,
      });

      res.cookie('refreshToken', refreshToken, {
        httpOnly: true,
        secure: true,
        sameSite: 'none',
        maxAge: this.refreshTokenExpiry
      });

      // Clear failed attempts for this IP/email
      await this.clearFailedAttempts(deviceInfo.ip, user.email);

      console.log('Session created successfully for user:', user.email);

      return {
        sessionId,
        riskScore,
        requiresReauth: riskScore > RISK_THRESHOLDS.HIGH
      };

    } catch (error) {
      console.error('Session creation error:', error);
      throw error;
    }
  }

  // ✅ FIXED: Validate session from cookies (used by protect middleware)
  async validateWebSession(accessToken, ip, userAgent) {
    try {
      if (await this.isIPBlocked(ip)) {
        throw new Error('IP address is blocked');
      }

      // Verify access token
      const decoded = jwt.verify(accessToken, process.env.JWT_SECRET);
      
      if (decoded.type !== 'access') {
        throw new Error('Invalid token type');
      }

      const user = await User.findOne({
        'loginSessions.sessionId': decoded.sessionId,
        'loginSessions.isActive': true,
        'loginSessions.expiresAt': { $gt: new Date() }
      });

      if (!user) {
        await this.recordSecurityEvent(ip, null, 'invalid_session', userAgent);
        throw new Error('Invalid or expired session');
      }

      const session = user.loginSessions.find(s => s.sessionId === decoded.sessionId);
      
      if (!session) {
        await this.recordSecurityEvent(ip, null, 'invalid_session', userAgent);
        throw new Error('Session not found');
      }

      // Check if session requires reauthentication
      if (session.flags.requiresReauth) {
        throw new Error('Reauthentication required');
      }

      // Update last used timestamp
      session.lastUsedAt = new Date();
      await user.save();

      return {
        userId: user._id,
        session,
        user
      };

    } catch (error) {
      console.error('Session validation error:', error);
      
      // Record security event for invalid sessions
      if (error.message.includes('Invalid') || error.message.includes('expired')) {
        await this.recordSecurityEvent(ip, null, 'invalid_session', userAgent);
      }
      
      throw error;
    }
  }

  // Refresh access token using refresh token
  async refreshAccessToken(refreshToken, res) {
    try {
      const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET + '_refresh');
      
      if (decoded.type !== 'refresh') {
        throw new Error('Invalid token type');
      }

      const user = await User.findOne({
        'loginSessions.sessionId': decoded.sessionId,
        'loginSessions.isActive': true,
        'loginSessions.expiresAt': { $gt: new Date() }
      });

      if (!user) {
        throw new Error('Invalid refresh token');
      }

      const session = user.loginSessions.find(s => s.sessionId === decoded.sessionId);
      
      if (!session) {
        throw new Error('Session not found');
      }

      // Generate new access token
      const newAccessToken = jwt.sign(
        { 
          userId: user._id, 
          sessionId: decoded.sessionId,
          type: 'access'
        }, 
        process.env.JWT_SECRET, 
        { expiresIn: '1h' }
      );

      // Set new access token cookie
      res.cookie('accessToken', newAccessToken, {
        httpOnly: true,
        secure: true,
        sameSite: 'none',
        maxAge: this.accessTokenExpiry,
      });

      // Update session last used
      session.lastUsedAt = new Date();
      await user.save();

      return {
        accessToken: newAccessToken,
        sessionId: decoded.sessionId
      };

    } catch (error) {
      console.error('Token refresh error:', error);
      throw error;
    }
  }

  // Revoke session (logout)
  async revokeSession(userId, sessionId, res) {
    try {
      const user = await User.findById(userId);
      if (!user) throw new Error("User not found");
      
      const sessionIndex = user.loginSessions.findIndex(s => s.sessionId === sessionId);
      if (sessionIndex === -1) throw new Error("Session not found");
      
      // Remove session
      user.loginSessions.splice(sessionIndex, 1);
      await user.save();
      
      // Clear cookies
      res.clearCookie('accessToken');
      res.clearCookie('refreshToken');
      
      return true;
    } catch (error) {
      console.error('Error revoking session:', error);
      throw error;
    }
  }

  // Get active web sessions for user
  async getActiveSessions(userId) {
    try {
      const user = await User.findById(userId);
      if (!user) throw new Error('User not found');

      return user.loginSessions
        .filter(session => session.isActive && session.authType === 'web')
        .map(session => ({
          sessionId: session.sessionId,
          deviceType: session.deviceType,
          deviceName: session.deviceName,
          os: session.os,
          browser: session.browser,
          ip: session.ip,
          country: session.country,
          city: session.city,
          createdAt: session.createdAt,
          lastUsedAt: session.lastUsedAt,
          isTrusted: session.isTrusted,
          riskScore: session.riskScore
        }));
    } catch (error) {
      console.error('Get active sessions error:', error);
      throw error;
    }
  }

  // Cleanup expired sessions (can be called periodically)
  async cleanupExpiredSessions() {
    try {
      const result = await User.updateMany(
        {
          'loginSessions.expiresAt': { $lt: new Date() }
        },
        {
          $pull: {
            loginSessions: {
              expiresAt: { $lt: new Date() }
            }
          }
        }
      );
      
      console.log(`Cleaned up expired sessions for ${result.modifiedCount} users`);
      return result.modifiedCount;
    } catch (error) {
      console.error('Error cleaning up expired sessions:', error);
      throw error;
    }
  }
}

// Create singleton instance
const sessionManager = new SessionManager();

export default sessionManager;