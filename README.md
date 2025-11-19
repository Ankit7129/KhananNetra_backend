# KhananNetra Backend

Government platform for mining activity monitoring and compliance using satellite imagery analysis and geospatial intelligence.

## 📋 Overview

KhananNetra Backend provides:
- **RESTful API** (Node.js/Express) for authentication, user management, and analysis operations
- **Python FastAPI Backend** for geospatial analysis and ML inference
- **MongoDB** for data persistence
- **Docker** for containerized deployment
- **GCP Cloud Run** ready for production deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│      Production Container (GCP)         │
│                                          │
│  ┌────────────────┐  ┌───────────────┐ │
│  │   Node.js      │  │   Python      │ │
│  │   Express      │──│   FastAPI     │ │
│  │   (Port 8080)  │  │   (Port 8001) │ │
│  └────────────────┘  └───────────────┘ │
└─────────────────────────────────────────┘
```

## 🚀 Deployment

### Production (GCP Cloud Run)

See **[GCP_DEPLOYMENT.md](./GCP_DEPLOYMENT.md)** for complete deployment instructions.

Quick deploy:
```bash
# Build and deploy to GCP
docker build -f Dockerfile.production -t gcr.io/$PROJECT_ID/khanannetra-backend:latest .
docker push gcr.io/$PROJECT_ID/khanannetra-backend:latest
gcloud run deploy khanannetra-backend --image gcr.io/$PROJECT_ID/khanannetra-backend:latest
```

### Local Development (Docker Compose)

```bash
# Clone repository
git clone <repo-url>
cd KhananNetra_backend

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Check health
curl http://localhost:5000/api/health
```

Services available at:
- **Node.js API**: http://localhost:5000/api
- **Python Backend**: http://localhost:8001
- **MongoDB**: localhost:27017

### Local Development (Without Docker)

**Terminal 1: Node.js Backend**
```bash
npm install
npm start
# Running on http://localhost:5000/api
```

**Terminal 2: Python Backend**
```bash
cd python-backend
pip install -r requirements.txt
python main.py
# Running on http://localhost:8001
```

## 📚 API Documentation

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Authentication
```bash
# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# Refresh Token
curl -X POST http://localhost:5000/api/auth/refresh-token \
  -H "Cookie: refreshToken=<token>"

# Logout
curl -X POST http://localhost:5000/api/auth/logout
```

### Analysis History
```bash
# Get analysis history
curl http://localhost:5000/api/history?page=1&limit=10 \
  -H "Authorization: Bearer <token>"

# Get statistics
curl http://localhost:5000/api/history/stats \
  -H "Authorization: Bearer <token>"

# Get single analysis
curl http://localhost:5000/api/history/<analysisId> \
  -H "Authorization: Bearer <token>"
```

### Python Backend Analysis
```bash
# Create AOI
curl -X POST http://localhost:5000/api/python/aoi/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"geometry":{"type":"Polygon","coordinates":[...]}}'

# Start analysis
curl -X POST http://localhost:5000/api/python/analysis/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"aoiId":"<aoiId>","dates":{"start":"2024-01-01","end":"2024-12-31"}}'

# Get analysis status
curl http://localhost:5000/api/python/analysis/<analysisId>/status \
  -H "Authorization: Bearer <token>"
```

## 🏗️ Project Structure

```
KhananNetra_backend/
├── Dockerfile                    # Node.js backend container
├── docker-compose.yml           # Multi-service orchestration
├── start.sh / start.bat         # Quick start scripts
├── DOCKER_DEPLOYMENT_GUIDE.md   # Deployment documentation
├── .env.example                 # Environment variables template
├── package.json                 # Node.js dependencies
├── server.js                    # Express server entry point
│
├── config/
│   ├── database.js              # MongoDB connection
│   └── ...
│
├── models/
│   ├── User.js                  # User model
│   ├── AnalysisHistory.js       # Analysis records
│   ├── VerifierRegistry.js      # Verifier management
│   └── ...
│
├── routes/
│   ├── auth.js                  # Authentication routes
│   ├── users.js                 # User management
│   ├── historyRoutes.js         # Analysis history
│   ├── pythonProxy.js           # Python backend proxy
│   ├── adminRoutes.js           # Admin operations
│   └── ...
│
├── middleware/
│   ├── auth.js                  # Authentication middleware
│   └── sessionManager.js        # Session management
│
├── python-backend/              # Python FastAPI service
│   ├── Dockerfile               # Python container
│   ├── requirements.txt         # Python dependencies
│   ├── main.py                  # FastAPI entry point
│   ├── app/
│   │   ├── models/              # Data models
│   │   ├── routers/             # API routes
│   │   ├── services/            # Business logic
│   │   └── utils/               # Utilities
│   └── ...
│
└── .github/
    └── workflows/
        └── deploy-gcp.yml       # GitHub Actions for GCP
```

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

```env
# Server
NODE_ENV=production
PORT=8000

# Database
MONGODB_URI=mongodb://admin:password@mongodb:27017/khanannetra?authSource=admin

# Python Backend
PYTHON_BACKEND_URL=http://python-backend:8001

# JWT
JWT_SECRET=your-secret-key-min-32-chars
JWT_EXPIRE=7d

# Client
CLIENT_URL=http://localhost:3000
```

See `.env.example` for all available options.

## 🐳 Docker Commands

Using the startup scripts:

```bash
# Start services
./start.sh up -d

# Stop services
./start.sh down

# View logs
./start.sh logs [service]

# Check health
./start.sh health

# Restart services
./start.sh restart

# Clean up
./start.sh clean

# Open shell
./start.sh shell [service]

# Build images
./start.sh build
```

Or use docker-compose directly:

```bash
# Build
docker-compose build

# Start detached
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## 📊 Monitoring & Debugging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f nodejs-backend
docker-compose logs -f python-backend

# With timestamps
docker-compose logs -f --timestamps
```

### Access Service Shell
```bash
# Node.js
docker-compose exec nodejs-backend sh

# Python
docker-compose exec python-backend bash

# MongoDB
docker-compose exec mongodb mongosh
```

### Network Testing
```bash
# Test service communication
docker-compose exec nodejs-backend curl http://python-backend:8001/health

# Test from Python
docker-compose exec python-backend curl http://nodejs-backend:8000/api/health
```

## 🚢 Deployment

### Local to GCP Cloud Run

See [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md) for detailed instructions.

Quick deploy:
```bash
# Set up GCP
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Build and push images
docker-compose build
docker tag khanannetra-nodejs-backend gcr.io/$PROJECT_ID/khanannetra-nodejs-backend
docker push gcr.io/$PROJECT_ID/khanannetra-nodejs-backend

# Deploy
gcloud run deploy khanannetra-nodejs-backend \
  --image gcr.io/$PROJECT_ID/khanannetra-nodejs-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Automated Deployment via GitHub Actions

Push to `main` branch to trigger automatic deployment to GCP:

```bash
git add .
git commit -m "Deploy to GCP"
git push origin main
```

GitHub Actions will:
1. Build Docker images
2. Push to Container Registry
3. Deploy to Cloud Run

See `.github/workflows/deploy-gcp.yml` for configuration.

## 🔒 Security

- **JWT Authentication** for API routes
- **CORS** configured for frontend domain
- **Rate Limiting** on API endpoints
- **Environment Variables** for sensitive data
- **HTTPS** enforced in production
- **CSRF Protection** available
- **Input Validation** on all endpoints

## 📈 Performance

- **MongoDB Indexes** for fast queries
- **Redis Caching** for session and data
- **Multi-worker Python Backend** with Uvicorn
- **Horizontal Scaling** with Cloud Run

## 🆘 Troubleshooting

### Services not starting
```bash
# Check Docker daemon
docker ps

# View startup logs
docker-compose logs

# Rebuild images
docker-compose build --no-cache
docker-compose up
```

### Database connection error
```bash
# Check MongoDB
docker-compose exec mongodb mongosh

# Verify connection string
docker-compose logs mongodb
```

### Python backend not reachable
```bash
# Check service
docker-compose exec nodejs-backend \
  curl http://python-backend:8001/health

# View Python logs
docker-compose logs python-backend
```

### Port conflicts
```bash
# Find process on port
lsof -i :5000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

## 📞 Support

- Check logs: `docker-compose logs -f`
- Review [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md)
- Check GitHub Issues
- Contact development team

## 📄 License

Government of India - Ministry of Mines

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/feature-name`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/feature-name`
4. Submit Pull Request

## 📝 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for release history.

---

**Last Updated**: November 18, 2025

For deployment to production, see [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md)
