# 🚀 Workflow Generator - Docker Setup

## Overview
This is a containerized workflow generation application that combines:
- Flask REST API server for frontend communication
- AI-powered workflow engine using Google Gemini
- Interactive web interface for workflow design

## 📋 Prerequisites

- Docker installed on your system
  - Modern Docker: includes `docker compose` command
  - Legacy Docker: requires separate `docker-compose` installation
  - Cloud environments: may only have `docker` without compose
- Google Gemini API key

## 🔧 Setup

### 1. Create Environment File

Create a `.env` file in the root directory with your API credentials:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

**Important:** Never commit your `.env` file to version control!

### 2. Build the Docker Image

```bash
docker build -t workflow-generator .
```

This will:
- Use Python 3.11 slim image
- Install all dependencies from requirements.txt
- Copy application files
- Expose port 5000

## 🐳 Running the Container

### Option 1: Using Docker Compose (Recommended)

```bash
docker compose up
```

To run in detached mode:
```bash
docker compose up -d
```

To stop:
```bash
docker compose down
```

### Option 2: Using Docker Run (Cloud Environments)

**Linux/Mac/Cloud:**
```bash
docker run -d \
  --name workflow-generator \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e GEMINI_MODEL=gemini-2.5-flash \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/client:/app/client \
  -v $(pwd)/server:/app/server \
  --restart unless-stopped \
  workflow-generator
```

**Windows PowerShell:**
```powershell
docker run -d `
  --name workflow-generator `
  -p 5000:5000 `
  -e GEMINI_API_KEY=your_api_key_here `
  -e GEMINI_MODEL=gemini-2.5-flash `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/client:/app/client `
  -v ${PWD}/server:/app/server `
  --restart unless-stopped `
  workflow-generator
```

**Useful Docker Run Commands:**
```bash
# View logs
docker logs -f workflow-generator

# Stop container
docker stop workflow-generator

# Start container
docker start workflow-generator

# Restart container
docker restart workflow-generator

# Remove container
docker rm -f workflow-generator

# Check status
docker ps -a | grep workflow-generator
```

## 🌐 Accessing the Application

Once the container is running:

- **Web Interface:** http://localhost:5000
- **API Health Check:** http://localhost:5000/api/health

## 📁 Volume Mounts

The Docker setup includes these volume mounts:

- `./data:/app/data` - Persistent workflow storage
- `./client:/app/client` - Frontend files (for development)
- `./server:/app/server` - Server files (for development)

For production, remove the client and server mounts from docker-compose.yml

## 🔍 Monitoring and Logs

### View Logs
```bash
# Docker Compose
docker compose logs -f

# Docker Run
docker logs -f workflow-generator
```

### Check Container Status
```bash
docker ps
```

### Health Check
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "engine_ready": true,
  "gemini_available": true
}
```

## 🛠️ Troubleshooting

### Docker Compose Command Not Found (Cloud Environments)

If you see `bash: docker-compose: command not found`:

**Solution 1: Use modern docker compose (with space):**
```bash
docker compose up
```

**Solution 2: Use docker run instead:**
```bash
docker run -d \
  --name workflow-generator \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e GEMINI_MODEL=gemini-2.5-flash \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/client:/app/client \
  -v $(pwd)/server:/app/server \
  --restart unless-stopped \
  workflow-generator
```

### Container Won't Start

1. Check if port 5000 is already in use:
   ```bash
   # Windows
   netstat -ano | findstr :5000
   
   # Linux/Mac
   lsof -i :5000
   ```

2. Check Docker logs:
   ```bash
   docker logs workflow-generator
   ```

### API Key Issues

If you see "GEMINI_API_KEY: NO" in the logs:

1. Verify your `.env` file exists and contains the API key
2. Restart the container:
   ```bash
   docker compose restart
   # or
   docker restart workflow-generator
   ```

### Configuration File Not Found

Ensure `config/data.json` exists in your project directory. The container expects this structure:
```
.
├── config/
│   └── data.json
├── server/
│   └── app.py
├── client/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── data/
├── Dockerfile
└── docker-compose.yml
```

## 🧹 Cleanup

### Stop and remove container:
```bash
# Docker Compose
docker compose down

# Docker Run
docker stop workflow-generator
docker rm workflow-generator
```

### Remove image:
```bash
docker rmi workflow-generator
```

### Clean up all Docker resources:
```bash
docker system prune -a
```

## 🚀 Production Deployment

For production:

1. Remove development volume mounts from `docker-compose.yml`:
   ```yaml
   # Comment out or remove these lines:
   # - ./client:/app/client
   # - ./server:/app/server
   ```

2. Use environment variables for secrets (don't use .env file):
   ```bash
   export GEMINI_API_KEY=your_key
   export GEMINI_MODEL=gemini-2.5-flash
   docker compose up -d
   ```

3. Consider using Docker secrets or a secrets management service

4. Set up a reverse proxy (nginx, traefik) for HTTPS

5. Configure proper logging and monitoring

## 📝 API Endpoints

- `GET /api/health` - Health check
- `POST /api/generate` - Generate workflow from description
  ```json
  {
    "description": "Bug reporting workflow",
    "domain": "customer_service"
  }
  ```
- `POST /api/validate` - Validate workflow structure
- `POST /api/export` - Export workflow to JSON/YAML/BPMN

## 🔐 Security Notes

- Never commit `.env` files
- Use secrets management in production
- The container runs as root by default - consider creating a non-root user
- Keep your API keys secure
- Use HTTPS in production

## 📞 Support

For issues or questions:
1. Check the logs first
2. Verify your configuration
3. Ensure all dependencies are installed
4. Check Docker and Docker Compose versions

## 📊 Resource Requirements

- **Memory:** Minimum 512MB, recommended 1GB
- **CPU:** 1 core minimum, 2+ cores recommended
- **Disk:** ~500MB for image + data storage
