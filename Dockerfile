# Unified Workflow Generator Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server/ ./server/
COPY client/ ./client/
COPY config/ ./config/
COPY data/ ./data/

# Environment variables will be passed via docker-compose or docker run

# Expose port
EXPOSE 5000

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Run the unified server
CMD ["python", "server/app.py"]
