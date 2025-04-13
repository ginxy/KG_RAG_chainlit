FROM python:3.10-slim

WORKDIR /app

# Create directories
RUN mkdir -p /app/logs /app/data

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    poppler-utils \
    libpoppler-cpp-dev \
    libpq-dev \
    libenchant-2-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure proper permissions
RUN chmod +x /app/scripts/*.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG
ENV LOG_FILE=/app/logs/app.log
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV MCP_AUTH_TOKEN=supersecrettoken
ENV MCP_HOST=http://mcp:7432


# Create a healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/_health || exit 1

# Run the application
CMD ["chainlit", "run", "src/main.py", "--port", "8000", "--host", "0.0.0.0"]