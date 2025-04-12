FROM ollama/ollama:latest

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy startup script
COPY scripts/start_ollama.sh /
RUN chmod +x /start_ollama.sh

# Set default model
ARG OLLAMA_MODEL=deepseek-r1:1.5b
ENV OLLAMA_MODEL=$OLLAMA_MODEL
ENV OLLAMA_KEEP_ALIVE=5m

# Run startup script as entrypoint
ENTRYPOINT ["/start_ollama.sh"]


