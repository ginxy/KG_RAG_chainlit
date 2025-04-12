#!/bin/sh
set -e

# Start Ollama server
echo "🔄 Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for Ollama server to start..."
sleep 5

# Pull the specified model
echo "📥 Downloading ${OLLAMA_MODEL}..."
ollama pull ${OLLAMA_MODEL}

# Verify model was downloaded
echo "✅ Verifying model..."
ollama list | grep -q ${OLLAMA_MODEL} || (echo "❌ Model download failed"; exit 1)

echo "✅ Ollama ready with model: ${OLLAMA_MODEL}"

# Keep container running
wait $SERVER_PID

##!/bin/sh
#set -e
#
## Start server in background
#ollama serve &
#
## Wait for server to initialize
#sleep 5
#
## Pull the model
#echo "Pulling ${OLLAMA_MODEL}..."
#ollama pull ${OLLAMA_MODEL}

