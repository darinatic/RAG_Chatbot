#!/bin/sh

# Get model from environment variable (default to gemma3:4b-it-qat)
MODEL=${OLLAMA_MODEL:-"gemma3:4b-it-qat"}

echo "Starting Ollama server..."
ollama serve &

# Wait for server to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama server is ready!"

# Verify model is available (should be pre-installed during build)
if ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL is ready to use!"
else
    echo "Warning: Model $MODEL not found, pulling now..."
    ollama pull "$MODEL"
fi

echo "Available models:"
ollama list

echo "Ollama is ready to serve requests with model: $MODEL"

# Keep Ollama running in foreground
wait