#!/bin/sh

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &

# Wait for server to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama server is ready, pulling gemma3:4b model..."
ollama pull gemma3:4b

echo "Model pulled successfully!"
ollama list

echo "Ollama is ready to serve requests!"

# Keep Ollama running in foreground
wait