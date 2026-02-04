#!/bin/bash
# Script to install and configure Ollama on macOS/Linux

echo "Ollama Setup for Data Anonymization"
echo "======================================"

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "Ollama is already installed"
    ollama --version
else
    echo "Installing Ollama..."
    
    # Install Ollama
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS. Downloading installer..."
        echo "Visit: https://ollama.ai/download"
        echo "Or use: brew install ollama"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux. Installing via curl..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Operating system not automatically supported"
        echo "Visit https://ollama.ai/download for instructions"
        exit 1
    fi
fi

# Start Ollama server in background
echo ""
echo "🔧 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 3

# Download default model
MODEL_NAME=${1:-"llama3.2"}
echo ""
echo "Downloading model: $MODEL_NAME"
echo "   (This may take a few minutes...)"
ollama pull $MODEL_NAME

# Verify model was downloaded
echo ""
echo "Available models:"
ollama list

echo ""
echo "Setup complete!"
echo ""
echo "To use other models:"
echo "  ollama pull llama3.1"
echo "  ollama pull mistral"
echo "  ollama pull gemma2"
echo ""
echo "To stop Ollama:"
echo "  kill $OLLAMA_PID"
