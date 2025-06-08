#!/bin/bash
echo "🚀 Starting Jordan Legal Assistant..."
echo "📂 Current directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo "📦 Installed packages:"
pip list | grep -E "(Flask|openai|chromadb)"
echo "🌐 Starting Flask app on port $PORT"

# Start the Flask application
exec python3 main.py 