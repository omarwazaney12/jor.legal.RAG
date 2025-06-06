#!/bin/bash

# Jordan Legal RAG Deployment Script
# This script helps set up and deploy the application

set -e

echo "🚀 Jordan Legal RAG Deployment Script"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable not set"
    echo "   Please set it before running the application:"
    echo "   export OPENAI_API_KEY='your_api_key_here'"
    echo ""
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p cbj-scraper/chroma_db
mkdir -p cbj-scraper/mit_jordan_data/txt_output

# Set environment variables for development
export FLASK_ENV=development
export PORT=5003

echo "✅ Setup complete!"
echo ""
echo "🌟 To run locally:"
echo "   cd cbj-scraper"
echo "   python advanced_web_demo.py"
echo ""
echo "🌐 To deploy on Render:"
echo "   1. Push this code to GitHub"
echo "   2. Connect GitHub repo to Render"
echo "   3. Set OPENAI_API_KEY in Render environment variables"
echo "   4. Deploy using render.yaml configuration"
echo ""
echo "🔴 To deploy on Heroku:"
echo "   1. heroku create your-app-name"
echo "   2. heroku config:set OPENAI_API_KEY='your_key'"
echo "   3. git push heroku main"
echo ""
echo "📖 Check README.md for detailed instructions" 