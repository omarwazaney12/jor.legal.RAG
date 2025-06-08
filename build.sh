#!/bin/bash
# Build script for Render deployment
# This script handles the directory structure issue and ChromaDB initialization

echo "🔧 Setting up build environment..."

# Create cbj-scraper directory if it doesn't exist
if [ ! -d "cbj-scraper" ]; then
    echo "📁 Creating cbj-scraper directory..."
    mkdir -p cbj-scraper
fi

# Copy requirements.txt to cbj-scraper if it doesn't exist there
if [ ! -f "cbj-scraper/requirements.txt" ]; then
    echo "📋 Copying requirements.txt to cbj-scraper..."
    cp requirements.txt cbj-scraper/
fi

# Install dependencies
echo "📦 Installing dependencies..."
cd cbj-scraper && pip install -r requirements.txt

# Initialize ChromaDB with Render-compatible settings
echo "🗄️  Initializing ChromaDB for Render..."
cd .. && python3 render_chromadb_init.py

echo "✅ Build completed successfully!" 