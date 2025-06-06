#!/bin/bash
echo "🔧 Build Fix Script - Working around directory structure"

# Use the root directory requirements.txt since we have all files there
echo "📦 Installing dependencies from root directory..."
pip install -r requirements.txt

echo "✅ Build completed successfully!" 