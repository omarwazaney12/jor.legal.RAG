#!/usr/bin/env python3
"""
Railway Deployment Entry Point
Advanced Legal RAG System for Jordanian Laws
"""

import os
from advanced_web_demo import app

if __name__ == '__main__':
    # Get port from Railway environment
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Jordan Legal Assistant on Railway...")
    print(f"🌐 Server starting on port {port}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    ) 