#!/usr/bin/env python3
"""
Railway Deployment Entry Point
Advanced Legal RAG System for Jordanian Laws
"""

import os
from advanced_web_demo import app, initialize_system

if __name__ == '__main__':
    # Get port from Railway environment
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Jordan Legal Assistant on Railway...")
    print(f"🌐 Server starting on port {port}")
    
    # Initialize the legal system BEFORE starting the server
    print("🔧 Initializing Legal RAG System...")
    initialize_system()
    print("✅ System initialization complete!")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    ) 