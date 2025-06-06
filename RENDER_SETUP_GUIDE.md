
🚀 RENDER PERSISTENT STORAGE SETUP GUIDE
========================================

Your Render service is already configured with persistent storage!

Current Configuration (from render.yaml):
- Service: jordan-legal-rag
- Mount Path: /opt/render/project/src/chroma_db
- Storage Size: 1 GB
- Environment Variable: CHROMA_DB_PATH=/opt/render/project/src/chroma_db

📋 STEP-BY-STEP UPLOAD PROCESS:

1. Install Render CLI (if not already installed):
   npm install -g @render/cli
   
2. Login to Render:
   render auth login

3. Create backup of your ChromaDB:
   python render_storage_setup.py --backup

4. Upload Options:

   Option A - Direct Shell Upload:
   • render shell jordan-legal-rag
   • mkdir -p /tmp && cd /tmp
   • [Upload your backup file here using file manager or curl]
   • cd /opt/render/project/src/
   • tar -xzf /tmp/chroma_backup_*.tar.gz
   • ls -la chroma_db/  # Verify
   • exit

   Option B - Git LFS (Recommended for version control):
   • Add ChromaDB to Git LFS
   • Push to your repository
   • Redeploy service

   Option C - External Storage:
   • Upload backup to cloud storage (S3, Google Drive, etc.)
   • Download in build script

5. Restart your service:
   render deploy jordan-legal-rag

6. Verify deployment:
   • Check logs: render logs jordan-legal-rag
   • Test your application

🔧 TROUBLESHOOTING:

If ChromaDB not found:
- Check mount path: /opt/render/project/src/chroma_db
- Verify CHROMA_DB_PATH environment variable
- Check file permissions

If size limit exceeded:
- Current limit: 1 GB
- Upgrade storage plan if needed
- Consider data compression

🌟 YOUR APP IS CONFIGURED FOR:
- Automatic path detection (local dev vs. Render)
- Environment-based configuration  
- Persistent storage mounting
- Deployment-ready setup

Next: Upload your ChromaDB and deploy! 🚀
