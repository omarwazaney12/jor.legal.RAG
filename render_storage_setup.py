#!/usr/bin/env python3
"""
Render Persistent Storage Setup Script for ChromaDB
This script helps upload your ChromaDB to Render's persistent storage
"""

import os
import sys
import tarfile
import subprocess
from pathlib import Path
import shutil
from datetime import datetime

class RenderStorageManager:
    """Manage ChromaDB upload to Render persistent storage"""
    
    def __init__(self):
        self.chroma_path = Path("chroma_db")
        self.backup_path = Path("chroma_backup")
        self.service_name = "jordan-legal-rag"  # From render.yaml
        
    def validate_setup(self):
        """Validate local setup before upload"""
        print("🔍 Validating setup...")
        
        # Check ChromaDB exists
        if not self.chroma_path.exists():
            print(f"❌ ChromaDB not found at {self.chroma_path}")
            return False
            
        # Check ChromaDB size
        size = self.get_directory_size(self.chroma_path)
        size_mb = size / (1024 * 1024)
        print(f"📊 ChromaDB size: {size_mb:.1f} MB")
        
        if size_mb > 900:  # Leave some buffer for 1GB limit
            print(f"⚠️  ChromaDB size ({size_mb:.1f} MB) is close to 1GB limit")
            
        # Check Render CLI
        try:
            result = subprocess.run(['render', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Render CLI detected")
            else:
                print("❌ Render CLI not found. Install from: https://render.com/docs/cli")
                return False
        except FileNotFoundError:
            print("❌ Render CLI not found. Install from: https://render.com/docs/cli")
            return False
            
        print("✅ Setup validation complete")
        return True
    
    def get_directory_size(self, path: Path) -> int:
        """Calculate directory size in bytes"""
        total = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total
    
    def create_backup(self):
        """Create compressed backup of ChromaDB"""
        print("📦 Creating ChromaDB backup...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"chroma_backup_{timestamp}.tar.gz"
        
        with tarfile.open(backup_file, 'w:gz') as tar:
            tar.add(self.chroma_path, arcname='chroma_db')
            
        backup_size = Path(backup_file).stat().st_size / (1024 * 1024)
        print(f"✅ Backup created: {backup_file} ({backup_size:.1f} MB)")
        return backup_file
    
    def upload_to_render(self, backup_file: str):
        """Upload ChromaDB to Render persistent storage"""
        print("🚀 Uploading to Render persistent storage...")
        
        commands = [
            "# 1. Connect to your Render service shell",
            f"render shell {self.service_name}",
            "",
            "# 2. Inside the shell, create temp directory and download backup",
            "mkdir -p /tmp/upload",
            "cd /tmp/upload",
            "",
            "# 3. Upload your backup file using one of these methods:",
            "",
            "# Method A: Upload via SCP (if you have SSH access)",
            f"# scp {backup_file} user@your-server:/tmp/upload/",
            "",
            "# Method B: Upload via curl (you'll need to host the file somewhere)",
            f"# curl -o chroma_backup.tar.gz 'YOUR_FILE_URL'",
            "",
            "# Method C: Direct file transfer through Render dashboard",
            "# Use the file manager in Render dashboard",
            "",
            "# 4. Extract to persistent storage location",
            "cd /opt/render/project/src/",
            "tar -xzf /tmp/upload/chroma_backup.tar.gz",
            "",
            "# 5. Verify the upload",
            "ls -la /opt/render/project/src/chroma_db/",
            "",
            "# 6. Clean up",
            "rm -rf /tmp/upload",
            "",
            "# 7. Restart your service",
            "exit  # Exit the shell",
            f"render deploy {self.service_name}"
        ]
        
        print("\n📋 Execute these commands to upload ChromaDB:")
        print("=" * 60)
        for cmd in commands:
            print(cmd)
        print("=" * 60)
        
    def generate_render_instructions(self):
        """Generate comprehensive Render setup instructions"""
        instructions = f"""
🚀 RENDER PERSISTENT STORAGE SETUP GUIDE
========================================

Your Render service is already configured with persistent storage!

Current Configuration (from render.yaml):
- Service: {self.service_name}
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
   • render shell {self.service_name}
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
   render deploy {self.service_name}

6. Verify deployment:
   • Check logs: render logs {self.service_name}
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
"""
        
        print(instructions)
        
        # Save to file
        with open("RENDER_SETUP_GUIDE.md", "w", encoding="utf-8") as f:
            f.write(instructions)
        print("📝 Instructions saved to: RENDER_SETUP_GUIDE.md")

def main():
    manager = RenderStorageManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--backup":
        # Create backup only
        if manager.validate_setup():
            backup_file = manager.create_backup()
            print(f"✅ Backup ready: {backup_file}")
    else:
        # Full guide
        print("🚀 Render Persistent Storage Setup")
        print("=" * 40)
        
        if manager.validate_setup():
            backup_file = manager.create_backup()
            manager.upload_to_render(backup_file)
            
        manager.generate_render_instructions()

if __name__ == "__main__":
    main() 