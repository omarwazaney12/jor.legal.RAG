#!/usr/bin/env python3
"""Simple ChromaDB backup creator"""

import zipfile
import os
from pathlib import Path
from datetime import datetime

def create_backup():
    """Create a compressed backup of ChromaDB"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'chroma_backup_{timestamp}.zip'
    
    print(f'Creating {backup_file}...')
    
    with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in Path('chroma_db').rglob('*'):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to('.'))
    
    backup_size = os.path.getsize(backup_file) / 1024 / 1024
    print(f'✅ Backup created: {backup_file} ({backup_size:.1f} MB)')
    return backup_file

if __name__ == "__main__":
    create_backup() 