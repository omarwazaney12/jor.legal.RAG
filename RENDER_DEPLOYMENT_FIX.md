# Render Deployment Fix Guide

## Issues Identified

From the diagnostic output, we identified two critical issues preventing the Jordan Legal RAG application from working on Render:

### 1. ChromaDB Schema Mismatch
**Error**: `sqlite3.OperationalError: no such column: collections.topic`
**Cause**: The local ChromaDB version created a different database schema than what's expected by the ChromaDB version running on Render.

### 2. OpenAI Library Compatibility  
**Error**: `Client.__init__() got an unexpected keyword argument 'proxies'`
**Cause**: The OpenAIEmbeddings initialization uses parameters not supported by the OpenAI library version (1.6.1) in requirements.txt.

## Fixes Applied

### Fix 1: OpenAI Compatibility (✅ COMPLETED)
- Updated `advanced_rag_system.py` to remove unsupported parameters
- Added fallback mechanism for different OpenAI versions
- Changed from complex initialization to simple model specification

### Fix 2: ChromaDB Schema Fix Script (✅ CREATED)
- Created `fix_chromadb_schema.py` to reset and recreate ChromaDB with correct schema
- Script includes backup functionality to prevent data loss
- Provides verification and testing of the fix

## Deployment Steps

### Step 1: Apply Code Fixes (Already Done)
The OpenAI compatibility fix has been applied to `advanced_rag_system.py`.

### Step 2: Fix ChromaDB Schema on Render

Run this command on your Render service to fix the schema:

```bash
cd ~/project/src
python3 fix_chromadb_schema.py
```

**What this does:**
1. Creates a backup of existing ChromaDB
2. Resets the database to fix schema issues  
3. Creates a fresh collection with correct schema
4. Verifies the fix worked

### Step 3: Rebuild Embeddings (if needed)

If the schema fix empties your database, you'll need to rebuild embeddings:

```bash
cd ~/project/src
python3 -c "
from advanced_rag_system import AdvancedLegalRAGSystem
system = AdvancedLegalRAGSystem()
count = system.load_documents(force_rebuild=True)
print(f'Rebuilt {count} documents')
"
```

### Step 4: Test the Fix

Test ChromaDB connection:
```bash
python3 -c "
import chromadb
import os
from chromadb.config import Settings

chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
for collection in collections:
    count = client.get_collection(collection.name).count()
    print(f'{collection.name}: {count} items')
"
```

Test the full application:
```bash
python3 -c "
from deploy_without_docs import DeploymentRAGSystem
system = DeploymentRAGSystem()
result = system.load_embeddings_only()
print(f'System loaded: {result}')
if result:
    test = system.query('ما هي شروط تأسيس الشركات؟')
    print(f'Test query confidence: {test.confidence}')
"
```

### Step 5: Deploy Updated Code

Commit and push the fixes:
```bash
git add .
git commit -m "Fix OpenAI compatibility and ChromaDB schema issues for Render deployment"
git push
```

Then redeploy on Render Dashboard or via CLI:
```bash
render deploy --service jordan-legal-rag
```

## Verification Checklist

After deployment, verify these items work:

- [ ] Environment variable `CHROMA_DB_PATH` is set correctly
- [ ] ChromaDB files exist at the specified path
- [ ] ChromaDB connection works without schema errors
- [ ] OpenAI embeddings initialize without parameter errors
- [ ] Application loads embeddings successfully
- [ ] Query system returns results with confidence scores
- [ ] Web interface shows "System Ready" instead of "Limited Mode"

## Alternative: Fresh Database Approach

If the schema fix doesn't work, you can start fresh:

1. **Delete existing ChromaDB** on Render:
   ```bash
   rm -rf /opt/render/project/src/chroma_db/*
   ```

2. **Create minimal test collection**:
   ```bash
   python3 -c "
   import chromadb
   import os
   from chromadb.config import Settings
   
   chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
   client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))
   collection = client.create_collection('jordanian_legal_docs')
   print('Fresh collection created')
   "
   ```

3. **Re-upload documents** if you have the original text files, or
4. **Use deployment without full database** and manually add key documents

## Troubleshooting

### Issue: "System in limited mode" persists
- Check that ChromaDB files aren't corrupted
- Verify collection has embeddings: `collection.count() > 0`
- Test query directly against collection

### Issue: OpenAI embeddings still failing
- Check OpenAI API key is set: `echo $OPENAI_API_KEY`  
- Try using ada-002 model instead of text-embedding-3-large
- Check OpenAI account has API credits

### Issue: Files not accessible
- Verify persistent disk is mounted: `ls -la /opt/render/project/src/chroma_db`
- Check disk space: `df -h`
- Verify permissions: `ls -la /opt/render/project/src/`

## Expected Results

After applying these fixes:
- ✅ ChromaDB loads without schema errors
- ✅ OpenAI embeddings initialize successfully  
- ✅ Application shows "System Ready" status
- ✅ Query system returns accurate results
- ✅ Web interface is fully functional

The application should now work correctly on Render with persistent storage! 