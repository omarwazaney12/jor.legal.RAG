# Render Deployment Workflow: Pre-built ChromaDB

## Overview

This workflow builds ChromaDB embeddings **locally** with Render-compatible schema, then deploys the pre-built database to Render.

### Why This Approach?
- ✅ **No FTS5 issues** on Render
- ✅ **Fast deployment** - no embedding creation on Render
- ✅ **Predictable schema** - database built with exact configuration
- ✅ **Resource efficient** - heavy computation done locally

## Step-by-Step Workflow

### 1. **Build Embeddings Locally**

```bash
# Install dependencies locally
pip install -r requirements.txt

# Build the pre-built database with Render-compatible schema
python build_embeddings_locally.py
```

**What this does:**
- Creates `./chroma_db_render/` with DuckDB backend
- Processes all legal documents and creates embeddings
- Uses the exact schema configuration that works on Render
- Verifies compatibility with Render's expected configuration

### 2. **Test the Pre-built Database**

```bash
# Test deployment system with pre-built database
python deploy_without_docs.py
```

**Expected output:**
```
🧪 Testing Deployment RAG System
✅ Database loaded successfully
🔍 Testing query: ما هي شروط تأسيس الشركات؟
✅ Query result:
   Confidence: 0.85
   Sources: 5
   Answer preview: بناءً على الوثائق القانونية المتاحة...
```

### 3. **Commit and Deploy**

```bash
# Add the pre-built database to git
git add chroma_db_render/
git add .
git commit -m "Add pre-built ChromaDB database with Render-compatible schema"

# Push to trigger Render deployment
git push
```

### 4. **Verify on Render**

After deployment, check Render logs should show:
```
📂 Loading pre-built database from: /opt/render/project/src/chroma_db
✅ Found 1 collections
✅ Loaded collection 'jordanian_legal_docs' with 1500+ documents
```

## File Structure

```
├── build_embeddings_locally.py    # Builds database locally
├── deploy_without_docs.py          # Deployment system (loads pre-built DB)
├── advanced_rag_system.py          # Main RAG system (for local development)
├── chroma_db_render/               # Pre-built database (COMMIT THIS)
│   ├── chroma.sqlite3             # DuckDB database files
│   └── ...                        # Other ChromaDB files
└── mit_jordan_data/               # Source documents
```

## Key Configuration

### Local Building (build_embeddings_locally.py)
```python
# Uses DuckDB backend compatible with Render
os.environ['CHROMA_DB_PATH'] = "./chroma_db_render"

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(
        chroma_db_impl="duckdb+parquet",  # Render-compatible
        allow_reset=True
    )
)
```

### Render Deployment (deploy_without_docs.py)
```python
# Loads the pre-built database
chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db_render')

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(
        chroma_db_impl="duckdb+parquet",  # Same as local
        allow_reset=True
    )
)
```

## Render Configuration

### Environment Variables
```yaml
# In render.yaml
envVars:
  - key: CHROMA_DB_PATH
    value: /opt/render/project/src/chroma_db_render
```

### Mount Point
```yaml
# In render.yaml  
disk:
  name: jordan-legal-data
  mountPath: /opt/render/project/src/chroma_db_render
  sizeGB: 1
```

## Troubleshooting

### Database Not Found on Render
```bash
# Check if files were committed
ls -la chroma_db_render/

# Check Render environment
echo $CHROMA_DB_PATH
ls -la /opt/render/project/src/chroma_db_render/
```

### Schema Mismatch
```bash
# Rebuild locally with clean state
rm -rf chroma_db_render/
python build_embeddings_locally.py

# Commit the new database
git add chroma_db_render/
git commit -m "Rebuild ChromaDB with fresh schema"
git push
```

### Performance Issues
The pre-built database should be **fast to load**:
- Database load: < 5 seconds
- First query: < 2 seconds  
- Subsequent queries: < 1 second

## Benefits

✅ **Eliminates FTS5 Issues** - DuckDB backend bypasses SQLite problems
✅ **Fast Deployment** - No embedding computation on Render
✅ **Predictable Results** - Same database that worked locally
✅ **Version Control** - Database changes are tracked in git
✅ **Easy Rollback** - Can revert to previous database versions

🎉 **Result: Reliable ChromaDB deployment on Render!** 