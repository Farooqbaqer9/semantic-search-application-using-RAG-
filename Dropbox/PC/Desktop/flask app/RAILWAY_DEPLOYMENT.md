# Railway Deployment Guide for RAG Application

## Overview
Your RAG application can be deployed on Railway using ChromaDB for vector storage (no Supabase needed). Railway will provide persistent storage for your ChromaDB database.

## Prerequisites
1. Railway account (free tier available)
2. GitHub repository with your code
3. Gemini API key

## Deployment Steps

### 1. Prepare Your Application

Your app is already configured correctly with:
- ✅ ChromaDB for vector database (local storage)
- ✅ Streamlit for web interface
- ✅ Gemini API for LLM functionality
- ✅ All dependencies in requirements.txt

### 2. Create Railway Configuration

Create these files in your project root:

#### `Procfile` (Railway startup command)
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

#### `railway.json` (Optional configuration)
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "numReplicas": 1,
    "sleepApplication": false,
    "restartPolicyType": "always"
  }
}
```

### 3. Environment Variables

Set these in Railway dashboard:
- `GEMINI_API_KEY`: Your Google Gemini API key
- `PORT`: 8080 (Railway will set this automatically)

### 4. Database Persistence

Railway provides persistent volumes. Your ChromaDB data will be stored in `/app/database/` and persist between deployments.

### 5. Memory and Storage

- **Memory**: 512MB-1GB should be sufficient
- **Storage**: Railway provides persistent disk storage
- **Database**: ChromaDB files will be stored locally and persist

## Deployment Process

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect Railway**: Link your GitHub repo to Railway
3. **Set Environment Variables**: Add your GEMINI_API_KEY
4. **Deploy**: Railway will automatically build and deploy

## Cost Considerations

**Railway Free Tier Includes:**
- 512MB RAM
- 1GB disk storage
- 500 hours of usage per month
- Persistent storage

**Estimated Monthly Cost:**
- Free tier: $0 (with usage limits)
- Paid: ~$5-10/month for small apps

## Performance Optimizations

1. **Startup Time**: First load may be slow due to model downloads
2. **Memory Usage**: Sentence transformer models use ~200MB RAM
3. **Storage**: ChromaDB files will grow with document uploads

## Monitoring

Railway provides:
- Application logs
- Resource usage metrics
- Uptime monitoring
- Custom domains (paid plans)

## Advantages of This Setup

✅ **No External Database Needed**: ChromaDB runs locally
✅ **Simple Deployment**: Single container deployment
✅ **Cost Effective**: No additional database costs
✅ **Fast Queries**: Local vector database is very fast
✅ **Easy Scaling**: Can upgrade resources as needed

## Alternative Database Options (if needed later)

If you need to scale beyond local storage:
- **Pinecone**: Managed vector database (~$70/month)
- **Weaviate Cloud**: Managed vector database (~$25/month)
- **Qdrant Cloud**: Managed vector database (~$20/month)

But for most use cases, ChromaDB on Railway will work perfectly!
