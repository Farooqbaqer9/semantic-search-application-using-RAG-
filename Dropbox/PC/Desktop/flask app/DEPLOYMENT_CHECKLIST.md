# Railway Deployment Checklist

## ✅ Pre-Deployment Checklist

### Files Created/Updated:
- ✅ `Procfile` - Railway startup command
- ✅ `railway.json` - Railway configuration
- ✅ `requirements.txt` - All dependencies listed
- ✅ `.gitignore` - Test files excluded
- ✅ `app.py` - Main application ready

### Configuration Verified:
- ✅ ChromaDB uses local path (`./database`) - will persist on Railway
- ✅ Streamlit configured for web deployment
- ✅ Gemini API integration ready
- ✅ All dependencies in requirements.txt

## 🚀 Deployment Steps

### 1. Railway Setup
1. Go to [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository

### 2. Environment Variables
In Railway dashboard, add:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Automatic Deployment
Railway will:
- Install Python dependencies from requirements.txt
- Use Procfile to start Streamlit
- Provide a public URL
- Enable automatic deployments on git push

### 4. Verify Deployment
- Check Railway logs for any errors
- Visit the provided URL
- Test document upload and search

## 💰 Cost Estimation

### Railway Free Tier:
- ✅ 512MB RAM (sufficient for your app)
- ✅ 1GB disk storage (for ChromaDB)
- ✅ 500 hours/month usage
- ✅ Custom domain support

### Expected Usage:
- **RAM**: ~300-400MB (Streamlit + ChromaDB + ML models)
- **Storage**: Grows with uploaded documents
- **CPU**: Low usage for typical queries

## 🔧 Potential Issues & Solutions

### Issue 1: Memory Limits
**Problem**: App crashes due to memory
**Solution**: Upgrade to Railway Pro ($5/month for 1GB RAM)

### Issue 2: Slow First Load
**Problem**: Cold start downloads ML models
**Solution**: Normal behavior, subsequent loads are fast

### Issue 3: File Upload Limits
**Problem**: Large documents fail to upload
**Solution**: Add file size validation in app

### Issue 4: Database Growing Too Large
**Problem**: ChromaDB files exceed storage limit
**Solution**: 
- Implement document cleanup
- Or upgrade storage
- Or migrate to external vector DB

## 📊 Monitoring

### Railway Dashboard Shows:
- CPU/Memory usage
- Network traffic
- Storage usage
- Application logs
- Deployment status

### Key Metrics to Watch:
- Memory usage (keep under 80% of limit)
- Storage growth rate
- Response times
- Error rates

## 🔄 Updates & Maintenance

### Automatic Deployments:
- Push to GitHub → Railway auto-deploys
- No manual intervention needed
- Zero-downtime deployments

### Database Backups:
- ChromaDB files persist automatically
- Consider periodic exports for safety

## 🌐 Access Your App

After deployment:
1. Railway provides a URL like: `https://yourapp-production.up.railway.app`
2. Optionally add custom domain
3. Share with users!

## ✨ Success Criteria

Your deployment is successful when:
- ✅ App loads without errors
- ✅ Document upload works
- ✅ Search returns relevant results
- ✅ RAG responses are generated
- ✅ UI displays correctly with animations
- ✅ Database persists between restarts
