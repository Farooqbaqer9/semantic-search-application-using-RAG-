# Railway Persistent Storage Explanation

## 🖥️ Railway Infrastructure vs Your Laptop

### What Happens When You Deploy:

1. **Code Upload**: Your code goes to Railway's servers (NOT your laptop)
2. **Container Creation**: Railway creates a Linux container in their cloud
3. **App Execution**: Your app runs 24/7 on Railway's servers
4. **Data Storage**: ChromaDB files are stored on Railway's persistent disk

### 📁 File System Structure on Railway:

```
/app/                          (Your application directory)
├── app.py                     (Your Streamlit app)
├── requirements.txt           (Dependencies)
├── database/                  (ChromaDB storage - PERSISTENT)
│   ├── chroma.sqlite3         (Vector database file)
│   ├── *.parquet              (Document embeddings)
│   └── metadata/              (Document metadata)
└── src/                       (Your modules)
```

### 🔄 What Happens When Your Laptop is OFF:

✅ **Your app keeps running** - Railway servers are always on
✅ **Users can still access** - 24/7 availability  
✅ **Data persists** - ChromaDB files stay on Railway's disk
✅ **New uploads work** - Users can add documents anytime
✅ **Search continues** - All functionality available

### 💡 Key Concepts:

#### **Local Development** (laptop on):
- Code on your laptop
- Test database on your laptop
- Development only

#### **Production Deployment** (laptop can be off):
- Code runs on Railway's servers
- Database files on Railway's persistent storage
- Available worldwide 24/7

## 🗃️ Railway's Persistent Storage Details

### **Volume Mounting:**
```yaml
# Railway automatically mounts persistent storage
Container Path: /app/database/
Railway Storage: Persistent SSD volume
Backup: Automatic snapshots
Access: Only your application
```

### **Data Persistence Guarantees:**
- ✅ **Survives app restarts**
- ✅ **Survives code deployments** 
- ✅ **Survives Railway maintenance**
- ✅ **Available across container recreation**
- ✅ **Backed up by Railway**

### **Storage Specifications:**
- **Type**: SSD-based persistent volumes
- **Performance**: High IOPS for fast vector searches
- **Durability**: Railway handles redundancy/backups
- **Size**: Starts at 1GB (free tier), scalable

## 🔍 Data Flow Example:

### **User Uploads Document** (your laptop OFF):
1. User visits your Railway URL
2. Uploads PDF through Streamlit interface
3. Railway server processes the document
4. ChromaDB saves vectors to `/app/database/`
5. Data persists on Railway's SSD storage

### **User Searches** (your laptop OFF):
1. User enters search query
2. Railway server generates query embedding
3. ChromaDB searches persistent vector database
4. Results returned from Railway's storage
5. Gemini API generates response

## 📊 Railway vs Other Hosting

| Feature | Railway | Your Laptop | Heroku | Vercel |
|---------|---------|-------------|---------|---------|
| 24/7 Uptime | ✅ | ❌ | ✅ | ✅ |
| Persistent DB | ✅ | N/A | ❌* | ❌ |
| Auto-scaling | ✅ | ❌ | ✅ | ✅ |
| Zero Config | ✅ | N/A | ❌ | ❌ |

*Heroku requires separate database service

## 🛡️ Data Safety on Railway

### **Railway's Responsibilities:**
- Server uptime (99.9% SLA)
- Data backup and recovery
- Hardware maintenance
- Security and encryption
- DDoS protection

### **Your Responsibilities:**
- Application code quality
- API key security
- Data validation
- User access control

## 🔄 Development Workflow

### **While Developing** (laptop on):
```bash
# Local development
streamlit run app.py
# Test with local database in ./database/
```

### **After Deployment** (laptop can be off):
```bash
# Push changes
git push origin main
# Railway auto-deploys
# Users access: https://yourapp.railway.app
# Data persists in Railway's cloud storage
```

## 💰 Storage Costs

### **Railway Storage Pricing:**
- **Free Tier**: 1GB persistent storage included
- **Pro Plan**: $0.25/GB/month for additional storage
- **Example**: 10GB storage = ~$2.50/month

### **Storage Growth Estimates:**
- **1,000 documents**: ~100-200MB
- **10,000 documents**: ~1-2GB  
- **100,000 documents**: ~10-20GB

## 🚨 Important Notes

### **Your Laptop Role After Deployment:**
- ❌ **NOT hosting the app** - Railway hosts it
- ❌ **NOT storing data** - Railway stores it
- ✅ **Development only** - Code changes and testing
- ✅ **Git pushes** - Trigger automatic deployments

### **Data Accessibility:**
- **Always Available**: Users worldwide can access 24/7
- **Independent**: Works regardless of your laptop status
- **Scalable**: Can handle multiple concurrent users
- **Reliable**: Railway's enterprise infrastructure
