# Semantic Search Application using RAG

A powerful semantic search application leveraging Retrieval Augmented Generation (RAG) for intelligent context-aware information retrieval. This application uses embeddings stored in a vector database for efficient and accurate semantic search capabilities.

## 🚀 Features

- **Semantic Search**: Advanced text similarity search using vector embeddings
- **RAG Integration**: Retrieval Augmented Generation for context-aware responses
- **Vector Database**: Efficient storage and retrieval of document embeddings
- **Multiple Document Formats**: Support for PDF, TXT, DOCX, and more
- **Real-time Processing**: Fast document indexing and search capabilities
- **Web Interface**: User-friendly interface for document upload and search

## 📋 Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.8+** (Python 3.9+ recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Farooqbaqer9/semantic-search-application-using-RAG-.git
cd semantic-search-application-using-RAG-
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If `requirements.txt` doesn't exist yet, install the core dependencies manually:

```bash
pip install streamlit openai chromadb langchain sentence-transformers pypdf2 python-docx pandas numpy
```

### 4. Environment Configuration

Create a `.env` file in the root directory and add your API keys:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here  # Optional
```

**Getting API Keys:**
- **OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) to create an account and generate an API key
- **Hugging Face Token** (Optional): Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens) for additional model access

## 🚀 Getting Started

### 1. Initialize the Vector Database

```bash
python setup_database.py
```

### 2. Upload Documents

Place your documents in the `documents/` folder or use the web interface to upload files.

Supported formats:
- PDF (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`)
- Markdown files (`.md`)

### 3. Run the Application

```bash
# Start the web interface
streamlit run app.py

# Alternative: Run with Flask (if available)
python flask_app.py
```

The application will be available at `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask).

## 💡 Usage

### Web Interface

1. **Upload Documents**: Use the file uploader to add documents to your knowledge base
2. **Process Documents**: Click "Process Documents" to create embeddings
3. **Search**: Enter your query in the search box
4. **View Results**: Browse through semantically relevant results with context

### API Usage (if implemented)

```python
import requests

# Search endpoint
response = requests.post('http://localhost:5000/search', 
                        json={'query': 'your search query'})
results = response.json()
```

## 📁 Project Structure

```
semantic-search-application-using-RAG-/
├── app.py                 # Main Streamlit application
├── flask_app.py          # Flask API (alternative)
├── setup_database.py     # Database initialization
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
├── documents/           # Document storage folder
├── database/            # Vector database storage
├── src/                 # Source code
│   ├── __init__.py
│   ├── embeddings.py    # Embedding generation
│   ├── retrieval.py     # Document retrieval logic
│   ├── processing.py    # Document processing
│   └── utils.py         # Utility functions
├── tests/               # Test files
└── README.md           # This file
```

## ⚙️ Configuration

### Vector Database Options

The application supports multiple vector database backends:

1. **ChromaDB** (Default - Local)
2. **Pinecone** (Cloud-based)
3. **FAISS** (Facebook AI Similarity Search)

To change the vector database, modify the configuration in `config.py`:

```python
VECTOR_DB = "chromadb"  # Options: "chromadb", "pinecone", "faiss"
```

### Model Configuration

Customize the embedding model in the configuration:

```python
# Default embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Alternative models:
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better accuracy
# EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI (requires API key)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **API Key Errors**
   - Ensure your `.env` file is in the root directory
   - Verify your API keys are valid and have sufficient credits

3. **Memory Issues with Large Documents**
   - Increase chunk size in document processing
   - Consider using a more powerful machine or cloud deployment

4. **Slow Search Performance**
   - Reduce the number of retrieved documents
   - Use a more efficient embedding model
   - Consider upgrading your vector database setup

### Performance Optimization

- **Batch Processing**: Process multiple documents simultaneously
- **Caching**: Enable embedding caching for frequently accessed documents
- **Index Optimization**: Regularly optimize your vector database indices

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_retrieval.py

# Run with coverage
pip install pytest-cov
python -m pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Farooqbaqer9/semantic-search-application-using-RAG-/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer: [Your Email]

## 🚧 Roadmap

- [ ] Advanced document preprocessing
- [ ] Multi-language support
- [ ] Real-time collaborative features
- [ ] Advanced analytics dashboard
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Integration with more LLM providers

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- [Streamlit](https://streamlit.io/) for the web interface

---

**Built with ❤️ for intelligent information retrieval**
