import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
import docx
import pptx
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="Semantic Search RAG Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize vector database
@st.cache_resource
def init_vector_db():
    client = chromadb.PersistentClient(path="./database")
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# Custom CSS with abstract background and blue-orange theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    body {
        background-color: #0a0e27;
        background-image: 
            linear-gradient(135deg, rgba(255, 107, 53, 0.3) 0%, rgba(59, 130, 246, 0.4) 100%),
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 107, 53, 0.2) 0%, transparent 50%);
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        background-attachment: fixed;
        min-height: 100vh;
        overflow-x: hidden;
        position: relative;
    }
    
    .main > div {
        background: transparent;
        padding-top: 2rem;
    }
    
    /* Animated gradient border for containers */
    @keyframes borderGlow {
        0% { border-color: #FF6B35; box-shadow: 0 0 20px rgba(255, 107, 53, 0.3); }
        50% { border-color: #3B82F6; box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        100% { border-color: #FF6B35; box-shadow: 0 0 20px rgba(255, 107, 53, 0.3); }
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    .search-container {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 20px;
        border: 2px solid transparent;
        background-clip: padding-box;
        padding: 2.5rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        animation: fadeInUp 1.2s ease-out;
        position: relative;
        overflow: hidden;
        border: 2px solid transparent;
        background-clip: padding-box;
        background-clip: padding-box;
    }
    
    .search-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        border: 2px solid transparent;
        background: linear-gradient(45deg, #FF6B35, #3B82F6, #FF6B35) border-box;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: exclude;
        mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
        font-size: 16px;
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF6B35;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(255, 107, 53, 0.4);
        background: linear-gradient(135deg, #3B82F6 0%, #FF6B35 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    .response-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        animation: fadeInUp 0.8s ease-out;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .source-doc {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #FF6B35;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .source-doc:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(10px);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-5px);
    }
    
    .sidebar .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .sidebar .stFileUploader:hover {
        border-color: #FF6B35;
        background: rgba(255, 107, 53, 0.1);
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #FF6B35, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    .stMarkdown p {
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 107, 53, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        backdrop-filter: blur(20px);
    }
    
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #FF6B35, #3B82F6);
        border-radius: 10px;
    }
    
    /* Success/Error message styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    text = ""
    
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            prs = pptx.Presentation(uploaded_file)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
    
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def generate_rag_response(query, context):
    """Generate response using Gemini with retrieved context"""
    prompt = f"""Based on the following context, please answer the question comprehensively and accurately. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_metrics(collection):
    """Display database metrics"""
    try:
        count = collection.count()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF6B35; margin: 0;">📄</h3>
                <h2 style="margin: 0.5rem 0;">{count}</h2>
                <p style="margin: 0; color: rgba(255,255,255,0.8);">Documents Stored</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #3B82F6; margin: 0;">🔍</h3>
                <h2 style="margin: 0.5rem 0;">384</h2>
                <p style="margin: 0; color: rgba(255,255,255,0.8);">Embedding Dimensions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #10B981; margin: 0;">⚡</h3>
                <h2 style="margin: 0.5rem 0;">Cosine</h2>
                <p style="margin: 0; color: rgba(255,255,255,0.8);">Distance Metric</p>
            </div>
            """, unsafe_allow_html=True)
    except:
        pass

def main():
    # Title with beautiful styling
    st.markdown('<h1>🔍 Semantic Search RAG Application</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: rgba(255,255,255,0.8); margin-bottom: 2rem;">Powered by Google Gemini & ChromaDB Vector Database</p>', unsafe_allow_html=True)
    
    # Initialize components
    embedding_model = load_embedding_model()
    collection = init_vector_db()
    
    # Display metrics
    display_metrics(collection)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'txt'],
            help="Supported formats: PDF, DOCX, PPTX, TXT"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            # Show uploaded files
            for file in uploaded_files:
                st.markdown(f"""
                <div class="uploadedFile">
                    📄 {file.name}<br>
                    <small>Size: {file.size / 1024:.1f} KB</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 Process Documents", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                processed_chunks = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Extract text
                    text = extract_text_from_file(uploaded_file)
                    
                    if text.strip():
                        # Chunk text
                        chunks = chunk_text(text)
                        
                        if chunks:
                            # Generate embeddings
                            embeddings = embedding_model.encode(chunks)
                            
                            # Store in vector database
                            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                                try:
                                    collection.add(
                                        documents=[chunk],
                                        embeddings=[embedding.tolist()],
                                        ids=[f"{uploaded_file.name}_{int(time.time())}_{i}"],
                                        metadatas=[{
                                            "filename": uploaded_file.name,
                                            "chunk_id": i,
                                            "total_chunks": len(chunks),
                                            "file_type": uploaded_file.type
                                        }]
                                    )
                                    processed_chunks += 1
                                except Exception as e:
                                    st.warning(f"Error storing chunk from {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / total_files)
                
                status_text.text("✅ Processing complete!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"🎉 Successfully processed {processed_chunks} document chunks!")
                st.experimental_rerun()
        
        # Database management
        st.markdown("---")
        st.markdown("### 🗃️ Database Management")
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                try:
                    # Get all IDs and delete them
                    all_items = collection.get()
                    if all_items['ids']:
                        collection.delete(ids=all_items['ids'])
                    st.success("✅ Database cleared!")
                    st.session_state.confirm_clear = False
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm database deletion")
    
    # Main search interface
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    st.markdown("### 🔍 Ask Your Documents")
    
    # Search input with better styling
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search functionality
    if search_button and query:
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
        else:
            try:
                with st.spinner("🔍 Searching through your documents..."):
                    # Generate query embedding
                    query_embedding = embedding_model.encode([query])
                    
                    # Search for relevant documents
                    results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=5,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    if results['documents'][0]:
                        # Combine retrieved documents as context
                        context = "\n\n".join(results['documents'][0])
                        
                        # Generate RAG response
                        with st.spinner("🤖 Generating AI response..."):
                            response = generate_rag_response(query, context)
                        
                        # Display results in beautiful containers
                        st.markdown('<div class="response-container">', unsafe_allow_html=True)
                        st.markdown("### 🤖 AI Response")
                        st.markdown(response)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show source documents
                        with st.expander("📄 Source Documents", expanded=False):
                            for i, (doc, metadata, distance) in enumerate(zip(
                                results['documents'][0], 
                                results['metadatas'][0], 
                                results['distances'][0]
                            )):
                                similarity = 1 - distance
                                st.markdown(f"""
                                <div class="source-doc">
                                    <h4 style="color: #FF6B35; margin-top: 0;">📄 Source {i+1}</h4>
                                    <p><strong>File:</strong> {metadata.get('filename', 'Unknown')}</p>
                                    <p><strong>Similarity:</strong> {similarity:.1%}</p>
                                    <p><strong>Content:</strong></p>
                                    <p style="font-style: italic; opacity: 0.9;">{doc[:400]}{'...' if len(doc) > 400 else ''}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("🔍 No relevant documents found. Please upload some documents first or try a different search query.")
            
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
    
    # Tips and information
    if not query:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 💡 How to Use
            1. **Upload Documents** - Add PDF, DOCX, PPTX, or TXT files
            2. **Process Files** - Click "Process Documents" to analyze content
            3. **Search & Ask** - Enter questions about your documents
            4. **Get AI Answers** - Receive intelligent responses with sources
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Features
            - **Multi-format Support** - PDF, Word, PowerPoint, Text
            - **Semantic Search** - Find content by meaning, not just keywords
            - **AI-Powered Answers** - Get contextual responses from your data
            - **Source References** - See exactly where answers come from
            """)

if __name__ == "__main__":
    main()
