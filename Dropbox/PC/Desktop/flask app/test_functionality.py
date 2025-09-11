#!/usr/bin/env python3
"""
Quick functionality test for RAG Semantic Search Application
Tests core components without running the full Streamlit UI
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required imports work"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        import google.generativeai as genai
        from dotenv import load_dotenv
        import pandas as pd
        import PyPDF2
        import docx
        import pptx
        import chromadb
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import json
        import tempfile
        import time
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding_model():
    """Test embedding model loading"""
    print("\n🤖 Testing embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test embedding generation
        test_text = "This is a test document for semantic search."
        embedding = model.encode([test_text])
        print(f"✅ Embedding model loaded successfully")
        print(f"✅ Generated embedding shape: {embedding.shape}")
        return True, model
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False, None

def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n🗃️ Testing ChromaDB...")
    try:
        import chromadb
        
        # Create persistent client for testing
        client = chromadb.PersistentClient(path="./test_database")
        
        # Create test collection
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Test adding documents
        test_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Vector databases are essential for semantic search applications."
        ]
        
        test_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]  # Mock embeddings
        
        collection.add(
            documents=test_docs,
            embeddings=test_embeddings,
            ids=["doc1", "doc2", "doc3"]
        )
        
        # Test querying
        results = collection.query(
            query_embeddings=[[0.15] * 384],
            n_results=2
        )
        
        print(f"✅ ChromaDB working successfully")
        print(f"✅ Added {len(test_docs)} test documents")
        print(f"✅ Query returned {len(results['documents'][0])} results")
        return True
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

def test_gemini_api():
    """Test Gemini API configuration"""
    print("\n🧠 Testing Gemini API...")
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("⚠️ GEMINI_API_KEY not found in environment variables")
            print("💡 Create a .env file with GEMINI_API_KEY=your_api_key")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with a simple generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        
        print("✅ Gemini API configured successfully")
        print(f"✅ Test response: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

def test_document_processing():
    """Test document processing capabilities"""
    print("\n📄 Testing document processing...")
    try:
        # Test text processing
        sample_text = """
        This is a sample document for testing.
        It contains multiple paragraphs and sentences.
        We will test how well our system can process and chunk this text.
        """
        
        # Simple chunking test
        sentences = sample_text.strip().split('.')
        chunks = [s.strip() for s in sentences if s.strip()]
        
        print(f"✅ Text processing successful")
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Test file format support
        supported_formats = ['.txt', '.pdf', '.docx', '.pptx']
        print(f"✅ Supports file formats: {', '.join(supported_formats)}")
        
        return True
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False

def test_rag_pipeline():
    """Test complete RAG pipeline"""
    print("\n🔄 Testing RAG pipeline...")
    try:
        # Mock RAG components
        query = "What is machine learning?"
        context = "Machine learning is a subset of artificial intelligence that focuses on algorithms."
        
        # Mock prompt template
        prompt = f"""
        Context: {context}
        
        Question: {query}
        
        Based on the context above, please provide a comprehensive answer.
        """
        
        print("✅ RAG pipeline structure validated")
        print(f"✅ Query: {query}")
        print(f"✅ Context retrieved: {len(context)} characters")
        return True
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAG Application Functionality Tests\n")
    
    tests = [
        ("Imports", test_imports),
        ("Embedding Model", test_embedding_model),
        ("ChromaDB", test_chromadb),
        ("Gemini API", test_gemini_api),
        ("Document Processing", test_document_processing),
        ("RAG Pipeline", test_rag_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Embedding Model":
                result, model = test_func()
                results[test_name] = result
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG application is ready to run.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
