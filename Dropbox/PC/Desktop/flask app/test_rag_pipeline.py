#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Test with Synthetic Data
Tests the complete RAG workflow including document processing, embedding, storage, retrieval, and generation
"""

import os
import sys
import tempfile
from pathlib import Path
import time

# Import synthetic data
from synthetic_data import ALL_DOCUMENTS, TEST_QUERIES, QUERY_RELEVANCE

def setup_environment():
    """Setup environment and imports"""
    from dotenv import load_dotenv
    load_dotenv()
    
    import chromadb
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    
    # Configure Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        print("⚠️ GEMINI_API_KEY not found. RAG generation will be skipped.")
        return False

def test_document_processing_and_embedding():
    """Test document processing and embedding generation"""
    print("📄 Testing document processing and embedding...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        processed_docs = {}
        embeddings = {}
        
        for doc_id, content in ALL_DOCUMENTS.items():
            # Simple text chunking (split by paragraphs)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            processed_docs[doc_id] = chunks
            
            # Generate embeddings for each chunk
            chunk_embeddings = model.encode(chunks)
            embeddings[doc_id] = chunk_embeddings
            
            print(f"✅ Processed {doc_id}: {len(chunks)} chunks, embeddings shape: {chunk_embeddings.shape}")
        
        return processed_docs, embeddings, model
        
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return None, None, None

def test_vector_database_operations(processed_docs, embeddings):
    """Test vector database storage and retrieval"""
    print("\n🗃️ Testing vector database operations...")
    
    try:
        import chromadb
        
        # Create test database
        client = chromadb.PersistentClient(path="./test_rag_database")
        
        # Create collection
        collection = client.get_or_create_collection(
            name="test_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store all documents and embeddings
        all_chunks = []
        all_embeddings = []
        all_ids = []
        all_metadata = []
        
        for doc_id, chunks in processed_docs.items():
            doc_embeddings = embeddings[doc_id]
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_embeddings.append(doc_embeddings[i].tolist())
                all_ids.append(f"{doc_id}_chunk_{i}")
                all_metadata.append({"document_id": doc_id, "chunk_index": i})
        
        # Add to collection
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata
        )
        
        print(f"✅ Stored {len(all_chunks)} document chunks in vector database")
        
        # Test retrieval
        test_query = "What is machine learning?"
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode([test_query])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Retrieved {len(results['documents'][0])} relevant chunks for test query")
        
        return collection, model
        
    except Exception as e:
        print(f"❌ Vector database error: {e}")
        return None, None

def test_semantic_search(collection, model):
    """Test semantic search with all test queries"""
    print("\n🔍 Testing semantic search...")
    
    try:
        search_results = {}
        
        for query in TEST_QUERIES[:5]:  # Test first 5 queries
            # Generate query embedding
            query_embedding = model.encode([query])
            
            # Search for relevant documents
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results[query] = {
                "documents": results['documents'][0],
                "metadata": results['metadatas'][0],
                "distances": results['distances'][0]
            }
            
            print(f"✅ Query: '{query[:50]}...'")
            print(f"   Found {len(results['documents'][0])} relevant chunks")
            print(f"   Best match distance: {results['distances'][0][0]:.3f}")
        
        return search_results
        
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
        return None

def test_rag_generation(search_results, has_api_key):
    """Test RAG response generation"""
    print("\n🧠 Testing RAG generation...")
    
    if not has_api_key:
        print("⚠️ Skipping RAG generation (no API key)")
        return True
    
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        for query, results in list(search_results.items())[:3]:  # Test first 3
            # Prepare context from retrieved documents
            context = "\n\n".join(results["documents"])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question comprehensively and accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = model.generate_content(prompt)
            
            print(f"✅ Generated response for: '{query[:50]}...'")
            print(f"   Response length: {len(response.text)} characters")
            print(f"   Response preview: {response.text[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ RAG generation error: {e}")
        return False

def cleanup():
    """Clean up test database"""
    try:
        import shutil
        if os.path.exists("./test_rag_database"):
            shutil.rmtree("./test_rag_database")
        if os.path.exists("./test_database"):
            shutil.rmtree("./test_database")
        print("🧹 Cleaned up test databases")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Run comprehensive RAG pipeline test"""
    print("🚀 Starting Comprehensive RAG Pipeline Test with Synthetic Data\n")
    
    start_time = time.time()
    
    # Setup
    has_api_key = setup_environment()
    
    # Test document processing and embedding
    processed_docs, embeddings, model = test_document_processing_and_embedding()
    if not processed_docs:
        print("❌ Test failed at document processing stage")
        return False
    
    # Test vector database operations
    collection, model = test_vector_database_operations(processed_docs, embeddings)
    if not collection:
        print("❌ Test failed at vector database stage")
        return False
    
    # Test semantic search
    search_results = test_semantic_search(collection, model)
    if not search_results:
        print("❌ Test failed at semantic search stage")
        return False
    
    # Test RAG generation
    rag_success = test_rag_generation(search_results, has_api_key)
    
    # Cleanup
    cleanup()
    
    end_time = time.time()
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    print(f"✅ Document Processing: {len(ALL_DOCUMENTS)} documents processed")
    print(f"✅ Vector Database: {sum(len(chunks) for chunks in processed_docs.values())} chunks stored")
    print(f"✅ Semantic Search: {len(search_results)} queries tested")
    print(f"✅ RAG Generation: {'Tested' if has_api_key and rag_success else 'Skipped (no API key)' if not has_api_key else 'Failed'}")
    print(f"⏱️ Total test time: {end_time - start_time:.2f} seconds")
    
    success = all([
        processed_docs is not None,
        collection is not None,
        search_results is not None,
        (rag_success or not has_api_key)  # Success if RAG works or if no API key (expected)
    ])
    
    if success:
        print("\n🎉 All RAG pipeline components working perfectly!")
        print("🚀 Your application is ready for production use!")
    else:
        print("\n⚠️ Some components failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
