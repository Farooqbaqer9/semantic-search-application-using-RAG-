"""
Synthetic Test Data for RAG Application
This document contains sample content to test the semantic search functionality.
"""

# Document 1: Artificial Intelligence Overview
AI_OVERVIEW = """
Artificial Intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.

Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.

The applications of AI are vast and growing, including healthcare diagnostics, autonomous vehicles, recommendation systems, fraud detection, and virtual assistants.
"""

# Document 2: Natural Language Processing
NLP_OVERVIEW = """
Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a valuable way.

Key NLP tasks include:
- Text classification and sentiment analysis
- Named entity recognition
- Machine translation
- Question answering systems
- Text summarization
- Language generation

Modern NLP relies heavily on transformer architectures, which have revolutionized the field. Models like BERT, GPT, and T5 have achieved remarkable performance on various language understanding tasks.

Vector embeddings play a crucial role in NLP, converting text into numerical representations that capture semantic meaning. This enables semantic search, where queries can find relevant documents based on meaning rather than just keyword matching.
"""

# Document 3: Vector Databases and Semantic Search
VECTOR_DB_OVERVIEW = """
Vector databases are specialized databases designed to store, index, and query high-dimensional vector data efficiently. They are essential for applications involving machine learning, AI, and semantic search.

Unlike traditional databases that store structured data in tables, vector databases store embeddings - numerical representations of data points in high-dimensional space. These embeddings capture semantic relationships between data items.

Key features of vector databases include:
- Efficient similarity search using algorithms like HNSW or IVF
- Support for various distance metrics (cosine, euclidean, dot product)
- Scalability to handle millions or billions of vectors
- Real-time query performance
- Integration with machine learning pipelines

Popular vector databases include Pinecone, Weaviate, Qdrant, and ChromaDB. They enable applications like recommendation systems, image search, document retrieval, and retrieval-augmented generation (RAG).

In RAG systems, vector databases store document embeddings, allowing for semantic retrieval of relevant context to augment language model responses.
"""

# Document 4: Machine Learning Fundamentals
ML_FUNDAMENTALS = """
Machine Learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.

There are three main types of machine learning:

1. Supervised Learning: Uses labeled training data to learn a mapping function from inputs to outputs. Examples include classification and regression tasks.

2. Unsupervised Learning: Finds hidden patterns or structures in unlabeled data. Examples include clustering, dimensionality reduction, and association rule learning.

3. Reinforcement Learning: Learns through interaction with an environment, receiving rewards or penalties for actions taken.

Key concepts in machine learning include:
- Training and test datasets
- Feature engineering and selection
- Model evaluation metrics
- Overfitting and underfitting
- Cross-validation
- Hyperparameter tuning

Popular algorithms include linear regression, decision trees, random forests, support vector machines, neural networks, and ensemble methods.
"""

# Document 5: Data Science and Analytics
DATA_SCIENCE_OVERVIEW = """
Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

The data science process typically involves:
1. Data collection and acquisition
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model building and selection
6. Model evaluation and validation
7. Deployment and monitoring

Data scientists use a variety of tools and technologies including:
- Programming languages: Python, R, SQL
- Libraries: pandas, NumPy, scikit-learn, TensorFlow, PyTorch
- Visualization tools: Matplotlib, Seaborn, Plotly, Tableau
- Big data technologies: Spark, Hadoop, Kafka
- Cloud platforms: AWS, Google Cloud, Azure

Applications of data science span across industries including finance, healthcare, retail, manufacturing, and technology. It enables organizations to make data-driven decisions, optimize operations, and gain competitive advantages.
"""

# Test queries for evaluation
TEST_QUERIES = [
    "What is machine learning?",
    "How do vector databases work?",
    "What are the types of machine learning?",
    "Explain natural language processing",
    "What tools do data scientists use?",
    "How does semantic search work?",
    "What are transformer architectures?",
    "Describe the data science process",
    "What is deep learning?",
    "How are embeddings used in NLP?"
]

# Expected relevant documents for each query (for evaluation)
QUERY_RELEVANCE = {
    "What is machine learning?": ["AI_OVERVIEW", "ML_FUNDAMENTALS"],
    "How do vector databases work?": ["VECTOR_DB_OVERVIEW"],
    "What are the types of machine learning?": ["ML_FUNDAMENTALS"],
    "Explain natural language processing": ["NLP_OVERVIEW"],
    "What tools do data scientists use?": ["DATA_SCIENCE_OVERVIEW"],
    "How does semantic search work?": ["VECTOR_DB_OVERVIEW", "NLP_OVERVIEW"],
    "What are transformer architectures?": ["NLP_OVERVIEW"],
    "Describe the data science process": ["DATA_SCIENCE_OVERVIEW"],
    "What is deep learning?": ["AI_OVERVIEW"],
    "How are embeddings used in NLP?": ["NLP_OVERVIEW", "VECTOR_DB_OVERVIEW"]
}

# All documents
ALL_DOCUMENTS = {
    "AI_OVERVIEW": AI_OVERVIEW,
    "NLP_OVERVIEW": NLP_OVERVIEW,
    "VECTOR_DB_OVERVIEW": VECTOR_DB_OVERVIEW,
    "ML_FUNDAMENTALS": ML_FUNDAMENTALS,
    "DATA_SCIENCE_OVERVIEW": DATA_SCIENCE_OVERVIEW
}
