# AI-Powered Resume Analysis System
https://ganesh-gaiy-resume-rag.streamlit.app/
## Overview
This intelligent resume analysis system combines advanced AI and search technologies to provide quick, accurate insights from resume collections. Built with state-of-the-art language models and hybrid search capabilities, it enables HR professionals and recruiters to efficiently analyze multiple resumes through natural language queries.

## Key Features
- **Natural Language Querying**: Ask questions about resumes in plain English
- **Intelligent Search**: Combines semantic and keyword-based search for optimal results
- **Real-time Analysis**: Get instant answers about candidates' skills, experience, and qualifications
- **Interactive Web Interface**: User-friendly Streamlit application for easy access
- **High-Performance Backend**: Utilizes Groq's LLM for fast response generation

## Technical Architecture
The system is built on a modern tech stack:
- **Frontend**: Streamlit web application
- **Search Engine**: Pinecone vector database with hybrid search capability
- **Language Models**: 
  - Groq LLM (Llama3-8b-8192) for answer generation
  - OpenAI embeddings for semantic search
- **Search Technologies**:
  - Dense vectors (OpenAI embeddings) for semantic understanding
  - Sparse vectors (BM25) for keyword matching
  - Hybrid search combining both approaches for optimal results

## Getting Started

### Prerequisites
- Python 3.8+
- Pinecone API key
- Groq API key
- OpenAI API key

### Environment Setup
1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file with your API keys:
```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

### Running the Application
1. Index your resumes:
```bash
python index_resumes.py
```
2. Start the web application:
```bash
streamlit run app.py
```

## Usage
1. Upload PDF resumes to the designated directory
2. Launch the web application
3. Enter natural language queries about the resumes
4. Receive AI-generated answers based on the resume content

Example queries:
- "What candidates have experience with Python and machine learning?"
- "Who has worked at Fortune 500 companies?"
- "List candidates with more than 5 years of management experience"

## Technical Details

### Indexing Process
1. PDF documents are loaded and split into manageable chunks
2. Both dense (semantic) and sparse (keyword) embeddings are generated
3. Embeddings are stored in Pinecone's vector database
4. Hybrid search combines both embedding types for optimal retrieval

### Query Process
1. User query is processed for both semantic and keyword search
2. Relevant resume sections are retrieved using hybrid search
3. Retrieved content is analyzed by Groq's LLM
4. Structured response is generated and displayed to the user

## Performance and Scalability
- Serverless architecture for automatic scaling
- Optimized for quick response times
- Efficient document processing with batched operations
- Hybrid search ensures both accuracy and speed

## Security Considerations
- API keys stored securely in environment variables
- Document access controlled through application logic
- Secure cloud infrastructure for data storage

## Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

