---
title: NHS Clinical Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
- healthcare
- nhs
- rag
- llm
- medical
pinned: false
short_description: RAG-powered NHS health information chatbot
---

# NHS Clinical Assistant

A RAG-based chatbot for querying NHS health condition information. This application uses Retrieval-Augmented Generation to provide accurate, evidence-based responses from official NHS health documentation.

## 🌟 Features

- **NHS Health Information Search**: Search through NHS health conditions using semantic search powered by Voyage AI embeddings
- **RAG-powered Chat**: Ask questions and get contextually relevant answers from NHS health information with source citations
- **Multiple LLM Support**: Choose between Gemini models (2.5-flash, 2.5-flash-lite, 2.5-pro) for generating responses
- **Source Attribution**: All responses include links to original NHS web pages
- **Streaming Responses**: Real-time response generation for better user experience
- **Interactive Interface**: Clean Streamlit frontend optimized for healthcare information queries

## 📁 Project Structure

### Core Application Files

#### [`src/streamlit_app.py`](src/streamlit_app.py)
Main Streamlit application interface providing:
- User-friendly web interface for NHS health information queries
- Chat interface with conversation history
- Model selection (Gemini variants)
- Source attribution display with NHS links
- Suggested queries for common health topics

#### [`src/query_rag.py`](src/query_rag.py)
RAG (Retrieval-Augmented Generation) system that handles:
- Query processing and validation
- Integration with search engine and LLM clients
- Context generation from NHS health documents
- Streaming response generation
- Source extraction and formatting
- Can be used as standalone CLI tool for testing

#### [`src/search_engine.py`](src/search_engine.py)
Search functionality using Pinecone vector database:
- Similarity search using Voyage AI embeddings (voyage-context-3 model)
- Integration with Pinecone vector database
- NHS health information retrieval

### Configuration

#### [`src/config.py`](src/config.py)
Centralized configuration management:
- NHS source configuration
- System prompts and error messages
- Default search parameters

### Infrastructure

#### [`requirements.txt`](requirements.txt)
Python dependencies:
- `streamlit==1.40.1` - Web application framework
- `openai` - LLM client (used for Gemini API access)
- `voyageai` - Embedding generation
- `pinecone` - Vector database client
- `pandas` - Data manipulation
- `altair` - Visualization support

#### [`Dockerfile`](Dockerfile)
Container configuration for deployment:
- Python 3.9 base image
- Production-ready setup
- Health check configuration
- Streamlit server configuration

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Gemini API key (for LLM responses)
- Voyage AI API key (for embeddings)
- Pinecone API key (for vector search)

### Environment Variables
Set the following environment variables:
```bash
export GEMINI_API_KEY=your_gemini_api_key
export VOYAGE_API_KEY=your_voyage_api_key
export PINECONE_API_KEY=your_pinecone_api_key
```

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application
```bash
streamlit run src/streamlit_app.py
```

The application will be available at `http://localhost:8501`

### Docker Deployment
```bash
docker build -t nhs-clinical-assistant .
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e VOYAGE_API_KEY=your_voyage_api_key \
  -e PINECONE_API_KEY=your_pinecone_api_key \
  nhs-clinical-assistant
```

## 🔧 Usage

### Web Interface
1. Open the application in your browser
2. Select your preferred Gemini model from the sidebar
3. Type your NHS health-related question in the chat input
4. View the response with source attribution
5. Click "View Sources" to see NHS page references

### CLI Usage
Test the RAG system directly:
```bash
python src/query_rag.py --query_text "What are the symptoms of ADHD in adults?" --llm_model "gemini-2.5-flash"
```

### Example Queries
- "What are the symptoms of ADHD in adults?"
- "How is type 2 diabetes diagnosed?"
- "What are the treatment options for depression?"

## 🏗️ Architecture

The system uses a simple but effective RAG architecture:

1. **Query Processing**: User query is validated and processed
2. **Vector Search**: Query is embedded using Voyage AI and searched against Pinecone vector database containing NHS health information
3. **Context Generation**: Retrieved NHS documents are formatted into context
4. **LLM Response**: Gemini generates response based strictly on NHS context
5. **Source Attribution**: Original NHS page links are provided with responses

## 📊 Data Sources

The system is built on NHS health condition information, stored in a Pinecone vector database with the namespace `nhs_guidelines_voyage_3_large`. All responses include proper attribution to NHS sources with direct links to official NHS web pages.

## ⚠️ Important Notes

- **Medical Disclaimer**: This tool provides information from NHS sources but should not replace professional medical advice
- **Data Accuracy**: Always consult official NHS sources for the most current information
- **Context Limitation**: The system only responds based on information available in the indexed NHS documents

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Code License
The source code of this application is released under AGPL-3.0, which means:
- You can freely use, modify, and distribute this software
- Any modifications or derivative works must also be released under AGPL-3.0
- If you run this software as a network service, you must provide the source code to users
- See the [LICENSE](LICENSE) file for full terms

### NHS Data Usage
This tool utilizes NHS health information under the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). All NHS content remains subject to their original terms and conditions and is used for informational purposes in compliance with UK public sector information licensing.

**Note**: While the application code is AGPL-3.0 licensed, the NHS health information content accessed through this application remains under Crown Copyright and the Open Government Licence.