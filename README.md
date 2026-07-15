# Adaptive RAG Search

Adaptive RAG Search is an intelligent Retrieval-Augmented Generation (RAG) system built with FastAPI, LangGraph, and Streamlit.

## Key Features
- **Intelligent Query Routing**: Automatic classification of queries.
- **Advanced RAG Pipeline**: High performance vector search and relevance grading.
- **FastAPI Backend**: High performance async backend server.
- **Streamlit UI**: Simple, clean chat interface.

## Quick Start
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and fill in API keys.
5. Run the FastAPI backend: `uvicorn src.main:app --reload`.
6. Run the Streamlit app: `streamlit run streamlit_app/home.py`.
