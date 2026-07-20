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

## Software Development Phases

### Phase 1: Vector Store Setup & Document Uploading (`feat/1-vector-db-setup`)
- Initialized local Python virtual environment and installed core dependencies (LangChain, LangGraph, Streamlit, FAISS, OpenAI).
- Created an OpenAI chat LLM integration client in `src/llms/openai.py`.
- Developed description enhancement module in `src/tools/common_tools.py` using LLM prompts.
- Configured local FAISS vector database store and helper retriever tool builder in `src/rag/retriever_setup.py`.
- Replaced the mock document uploading pipeline in `src/rag/document_upload.py` to perform text/PDF loading, description enhancement, chunking, and FAISS vector storage.

### Phase 2: Query Routing Node (`feat/2-query-routing`)
- Created Pydantic models for structured output: `RouteIdentifier` (`src/models/route_identifier.py`) and `VerificationResult` (`src/models/verification_result.py`).
- Configured LangGraph conditional routing paths in `src/tools/graph_tools.py` (e.g. `routing_tool`, `doc_tool`, and `verify_answer`).
- Replaced the mock graph builder in `src/rag/graph_builder.py` with a completed `StateGraph` containing a fully functional `query_classifier` node to analyze context and route queries to `index`, `general`, or `search`.

### Phase 3: General Chit-Chat Response Generator (`feat/3-general-chit-chat`)
- Implemented the `general_llm` node inside `src/rag/graph_builder.py` to handle general chit-chat and common knowledge questions by directly invoking the OpenAI Chat LLM, avoiding database lookups for generic queries.

### Phase 4: Document Retrieval Node (`feat/4-document-retrieval`)
- Configured a local ReAct agent and executor inside `src/rag/reAct_agent.py` to coordinate tools for document retrieval.
- Implemented the `retriever_node` in `src/rag/graph_builder.py` which executes the ReAct executor to find information within the vector store, package intermediate tool calls, and append retrieved results as messages.
