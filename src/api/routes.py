"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history_mongo import ChatHistory
from src.memory.feedback_store import feedback_store
from src.models.query_request import QueryRequest
from src.models.feedback_request import FeedbackRequest, FeedbackStatsResponse
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result with route information.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    result = builder.invoke({
        "messages": messages
    })
    output_text = result["messages"][-1].content
    route = result.get("route", "general")

    # Save assistant message
    await chat_history.add_message(AIMessage(content=output_text))

    return {
        "result": result["messages"][-1],
        "route": route,
        "session_id": req.session_id
    }


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    status_upload = documents(description, file)
    return {"status": status_upload}


@router.post("/rag/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Submit rating and feedback for a RAG response.

    Args:
        req: FeedbackRequest containing session_id, rating, query, and optional text.

    Returns:
        Confirmation with stored feedback item.
    """
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    item = feedback_store.add_feedback(req)
    return {"status": "success", "feedback": item}


@router.get("/rag/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats():
    """
    Get aggregate feedback metrics and distribution per routing path.
    """
    return feedback_store.get_stats()


@router.get("/rag/history/{session_id}")
async def get_session_history(session_id: str):
    """
    Fetch session message history and associated feedback.

    Args:
        session_id: Session identifier.

    Returns:
        List of message objects and recorded session feedback.
    """
    chat_history = ChatHistory.get_session_history(session_id)
    messages = await chat_history.get_messages()
    formatted_messages = [
        {
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content
        }
        for msg in messages
    ]
    feedback_items = feedback_store.get_session_feedback(session_id)

    return {
        "session_id": session_id,
        "messages": formatted_messages,
        "feedback": feedback_items
    }

