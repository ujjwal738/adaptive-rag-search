"""
Models for RAG query feedback and analytics.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request model for submitting user feedback on a RAG answer."""
    session_id: str = Field(..., description="Unique session ID for the chat")
    query: str = Field(..., description="The user query associated with the response")
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Optional text feedback or comments")
    route: Optional[str] = Field("general", description="The RAG routing path taken (e.g. index, general, search)")


class FeedbackItem(BaseModel):
    """Stored feedback entry with timestamp."""
    feedback_id: str
    session_id: str
    query: str
    rating: int
    feedback_text: Optional[str] = None
    route: str
    timestamp: str


class FeedbackStatsResponse(BaseModel):
    """Response model for aggregate feedback statistics."""
    total_feedback_count: int
    average_rating: float
    rating_distribution: Dict[int, int]
    route_breakdown: Dict[str, Dict[str, Any]]
