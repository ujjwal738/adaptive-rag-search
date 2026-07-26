"""
Thread-safe memory store for RAG feedback ratings and aggregate analytics.
"""

from datetime import datetime
import threading
import uuid
from typing import List, Dict, Any

from src.models.feedback_request import FeedbackRequest, FeedbackItem, FeedbackStatsResponse


class FeedbackStore:
    """
    In-memory thread-safe store for user ratings and feedback analytics.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FeedbackStore, cls).__new__(cls)
                cls._instance._feedback_records: List[FeedbackItem] = []
                cls._instance._store_lock = threading.Lock()
            return cls._instance

    def add_feedback(self, req: FeedbackRequest) -> FeedbackItem:
        """
        Record a new feedback item.
        """
        item = FeedbackItem(
            feedback_id=str(uuid.uuid4()),
            session_id=req.session_id,
            query=req.query,
            rating=req.rating,
            feedback_text=req.feedback_text,
            route=req.route or "general",
            timestamp=datetime.utcnow().isoformat()
        )
        with self._store_lock:
            self._feedback_records.append(item)
        return item

    def get_session_feedback(self, session_id: str) -> List[FeedbackItem]:
        """
        Get all feedback recorded for a given session.
        """
        with self._store_lock:
            return [fb for fb in self._feedback_records if fb.session_id == session_id]

    def get_all_feedback(self) -> List[FeedbackItem]:
        """
        Get all feedback recorded in the system.
        """
        with self._store_lock:
            return list(self._feedback_records)

    def get_stats(self) -> FeedbackStatsResponse:
        """
        Calculate aggregate feedback metrics and rating distribution per routing path.
        """
        with self._store_lock:
            records = list(self._feedback_records)

        total = len(records)
        if total == 0:
            return FeedbackStatsResponse(
                total_feedback_count=0,
                average_rating=0.0,
                rating_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                route_breakdown={}
            )

        avg_rating = round(sum(item.rating for item in records) / total, 2)
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        routes: Dict[str, List[int]] = {}

        for item in records:
            dist[item.rating] = dist.get(item.rating, 0) + 1
            if item.route not in routes:
                routes[item.route] = []
            routes[item.route].append(item.rating)

        route_breakdown: Dict[str, Dict[str, Any]] = {}
        for r_name, r_ratings in routes.items():
            route_breakdown[r_name] = {
                "count": len(r_ratings),
                "average_rating": round(sum(r_ratings) / len(r_ratings), 2)
            }

        return FeedbackStatsResponse(
            total_feedback_count=total,
            average_rating=avg_rating,
            rating_distribution=dist,
            route_breakdown=route_breakdown
        )

    def clear(self):
        """Clear all stored feedback records (primarily for testing)."""
        with self._store_lock:
            self._feedback_records.clear()


# Global singleton instance
feedback_store = FeedbackStore()
