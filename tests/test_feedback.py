import os
os.environ["OPENAI_API_KEY"] = "mock-key"
os.environ["TAVILY_API_KEY"] = "mock-key"

from langchain_openai import OpenAIEmbeddings
OpenAIEmbeddings.embed_documents = lambda self, texts: [[0.1] * 1536] * len(texts)
OpenAIEmbeddings.embed_query = lambda self, text: [0.1] * 1536

import unittest
import sys
from unittest.mock import patch, AsyncMock

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage

from src.main import app
from src.memory.feedback_store import feedback_store
from src.models.feedback_request import FeedbackRequest


class TestFeedbackAndAnalytics(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        feedback_store.clear()

    def test_feedback_store_operations(self):
        req = FeedbackRequest(
            session_id="session_1",
            query="What is Adaptive RAG?",
            rating=5,
            feedback_text="Great answer!",
            route="index"
        )
        item = feedback_store.add_feedback(req)
        
        self.assertIsNotNone(item.feedback_id)
        self.assertEqual(item.session_id, "session_1")
        self.assertEqual(item.rating, 5)

        stats = feedback_store.get_stats()
        self.assertEqual(stats.total_feedback_count, 1)
        self.assertEqual(stats.average_rating, 5.0)
        self.assertIn("index", stats.route_breakdown)
        self.assertEqual(stats.route_breakdown["index"]["count"], 1)

    def test_submit_feedback_endpoint_valid(self):
        payload = {
            "session_id": "sess_123",
            "query": "How to deploy?",
            "rating": 4,
            "feedback_text": "Good explanation",
            "route": "general"
        }
        response = self.client.post("/rag/feedback", json=payload)
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["feedback"]["rating"], 4)

    def test_submit_feedback_endpoint_invalid_rating(self):
        payload = {
            "session_id": "sess_123",
            "query": "Invalid rating query",
            "rating": 6,
            "route": "general"
        }
        response = self.client.post("/rag/feedback", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_get_feedback_stats_endpoint(self):
        feedback_store.add_feedback(FeedbackRequest(
            session_id="s1", query="q1", rating=5, route="index"
        ))
        feedback_store.add_feedback(FeedbackRequest(
            session_id="s2", query="q2", rating=3, route="general"
        ))

        response = self.client.get("/rag/feedback/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_feedback_count"], 2)
        self.assertEqual(data["average_rating"], 4.0)

    @patch("src.api.routes.ChatHistory.get_session_history")
    def test_get_session_history_endpoint(self, mock_get_session_history):
        mock_chat_history = AsyncMock()
        mock_chat_history.get_messages.return_value = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        mock_get_session_history.return_value = mock_chat_history

        feedback_store.add_feedback(FeedbackRequest(
            session_id="test_history_sess", query="Hello", rating=5, route="general"
        ))

        response = self.client.get("/rag/history/test_history_sess")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], "test_history_sess")
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(len(data["feedback"]), 1)


if __name__ == "__main__":
    unittest.main()
