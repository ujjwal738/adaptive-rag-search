import os
os.environ["OPENAI_API_KEY"] = "mock-key"
os.environ["TAVILY_API_KEY"] = "mock-key"

# Mock OpenAIEmbeddings methods globally before other imports to prevent OpenAI API calls for FAISS
from langchain_openai import OpenAIEmbeddings
OpenAIEmbeddings.embed_documents = lambda self, texts: [[0.1] * 1536] * len(texts)
OpenAIEmbeddings.embed_query = lambda self, text: [0.1] * 1536

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app
from langchain_core.messages import AIMessage, HumanMessage


class TestFastAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Adaptive RAG API is running"})

    @patch("src.api.routes.ChatHistory.get_session_history")
    @patch("src.api.routes.builder")
    def test_rag_query_endpoint(self, mock_builder, mock_get_session_history):
        # Setup mock chat history
        mock_chat_history = AsyncMock()
        mock_chat_history.get_messages.return_value = [HumanMessage(content="test query")]
        mock_get_session_history.return_value = mock_chat_history

        # Setup mock graph builder invoke output
        mock_builder.invoke.return_value = {
            "messages": [
                HumanMessage(content="test query"),
                AIMessage(content="mocked final response")
            ]
        }

        payload = {"query": "test query", "session_id": "test_session_123"}
        response = self.client.post("/rag/query", json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.json())
        self.assertEqual(response.json()["result"]["content"], "mocked final response")
        
        self.assertEqual(mock_chat_history.add_message.call_count, 2)
        mock_builder.invoke.assert_called_once()

    @patch("src.api.routes.documents")
    def test_upload_file_endpoint(self, mock_documents):
        mock_documents.return_value = True
        
        # Create a mock file
        file_content = b"This is some dummy document content to test upload."
        files = {"file": ("test.txt", file_content, "text/plain")}
        headers = {"X-Description": "Test file description"}
        
        response = self.client.post("/rag/documents/upload", files=files, headers=headers)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": True})
        mock_documents.assert_called_once()


if __name__ == "__main__":
    unittest.main()
