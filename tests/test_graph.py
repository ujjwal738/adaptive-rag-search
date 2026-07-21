import os
os.environ["OPENAI_API_KEY"] = "mock-key"
os.environ["TAVILY_API_KEY"] = "mock-key"

# Mock OpenAIEmbeddings methods globally before other imports to prevent OpenAI API calls for FAISS
from langchain_openai import OpenAIEmbeddings
OpenAIEmbeddings.embed_documents = lambda self, texts: [[0.1] * 1536] * len(texts)
OpenAIEmbeddings.embed_query = lambda self, text: [0.1] * 1536

import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage
from src.models.route_identifier import RouteIdentifier
from src.models.grade_result import GradeResult


class TestRAGGraphNodes(unittest.TestCase):

    @patch('src.rag.graph_builder.llm')
    @patch('src.rag.graph_builder.get_retriever')
    def test_query_classifier(self, mock_get_retriever, mock_llm):
        from src.rag.graph_builder import query_classifier
        
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = "Mock context"
        mock_get_retriever.return_value = mock_retriever
        
        # Mock LLM structured output
        mock_llm_structured = MagicMock()
        mock_llm_structured.invoke.return_value = RouteIdentifier(route="index")
        mock_llm.with_structured_output.return_value = mock_llm_structured
        
        state = {"messages": [HumanMessage(content="Hello")]}
        result = query_classifier(state)
        
        self.assertEqual(result["route"], "index")
        self.assertEqual(result["latest_query"], "Hello")

    @patch('src.rag.graph_builder.llm')
    def test_general_llm(self, mock_llm):
        from src.rag.graph_builder import general_llm
        
        mock_llm.invoke.return_value = AIMessage(content="General response")
        
        state = {"messages": [HumanMessage(content="What is 2+2?")]}
        result = general_llm(state)
        
        self.assertEqual(result["messages"].content, "General response")

    @patch('src.rag.graph_builder.llm')
    def test_grade(self, mock_llm):
        from src.rag.graph_builder import grade
        
        mock_llm_structured = MagicMock()
        mock_llm_structured.invoke.return_value = GradeResult(binary_score="yes")
        mock_llm.with_structured_output.return_value = mock_llm_structured
        
        state = {
            "latest_query": "Test query",
            "messages": [AIMessage(content="Retrieved context content")]
        }
        result = grade(state)
        self.assertEqual(result["binary_score"], "yes")

    @patch('src.rag.graph_builder.llm')
    def test_rewrite_query(self, mock_llm):
        from src.rag.graph_builder import rewrite_query
        
        mock_llm.invoke.return_value = AIMessage(content="rewritten query")
        
        state = {"latest_query": "original query"}
        result = rewrite_query(state)
        self.assertEqual(result["latest_query"], "rewritten query")

    @patch('src.rag.graph_builder.llm')
    def test_generate(self, mock_llm):
        from src.rag.graph_builder import generate
        
        mock_llm.invoke.return_value = AIMessage(content="Generated answer")
        
        state = {"messages": [AIMessage(content="Retrieved context")]}
        result = generate(state)
        self.assertEqual(result["messages"][0].content, "Generated answer")


if __name__ == '__main__':
    unittest.main()
