"""
Graph builder module placeholder.
"""

from langchain_core.messages import AIMessage


class MockBuilder:
    """Mock LangGraph builder object."""

    def invoke(self, state: dict) -> dict:
        """
        Mock invoke method for processing queries.
        """
        messages = state.get("messages", [])
        last_content = messages[-1].content if messages else "No query received"
        return {
            "messages": messages + [AIMessage(content=f"Mock response to: '{last_content}'")]
        }


builder = MockBuilder()
