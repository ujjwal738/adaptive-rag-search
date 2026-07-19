"""
Graph builder module for the adaptive RAG system.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.openai import llm
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool

config = Config()


def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    question = state["messages"][-1].content
    retriever = get_retriever()
    context = retriever.invoke(question)
    print("docs received from FAISS")
    print(context)

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("result received is in query classifier")
    print(result.route)

    return {"messages": state["messages"], "route": result.route, "latest_query": question}


# Placeholder/Mock nodes for features to be implemented in subsequent phases
def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    print("inside general llm")
    print(result)
    return {"messages": result}


def retriever_node(state: State):
    """Placeholder for retriever node."""
    print("Placeholder: retriever_node node")
    return {"messages": state["messages"] + [AIMessage(content="Retriever node placeholder response")]}


def grade(state: State):
    """Placeholder for grading node."""
    print("Placeholder: grade node")
    return {"messages": state["messages"], "binary_score": "yes"}


def rewrite_query(state: State):
    """Placeholder for query rewriting node."""
    print("Placeholder: rewrite_query node")
    return {"latest_query": state["latest_query"]}


def generate(state: State):
    """Placeholder for generation node."""
    print("Placeholder: generate node")
    return {"messages": state["messages"] + [AIMessage(content="Generate node placeholder response")]}


def web_search(state: State):
    """Placeholder for web search node."""
    print("Placeholder: web_search node")
    return {"messages": state["messages"] + [AIMessage(content="Web search node placeholder response")]}


# Build the graph structure
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()
