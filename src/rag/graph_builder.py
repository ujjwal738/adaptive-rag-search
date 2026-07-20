"""
Graph builder module for the adaptive RAG system.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.retriever_setup import get_retriever
from src.rag.reAct_agent import agent_executor
from src.config.settings import Config
from src.llms.openai import llm
from src.models.route_identifier import RouteIdentifier
from src.models.grade_result import GradeResult
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
    """
    Retrieve results from vector stores using the reAct agent.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with tool calls.
    """
    messages = state["latest_query"]
    result = agent_executor.invoke({"input": messages})

    # Extract tool calls
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = []
    if intermediate_steps:
        for action, tool_result in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
            })

    new_message = AIMessage(
        content=result["output"],
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message]
    }


def grade(state: State):
    """
    Grade the retrieved documents for relevance to the user question.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated binary_score in the state.
    """
    question = state["latest_query"]
    context = state["messages"][-1].content

    llm_with_structured_output = llm.with_structured_output(GradeResult)
    grade_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    chain = grade_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("Grade score:", result.binary_score)

    return {"binary_score": result.binary_score}


def rewrite_query(state: State):
    """
    Rewrite the user query to optimize retrieval relevance.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated latest_query in the state.
    """
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    rewritten = result.content.strip()
    print("Rewritten query:", rewritten)

    return {"latest_query": rewritten}


def generate(state: State):
    """Placeholder for generation node."""
    print("Placeholder: generate node")
    return {"messages": state["messages"] + [AIMessage(content="Generate node placeholder response")]}


def web_search(state: State):
    """
    Search the web for information using Tavily API.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages in state.
    """
    query = state["latest_query"]
    print("Searching the web for:", query)
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search_tool = TavilySearchResults(max_results=3)
        results = web_search_tool.invoke(query)
        
        if isinstance(results, str):
            context_str = results
        else:
            context_str = "\n\n".join([
                f"URL: {res.get('url', '')}\nContent: {res.get('content', '')}"
                for res in results
            ])
    except Exception as e:
        print("Tavily search error, using fallback empty context:", e)
        context_str = f"No web search results found for: {query}"

    new_message = AIMessage(content=context_str)
    return {"messages": [new_message]}


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
