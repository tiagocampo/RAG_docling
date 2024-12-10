import streamlit as st
from typing import Annotated, Sequence, Literal, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from models.vectorstore import get_vectorstore
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import logging

from services.grading_service import GradingService
from services.routing_service import RoutingService
from services.web_search_service import WebSearchService
from models.model_manager import get_model, AVAILABLE_MODELS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory saver
memory = MemorySaver()

# Initialize services
grading_service = GradingService()
routing_service = RoutingService()
web_search_service = WebSearchService()

class GraphState(TypedDict):
    """State maintained between steps."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_question: str
    documents: List[Dict[str, Any]]
    generation: str
    failed_retrievals: int
    source: str

def initialize_state(question: str) -> GraphState:
    """Initialize the graph state with a question."""
    return {
        "messages": [],
        "current_question": question,
        "documents": [],
        "generation": "",
        "failed_retrievals": 0,
        "source": ""
    }

def route_question(state: GraphState) -> str:
    """Route the question to appropriate data source."""
    question = state["current_question"]
    route = routing_service.route_question(question)
    state["source"] = route.datasource
    logger.info(f"Routing question to: {route.datasource} ({route.explanation})")
    return route.datasource

def retrieve_from_vectorstore(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents from vectorstore."""
    logger.info("Retrieving from vectorstore")
    question = state["current_question"]
    
    # Get documents from vectorstore
    vectorstore = get_vectorstore()
    documents = vectorstore.similarity_search(question, k=3)
    
    return {
        **state,
        "documents": documents
    }

def web_search(state: GraphState) -> Dict[str, Any]:
    """Perform web search."""
    logger.info("Performing web search")
    question = state["current_question"]
    
    # Get web search results
    results = web_search_service.search(question)
    
    # If we also have vectorstore results, combine them
    if state.get("documents"):
        results = web_search_service.combine_results(results, state["documents"])
    
    return {
        **state,
        "documents": results
    }

def grade_documents(state: GraphState) -> str:
    """Grade retrieved documents and decide next step."""
    documents = state["documents"]
    question = state["current_question"]
    
    # Grade documents
    grade = grading_service.grade_documents(question, documents)
    logger.info(f"Document relevance: {grade}")
    
    if grade.binary_score == "yes":
        return "generate"
    
    # Increment failed retrievals
    state["failed_retrievals"] = state.get("failed_retrievals", 0) + 1
    
    # Check if we should try web search
    if state["source"] == "vectorstore" and routing_service.should_use_web_search(
        question, state["failed_retrievals"]
    ):
        return "web_search"
    
    return "rewrite"

def rewrite_question(state: GraphState) -> Dict[str, Any]:
    """Rewrite the question for better retrieval."""
    question = state["current_question"]
    failed_retrievals = state["failed_retrievals"]
    
    # Get rewritten question
    rewritten = grading_service.suggest_rewrite(question, failed_retrievals)
    
    return {
        **state,
        "current_question": rewritten
    }

def generate_response(state: GraphState) -> Dict[str, Any]:
    """Generate response using retrieved documents."""
    model = get_model()  # Use existing get_model function
    question = state["current_question"]
    documents = state["documents"]
    
    # Create system message
    system_message = SystemMessage(content="""You are a helpful AI assistant that answers questions based on the provided documents.
    Always cite your sources and be honest if you're not sure about something.
    If using web search results, mention that the information comes from the web.""")
    
    # Create messages
    messages = [
        system_message,
        HumanMessage(content=question)
    ]
    
    try:
        # Generate response
        response = model.invoke(messages)
        
        # Add source information
        sources = [doc.get("metadata", {}).get("source", "unknown") for doc in documents]
        if "web_search" in sources:
            response.content += "\n\n(Information from web search results)"
        
        return {
            **state,
            "generation": response.content,
            "messages": state["messages"] + [AIMessage(content=response.content)]
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        error_msg = "I encountered an error generating a response. Please try again."
        return {
            **state,
            "generation": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

def create_chat_graph():
    """Create the chat graph with adaptive RAG."""
    # Initialize graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("vectorstore", retrieve_from_vectorstore)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_response)
    
    # Add edges
    workflow.set_entry_point("route_question")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "route_question",
        route_question,
        {
            "vectorstore": "vectorstore",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("vectorstore", "grade_documents")
    workflow.add_edge("web_search", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("rewrite", "vectorstore")
    workflow.add_edge("generate", END)
    
    return workflow.compile(checkpointer=memory)

def process_question(question: str) -> Dict[str, Any]:
    """Process a question through the graph."""
    graph = create_chat_graph()
    initial_state = initialize_state(question)
    
    # Create configuration for the memory checkpointer
    config = {
        "configurable": {
            "thread_id": st.session_state.get("chat_id", "default"),
            "checkpoint_ns": "chat_history",
            "checkpoint_id": f"chat_{st.session_state.get('chat_id', 'default')}"
        }
    }
    
    result = graph.invoke(initial_state, config=config)
    return result