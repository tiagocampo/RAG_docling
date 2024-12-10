import streamlit as st
from typing import Annotated, Sequence, Literal, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from models.vectorstore import get_vectorstore
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
import logging

from services.grading_service import GradingService
from services.routing_service import RoutingService
from services.web_search_service import WebSearchService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory saver
memory = MemorySaver()

# Define available models and their configurations
AVAILABLE_MODELS = {
    "Claude 3.5 Sonnet": {
        "provider": "anthropic",
        "name": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 8192,
        "description": "Most intelligent Anthropic model, best for complex tasks and deep analysis"
    },
    "Claude 3.5 Haiku": {
        "provider": "anthropic",
        "name": "claude-3-5-haiku-20241022",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 8192,
        "description": "Fast Anthropic model, great for quick responses while maintaining quality"
    },
    "GPT-4o": {
        "provider": "openai",
        "name": "gpt-4o-2024-11-20",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 16384,
        "description": "OpenAI's most advanced model, 2x faster than GPT-4 Turbo with multimodal capabilities"
    },
    "GPT-4o Mini": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 16384,
        "description": "OpenAI's affordable and intelligent small model, more capable than GPT-3.5 Turbo"
    }
}

# Default model
DEFAULT_MODEL = "GPT-4o"

def get_model():
    """Get the model based on configuration."""
    selected_model = st.session_state.get("model_name", DEFAULT_MODEL)
    model_config = AVAILABLE_MODELS[selected_model]
    
    if model_config["provider"] == "anthropic":
        return ChatAnthropic(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=model_config["streaming"]
        )
    else:  # OpenAI
        return ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            streaming=model_config["streaming"]
        )

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