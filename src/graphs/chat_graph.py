import streamlit as st
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from models.vectorstore import get_vectorstore
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import os
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory saver
memory = MemorySaver()

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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

class LoggedRetriever(BaseRetriever):
    """A retriever that logs its operations."""
    
    vectorstore: Any = Field(description="The vector store to retrieve from")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query with logging."""
        logger.info(f"Searching documents with query: {query}")
        results = self.vectorstore.similarity_search(query, k=3)
        logger.info(f"Found {len(results)} documents")
        for i, doc in enumerate(results):
            logger.info(f"Document {i + 1}:")
            logger.info(f"Content: {doc.page_content[:200]}...")  # First 200 chars
            logger.info(f"Metadata: {doc.metadata}")
        return results

def get_retriever_tool():
    """Create the retriever tool with the vectorstore."""
    vectorstore = get_vectorstore()
    retriever = LoggedRetriever(vectorstore=vectorstore)
    
    return create_retriever_tool(
        retriever,
        "search_documents",
        "Search through the uploaded documents to find relevant information. Use this tool to find context for answering questions."
    )

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

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Grade the relevance of retrieved documents."""
    logger.info("Grading document relevance")
    
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    model = get_model()
    llm_with_tool = model.with_structured_output(Grade)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved documents to a user question.
        
        Here is the retrieved document:
        {context}
        
        Here is the user question:
        {question}
        
        If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm_with_tool
    
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    
    logger.info(f"Document relevance score: {score}")
    return "generate" if score == "yes" else "rewrite"

def agent(state):
    """Process the conversation using the agent."""
    logger.info("Calling agent")
    messages = state["messages"]
    model = get_model()
    tools = [get_retriever_tool()]
    
    # Create a proper system message
    system_message = SystemMessage(content="""You are a helpful AI assistant that answers questions based on documents.
    You have access to a search tool that can find relevant information in the documents.
    
    ALWAYS follow these rules:
    1. ALWAYS use the search_documents tool first to find relevant information
    2. Use the information from the search results to answer questions
    3. If the search doesn't return useful results, try rephrasing the search
    4. Be honest if you can't find relevant information
    5. Cite specific sections from the documents in your answers
    
    When using the search tool:
    1. Start with a focused search query
    2. If needed, do multiple searches with different queries
    3. Combine information from multiple searches if necessary""")
    
    # Add system message to the conversation
    full_messages = [system_message] + messages
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # Process with proper tool handling
    try:
        response = model_with_tools.invoke(full_messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent processing: {str(e)}")
        return {"messages": [AIMessage(content="I encountered an error while processing your request. Please try again.")]}

def rewrite(state):
    """Rewrite the query for better retrieval."""
    logger.info("Rewriting query")
    messages = state["messages"]
    question = messages[0].content
    
    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent/meaning.
            
            Here is the initial question:
            {question}
            
            Formulate an improved question that will help find more relevant information:"""
        )
    ]
    
    model = get_model()
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """Generate the final answer."""
    logger.info("Generating answer")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    
    # Handle different types of tool outputs
    if hasattr(last_message, 'content') and isinstance(last_message.content, (list, tuple)):
        # Handle retriever output (list of documents)
        docs = last_message.content
        sources = [{
            "page": doc.metadata.get("page", "N/A"),
            "text": doc.page_content,
            "section": doc.metadata.get("section", "N/A")
        } for doc in docs]
        context = "\n\n".join(str(doc.page_content) for doc in docs)
    else:
        # Handle string content
        context = str(last_message.content)
        sources = []
    
    prompt = PromptTemplate(
        template="""You are a helpful AI assistant answering questions based on the provided documents.
        Use the following context to answer the question. If you don't know the answer, just say that.
        Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        Context: {context}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    model = get_model()
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    
    # Create AIMessage with sources
    return {"messages": [AIMessage(
        content=str(response),
        additional_kwargs={"sources": sources} if sources else {}
    )]}

def create_chat_graph():
    """Create the chat graph with agentic RAG."""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent)
    retrieve = ToolNode([get_retriever_tool()])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Add conditional edges for tool use
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END
        }
    )
    
    # Add conditional edges for document relevance
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    # Compile the graph
    return workflow.compile(checkpointer=memory) 