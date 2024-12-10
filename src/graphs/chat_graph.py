import logging
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from models.model_factory import ModelFactory
from models.retriever import create_retriever_tool
from models.vectorstore import get_vectorstore
from services.chat_node_service import ChatNodeService
from config.prompt_config import GRADER_PROMPT

logger = logging.getLogger(__name__)
memory = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def create_chat_graph():
    """Create the chat graph with agentic RAG."""
    workflow = StateGraph(AgentState)
    chat_service = ChatNodeService()
    
    # Add nodes
    workflow.add_node("agent", lambda state: chat_service.process_agent(state["messages"]))
    retrieve = ToolNode([create_retriever_tool(get_vectorstore())])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", lambda state: chat_service.process_rewrite(state["messages"]))
    workflow.add_node("generate", lambda state: chat_service.process_generate(state["messages"]))
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    return workflow.compile(checkpointer=memory)

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Grade the relevance of retrieved documents."""
    logger.info("Grading document relevance")
    
    model = ModelFactory.create_model()
    llm_with_tool = model.with_structured_output(Grade)
    chain = GRADER_PROMPT | llm_with_tool
    
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "context": docs})
    logger.info(f"Document relevance score: {scored_result.binary_score}")
    return "generate" if scored_result.binary_score == "yes" else "rewrite" 