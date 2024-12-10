from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from config.prompt_config import SYSTEM_PROMPT, REWRITE_PROMPT
from models.model_factory import ModelFactory
from models.retriever import create_retriever_tool
from models.vectorstore import get_vectorstore
from services.document_service import DocumentService
import logging

logger = logging.getLogger(__name__)

class ChatNodeService:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.document_service = DocumentService()

    def process_agent(self, messages: List[BaseMessage]) -> Dict:
        """Process the agent node."""
        logger.info("Processing agent node")
        model = self.model_factory.create_model()
        tools = [create_retriever_tool(get_vectorstore())]
        
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        full_messages = [system_message] + messages
        
        try:
            model_with_tools = model.bind_tools(tools)
            response = model_with_tools.invoke(full_messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}")
            return {"messages": [AIMessage(content="I encountered an error while processing your request. Please try again.")]}

    def process_rewrite(self, messages: List[BaseMessage]) -> Dict:
        """Process the rewrite node."""
        logger.info("Processing rewrite node")
        question = self.document_service.get_question_from_messages(messages)
        
        msg = [HumanMessage(content=REWRITE_PROMPT.format(question=question))]
        model = self.model_factory.create_model()
        response = model.invoke(msg)
        return {"messages": [response]}

    def process_generate(self, messages: List[BaseMessage]) -> Dict:
        """Process the generate node."""
        logger.info("Processing generate node")
        question = self.document_service.get_question_from_messages(messages)
        last_message = messages[-1]
        
        context, sources = self.document_service.extract_content_and_sources(last_message)
        
        model = self.model_factory.create_model()
        chain = ANSWER_PROMPT | model
        response = chain.invoke({"context": context, "question": question})
        
        return {"messages": [AIMessage(
            content=str(response),
            additional_kwargs={"sources": sources} if sources else {}
        )]} 