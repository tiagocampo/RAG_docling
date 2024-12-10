import streamlit as st
from typing import List, Tuple, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from graphs.chat_graph import create_chat_graph
from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL
import asyncio
import uuid
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        """Initialize chat interface."""
        # Create unique chat ID if not exists
        if "chat_id" not in st.session_state:
            st.session_state.chat_id = str(uuid.uuid4())
        
        # Create chat graph
        self.chat_graph = create_chat_graph()
    
    def render(self):
        # Initialize chat history and summary
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "summary" not in st.session_state:
            st.session_state.summary = None

        # Check if any documents are uploaded
        if not st.session_state.get("processed_files"):
            st.info("👋 Welcome! Please upload some documents using the sidebar before starting the chat.")
            return

        # Display chat messages
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
                    # If this is a search result message, show the sources
                    if hasattr(message, 'additional_kwargs') and 'sources' in message.additional_kwargs:
                        with st.expander("🔍 Sources", expanded=False):
                            for i, source in enumerate(message.additional_kwargs['sources']):
                                st.markdown(f"""
                                **Source {i+1}** (Page {source.get('page', 'N/A')})
                                ```
                                {source.get('text', 'No text available')}
                                ```
                                """)

        # Display current summary in sidebar if available
        if st.session_state.summary:
            with st.sidebar:
                with st.expander("Conversation Summary", expanded=False):
                    st.write(st.session_state.summary)

        # Chat input
        if prompt := st.chat_input("Ask about your documents"):
            logger.info(f"Received user prompt: {prompt}")
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Create human message
            human_message = HumanMessage(content=prompt)
            
            # Create initial state with the new message
            messages = list(st.session_state.messages)
            messages.append(human_message)
            
            # Show searching status
            with st.status("🔍 Searching documents...", expanded=True) as status:
                try:
                    # Create and run chat graph with proper configuration
                    graph = create_chat_graph()
                    config = {
                        "configurable": {
                            "thread_id": st.session_state.chat_id,
                            "checkpoint_ns": "chat_history",
                            "checkpoint_id": f"chat_{st.session_state.chat_id}"
                        }
                    }
                    
                    logger.info("Invoking chat graph...")
                    result = graph.invoke(
                        {"messages": messages},
                        config=config
                    )
                    logger.info(f"Chat graph result: {result}")
                    
                    # Update messages with result
                    if result and "messages" in result and result["messages"]:
                        logger.info(f"Processing {len(result['messages'])} new messages")
                        for msg in result["messages"]:
                            logger.info(f"Adding message: {msg.content[:100]}...")  # Log first 100 chars
                            messages.append(msg)
                            # Immediately display the new message
                            if isinstance(msg, AIMessage):
                                with st.chat_message("assistant"):
                                    st.write(msg.content)
                                    if hasattr(msg, 'additional_kwargs') and 'sources' in msg.additional_kwargs:
                                        with st.expander("🔍 Sources", expanded=False):
                                            for i, source in enumerate(msg.additional_kwargs['sources']):
                                                st.markdown(f"""
                                                **Source {i+1}** (Page {source.get('page', 'N/A')})
                                                ```
                                                {source.get('text', 'No text available')}
                                                ```
                                                """)
                        
                        st.session_state.messages = messages
                        status.update(label="✅ Found relevant information!", state="complete")
                    else:
                        status.update(label="❌ No relevant information found", state="error")
                        logger.warning(f"No messages in result. Result structure: {result}")
                        with st.chat_message("assistant"):
                            st.write("I couldn't find relevant information in the documents to answer your question. Could you please rephrase or ask something else?")
                        
                except Exception as e:
                    logger.error(f"Error in chat processing: {str(e)}")
                    status.update(label=f"❌ Error: {str(e)}", state="error")
                    st.error("An error occurred while processing your request. Please try again.")