import streamlit as st
from typing import List, Tuple, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from graphs.chat_graph import create_chat_graph
import asyncio
import uuid
import os

class ChatInterface:
    def __init__(self):
        # Check for required API keys
        if not st.session_state.get("openai_api_key"):
            raise ValueError("OpenAI API key is required for embeddings. Please enter it in the sidebar.")
        if not st.session_state.get("anthropic_api_key"):
            raise ValueError("Anthropic API key is required for Claude models. Please enter it in the sidebar.")
        
        # Create the graph once when the interface is initialized
        self.graph = create_chat_graph()
        
        # Initialize or get the chat session ID
        if "chat_id" not in st.session_state:
            st.session_state.chat_id = str(uuid.uuid4())

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
                    # Create and run chat graph
                    graph = create_chat_graph()
                    result = graph.invoke({
                        "messages": messages
                    })
                    
                    # Update messages with result
                    if result and "messages" in result:
                        messages.extend(result["messages"])
                        st.session_state.messages = messages
                        status.update(label="✅ Found relevant information!", state="complete")
                    else:
                        status.update(label="❌ No relevant information found", state="error")
                        
                except Exception as e:
                    logger.error(f"Error in chat processing: {str(e)}")
                    status.update(label=f"❌ Error: {str(e)}", state="error")