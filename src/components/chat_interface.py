import streamlit as st
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from graphs.chat_graph import process_question
import uuid

class ChatInterface:
    """Chat interface component for the Streamlit app."""
    
    def __init__(self):
        """Initialize chat interface."""
        # Initialize chat ID if not exists
        if "chat_id" not in st.session_state:
            st.session_state.chat_id = str(uuid.uuid4())
        
        # Initialize messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def render(self):
        """Render the chat interface."""
        # Check if any documents are uploaded
        if not st.session_state.get("processed_files"):
            st.info("👋 Welcome! Please upload some documents using the sidebar before starting the chat.")
            return
        
        # Chat input - Place it at the bottom using columns
        col1, col2 = st.columns([6, 1])
        with col1:
            # Store the input in session state
            if "user_input" not in st.session_state:
                st.session_state.user_input = ""
            
            prompt = st.text_input(
                "Ask a question about your documents",
                key="chat_input",
                value=st.session_state.user_input,
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send", use_container_width=True)
            
        # Display chat messages
        messages_container = st.container()
        
        with messages_container:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
        
        # Process the input
        if send_button and prompt:
            # Clear the input
            st.session_state.user_input = ""
            
            # Add user message
            user_message = HumanMessage(content=prompt)
            st.session_state.messages.append(user_message)
            
            # Process the question
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = process_question(prompt)
                        response = result["generation"]
                        
                        # Add assistant message
                        assistant_message = AIMessage(content=response)
                        st.session_state.messages.append(assistant_message)
                        
                        st.write(response)
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
            
            # Rerun to update the interface
            st.rerun()