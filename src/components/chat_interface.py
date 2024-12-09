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
                st.chat_message("assistant").write(message.content)

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
            initial_state = {
                "messages": messages,
                "summary": st.session_state.summary,
                "next": None
            }
            
            # Create configuration for checkpointer
            config = {
                "configurable": {
                    "thread_id": st.session_state.chat_id,
                    "checkpoint_id": f"chat_{st.session_state.chat_id}",
                    "checkpoint_ns": "chat_history"
                }
            }
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Process response
                async def process_response():
                    # Get response from graph
                    response = await self.graph.ainvoke(initial_state, config)
                    return response
                
                # Run the async operation
                try:
                    response = loop.run_until_complete(process_response())
                    # Update messages and summary in session state
                    if response and "messages" in response:
                        st.session_state.messages = response["messages"]
                        st.session_state.summary = response.get("summary")
                        # Display the last message
                        last_message = response["messages"][-1]
                        message_placeholder.markdown(last_message.content)
                finally:
                    loop.close() 