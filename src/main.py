import streamlit as st
from components.file_uploader import FileUploader
from components.chat_interface import ChatInterface
from utils.session_state import initialize_session_state
from graphs.chat_graph import AVAILABLE_MODELS, DEFAULT_MODEL
import os

st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_document_structure():
    """Display the structure of processed documents."""
    if "document_structures" in st.session_state and st.session_state.document_structures:
        st.header("Document Structures")
        for filename, structure in st.session_state.document_structures.items():
            with st.expander(f"📄 {filename}"):
                st.write("**Title:**", structure.get("title", "N/A"))
                st.write("**Author:**", structure.get("author", "N/A"))
                st.write("**Date:**", structure.get("date", "N/A"))
                st.write("**Pages:**", structure["num_pages"])
                
                if structure["sections"]:
                    st.write("**Sections:**")
                    for section in structure["sections"]:
                        st.write(f"- {section['title']} (Page {section['page']})")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tables", structure["num_tables"])
                with col2:
                    st.metric("Charts", structure["num_charts"])
                with col3:
                    st.metric("Named Entities", structure["num_entities"])

def handle_api_key():
    """Handle API key input and storage."""
    with st.sidebar:
        with st.expander("🔑 API Settings", expanded=not (bool(st.session_state.get("openai_api_key")) and bool(st.session_state.get("anthropic_api_key")))):
            if st.session_state.get("api_key_source") == "env":
                st.success("API keys loaded from environment!")
                if st.button("Clear API Keys"):
                    st.session_state.openai_api_key = ""
                    st.session_state.anthropic_api_key = ""
                    st.session_state.api_key_source = "user"
                    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                        if key in os.environ:
                            del os.environ[key]
                    st.rerun()
            else:
                st.info("""This application requires two API keys:
                1. OpenAI API key for generating embeddings (vector representations) of documents
                2. Anthropic API key for the Claude chat model""")
                
                # OpenAI API Key
                openai_key = st.text_input(
                    "OpenAI API Key (for embeddings)",
                    type="password",
                    value=st.session_state.get("openai_api_key", ""),
                    help="Enter your OpenAI API key. This is used only for generating document embeddings."
                )
                if openai_key:
                    st.session_state.openai_api_key = openai_key
                    os.environ["OPENAI_API_KEY"] = openai_key
                
                # Anthropic API Key
                anthropic_key = st.text_input(
                    "Anthropic API Key (for Claude)",
                    type="password",
                    value=st.session_state.get("anthropic_api_key", ""),
                    help="Enter your Anthropic API key. This is used for the Claude chat model."
                )
                if anthropic_key:
                    st.session_state.anthropic_api_key = anthropic_key
                    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                
                if openai_key and anthropic_key:
                    st.session_state.api_key_source = "user"
                    
                    # Option to save to .env
                    if st.button("Save API Keys to .env"):
                        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
                        with open(env_path, "w") as f:
                            f.write(f"OPENAI_API_KEY={openai_key}\n")
                            f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
                        st.success("API keys saved to .env file!")
                else:
                    st.warning("Both API keys are required: OpenAI for embeddings and Anthropic for the chat model.")

def main():
    initialize_session_state()
    
    st.title("📚 Chat with Your Documents")
    
    # Handle API key
    handle_api_key()
    
    # Only show the rest of the UI if API key is set
    if st.session_state.get("anthropic_api_key"):
        # Sidebar for file upload and settings
        with st.sidebar:
            st.header("Document Upload")
            
            FileUploader().render()
            
            # Model settings
            with st.expander("🤖 Model Settings", expanded=True):
                # Model selection with descriptions
                model_options = list(AVAILABLE_MODELS.keys())
                current_model = st.session_state.get("model_name", DEFAULT_MODEL)
                
                st.selectbox(
                    "Model",
                    options=model_options,
                    index=model_options.index(current_model),
                    key="model_name",
                    help="Select the model for chat interactions"
                )
                
                # Show selected model details
                selected_model = st.session_state.get("model_name", DEFAULT_MODEL)
                model_config = AVAILABLE_MODELS[selected_model]
                
                st.write("**Model Details:**")
                st.write(f"- **Provider:** {model_config['provider'].title()}")
                st.write(f"- **Description:** {model_config['description']}")
                st.write(f"- **Temperature:** {model_config['temperature']}")
                st.write(f"- **Max Tokens:** {model_config['max_tokens']}")
                
                # Show API key status
                if model_config["provider"] == "openai":
                    key_status = "✅" if st.session_state.get("openai_api_key") else "❌"
                    st.write(f"- **OpenAI API Key:** {key_status}")
                else:
                    key_status = "✅" if st.session_state.get("anthropic_api_key") else "❌"
                    st.write(f"- **Anthropic API Key:** {key_status}")
        
        # Main layout with two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ChatInterface().render()
        
        with col2:
            # Document structure display
            display_document_structure()
    else:
        st.info("Please enter your Anthropic API key in the sidebar to start using the application.")

if __name__ == "__main__":
    main() 