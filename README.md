# Chat with Your Documents

A powerful document chat application built with Streamlit, LangChain, and LangGraph that allows you to have conversations with your documents using advanced AI models.

## Features

- 📄 Support for multiple file formats (PDF, DOCX, PPTX, PNG, JPG, HTML, MD)
- 🖼️ Advanced document processing with Docling
- 🔍 Intelligent document structure extraction (sections, tables, figures)
- 🤖 Multiple AI model support (OpenAI and Anthropic)
- 💾 Efficient document storage using Qdrant vector database
- 📊 Rich document metadata and context preservation
- 🎯 Modular and testable architecture

## Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings)
- Anthropic API key (for chat model)
- Docling (for document processing)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Alternatively, you can input your API keys through the application's UI.

## Usage

1. Start the application:
```bash
streamlit run src/main.py
```

2. Configure your API keys in the sidebar if not set through environment variables
3. Upload your documents through the sidebar
4. View detailed document structure analysis
5. Start chatting with your documents!

## Features in Detail

### Document Processing
- Automatic section detection and hierarchical structure extraction
- Table and figure detection
- Image analysis and description
- Rich metadata extraction (title, author, date)

### Vector Storage
- Efficient chunking with context preservation
- Metadata-rich storage including:
  - Page numbers
  - Section information
  - Table and figure presence
  - Image descriptions

### Chat Interface
- Multiple AI model support
- Configurable model parameters
- Context-aware responses
- Document structure awareness

## Development

### Project Structure
- `src/`
  - `components/`: UI components and interfaces
  - `utils/`: Document processing and helper functions
  - `models/`: Vector store and model configurations
  - `graphs/`: Chat flow management

### Running Tests
```bash
pytest tests/
```

## License

MIT 