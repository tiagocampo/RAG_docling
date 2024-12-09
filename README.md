# Chat with Your Documents

A powerful document chat application built with Streamlit, LangChain, and LangGraph that allows you to have conversations with your documents.

## Features

- 📄 Support for multiple file formats (PDF, DOCX, TXT, PPTX, HTML)
- 🖼️ Image processing with OCR capabilities
- 💾 Efficient document storage using Qdrant vector database
- 🤖 Advanced conversation capabilities using LangGraph
- 🎯 Modular and testable architecture

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tesseract OCR (for image processing)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run src/main.py
```

2. Upload your documents through the sidebar
3. Start chatting with your documents!

## Development

### Running Tests
```bash
pytest tests/
```

### Project Structure

- `src/`
  - `components/`: UI components
  - `utils/`: Helper functions
  - `models/`: Vector store setup
  - `chains/`: LangChain components
  - `graphs/`: LangGraph workflows
- `tests/`
  - `unit/`: Unit tests
  - `integration/`: Integration tests

## License

MIT 