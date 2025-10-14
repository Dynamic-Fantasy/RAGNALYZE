#  RAG Feedback Analyzer

An intelligent document Q&A system built with Streamlit, LangChain, and Groq LLM that enables interactive analysis of customer feedback from PDF documents using Retrieval Augmented Generation (RAG).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

##  Features

- **PDF Document Processing**: Upload and analyze customer feedback documents in PDF format
- **Intelligent Q&A**: Ask natural language questions about your feedback data
- **Multiple Query Types**: 
  - List and retrieve specific feedback items
  - Search for solutions and improvements
  - Get detailed analysis of feedback content
- **Export Functionality**: Export chat conversations to professionally formatted PDF files
- **Real-time Processing**: Instant responses powered by Groq's fast LLM inference
- **Beautiful UI**: Modern gradient design with intuitive chat interface
- **Debug Mode**: Built-in logging system for troubleshooting
- **Semantic Search**: Uses HuggingFace embeddings with ChromaDB for accurate retrieval

##  Technology Stack

### Core Frameworks
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and RAG implementation
- **LangGraph**: Agent workflow management

### AI/ML Components
- **Groq**: Fast LLM inference (supports GPT-OSS, LLaMA 3.1, Mixtral)
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **ChromaDB**: Vector database for semantic search

### Document Processing
- **PyPDF**: PDF text extraction
- **ReportLab**: PDF generation for exports
- **RecursiveCharacterTextSplitter**: Intelligent text chunking

##  Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com))
- 2GB+ RAM (for embeddings model)

##  Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-feedback-analyzer.git
cd rag-feedback-analyzer
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create requirements.txt** (if not exists)
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-groq>=0.0.1
langchain-chroma>=0.1.0
langgraph>=0.0.20
pypdf>=3.17.0
chromadb>=0.4.18
sentence-transformers>=2.2.2
reportlab>=4.0.0
typing-extensions>=4.8.0
streamlit>=1.50.0
```

##  Usage

1. **Start the application**
```bash
streamlit run web_ragnalyze.py
```

2. **Configure the system** (in sidebar)
   - Enter your Groq API key
   - Upload a PDF document containing feedback
   - Select your preferred LLM model
   - Adjust advanced settings if needed (chunk size, overlap, retrieval count)

3. **Initialize the system**
   - Click "🚀 Initialize System"
   - Wait for processing to complete

4. **Start chatting**
   - Ask questions about your feedback
   - Examples:
     - "Show me top 5 feedbacks"
     - "What are the common complaints?"
     - "How can we fix the login issues?"
     - "List all feedbacks about performance"

5. **Export conversations**
   - Type: "Export this chat to PDF"
   - File will be saved in the current directory

## 🎯 Key Capabilities

### 1. Feedback Listing
```
User: "Show me top 10 feedbacks"
AI: Returns numbered list with Feedback IDs and page numbers
```

### 2. Specific Queries
```
User: "What solutions are mentioned for performance issues?"
AI: Searches and combines relevant information from multiple feedbacks
```

### 3. Chat Export
```
User: "Export this conversation"
AI: Creates formatted PDF with full conversation history
```

## 🏗️ Architecture

### RAG Pipeline
```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                            ↓
User Query → LLM Agent → Tool Selection → Retrieval → Response
```

### Agent Tools

1. **show_feedbacks_tool**: Lists multiple feedback items
   - Triggered by: "list", "show", "top N", "all feedbacks"
   - Returns: Numbered list with metadata

2. **retriever_tool**: Searches document content
   - Triggered by: Specific questions, "how to fix", "solutions"
   - Returns: Relevant context from documents

3. **export_chat_to_pdf**: Exports conversation
   - Triggered by: "export", "save", "download"
   - Returns: PDF file with formatted chat history

## ⚙️ Configuration Options

### Model Selection
- `openai/gpt-oss-20b`: Balanced performance
- `llama-3.1-70b-versatile`: Most capable
- `mixtral-8x7b-32768`: Large context window

### Advanced Settings
- **Chunk Size**: 100-2000 (default: 1000)
- **Chunk Overlap**: 0-500 (default: 200)
- **Retrieval K**: 1-10 (default: 5)

##  UI Customization

The application uses custom CSS for styling. Key UI elements:

- **Gradient backgrounds**: Blue-purple theme
- **Chat bubbles**: User (blue) vs AI (pink-red)
- **Status badges**: Visual indicators for system state
- **Debug logs**: Monospace terminal-style logging

##  Project Structure

```
rag-feedback-analyzer/
├── web_ragnalyze.py       # Web Based file
├── requirements.txt        # Python dependencies
├── chroma_db/             # Vector store (generated)
├── chat_export.pdf        # Exported chats (generated)
├── adv_ragnalyze.py       # Terminal Based file
└── README.md              # Documentation
```

##  Troubleshooting

### Common Issues

**Issue**: "No module named 'streamlit'"
```bash
Solution: pip install -r requirements.txt
```

**Issue**: "API key not found"
```
Solution: Ensure you've entered your Groq API key in the sidebar
```

**Issue**: "ChromaDB error"
```bash
Solution: Delete chroma_db/ folder and reinitialize
```

**Issue**: Slow processing
```
Solution: Reduce chunk size or retrieval K value in settings
```

##  Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- The application stores data locally in ChromaDB
- Exported PDFs are saved in the application directory

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG implementation
- [Groq](https://groq.com/) for fast LLM inference
- [HuggingFace](https://huggingface.co/) for embeddings models


##  Roadmap

- [ ] Multi-document support
- [ ] Export to multiple formats (DOCX, HTML)
- [ ] User authentication
- [ ] Cloud deployment guide
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] API endpoint for integrations

---

**Built with ❤️ using Streamlit and LangChain**
