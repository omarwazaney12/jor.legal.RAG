# Jordan Legal RAG Assistant | المساعد القانوني الأردني

Advanced Legal Retrieval-Augmented Generation (RAG) system for Jordanian laws and regulations with sophisticated AI-powered legal consultation capabilities.

## 🌟 Features

### Core Capabilities
- **Advanced RAG System**: Hybrid semantic + keyword search for precise legal information retrieval
- **Multi-language Support**: Arabic and English query processing
- **Legal Document Processing**: Automated processing of Jordanian laws, regulations, and instructions
- **Intelligent Query Classification**: Automatic categorization of legal queries (procedures, requirements, establishment, etc.)
- **Conversation Memory**: Context-aware responses with conversation history
- **Real-time Web Interface**: Modern, responsive web UI with RTL Arabic support

### Legal Specializations
- 🏢 **Company Formation & Business Laws**: Registration procedures, corporate structures
- 📍 **Consumer Protection**: Rights, regulations, and remedies
- ® **Intellectual Property**: Trademarks, patents, and copyright laws
- 💰 **Investment Laws**: Foreign and domestic investment regulations
- 📋 **Commercial Registration**: Business licensing and registration
- 👔 **Labor Laws**: Employment rights, obligations, and procedures

### Technical Features
- **ChromaDB Vector Store**: Efficient document embedding and retrieval
- **OpenAI Integration**: GPT-4 powered legal reasoning and response generation
- **Multi-step Reasoning Engine**: Complex legal query decomposition and analysis
- **Source Attribution**: Accurate legal citations and references
- **Confidence Scoring**: AI confidence levels for response reliability
- **Error Handling & Retry Logic**: Robust error recovery and rate limit management

## 🚀 Quick Start

### Prerequisites
- Python 3.9.18+
- OpenAI API Key
- Git

### Local Development

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
```bash
export OPENAI_API_KEY="your_openai_api_key"
export FLASK_ENV="development"
```

4. **Run the application:**
```bash
cd cbj-scraper
python advanced_web_demo.py
```

5. **Access the application:**
Open your browser and go to `http://localhost:5003`

## 🌐 Deployment

### Deploy on Render

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `FLASK_ENV`: `production`
4. **Deploy using the included `render.yaml` configuration**

### Deploy on Heroku

1. **Install Heroku CLI**
2. **Create Heroku app:**
```bash
heroku create your-app-name
```

3. **Set environment variables:**
```bash
heroku config:set OPENAI_API_KEY="your_openai_api_key"
heroku config:set FLASK_ENV="production"
```

4. **Deploy:**
```bash
git push heroku main
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 access | Yes |
| `FLASK_ENV` | Environment (development/production) | No |
| `PORT` | Port number (auto-set by platforms) | No |

## 📁 Project Structure

```
├── cbj-scraper/
│   ├── advanced_web_demo.py      # Main Flask application
│   ├── advanced_rag_system.py    # Core RAG system
│   └── deploy_without_docs.py    # Deployment system
├── requirements.txt               # Python dependencies
├── Procfile                      # Heroku deployment config
├── render.yaml                   # Render deployment config
├── runtime.txt                   # Python version specification
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

### Vector Database Setup
The system uses ChromaDB for document storage and retrieval. On first run, it will:
1. Process legal documents from the `mit_jordan_data/txt_output` directory
2. Generate embeddings using OpenAI's text-embedding-3-large model
3. Build TF-IDF indices for hybrid search
4. Store everything in a persistent ChromaDB instance

### Legal Document Processing
The system automatically processes various types of legal documents:
- **Laws (قوانين)**: Primary legislation
- **Regulations (أنظمة)**: Executive regulations
- **Instructions (تعليمات)**: Administrative instructions
- **Constitution (دستور)**: Constitutional texts

## 🎯 Usage Examples

### Basic Legal Queries
```
ما هي شروط تأسيس الشركات في الأردن؟
What are the requirements for company formation in Jordan?
```

### Procedural Queries
```
كيف أسجل علامة تجارية؟
How do I register a trademark?
```

### Rights and Obligations
```
ما هي حقوق المستهلك في الأردن؟
What are consumer rights in Jordan?
```

## 🔒 Security & Legal Notice

⚠️ **Important Disclaimer**: This system provides general legal information only and is not a substitute for professional legal advice. Always consult a qualified attorney for personal legal matters.

### Data Privacy
- No personal data is stored permanently
- Conversation history is session-based only
- All queries are processed securely through OpenAI's API

### Security Features
- Input validation and sanitization
- Rate limiting and timeout handling
- Error logging and monitoring
- Secure environment variable management

## 🛠️ Development

### Adding New Features
1. **Legal Document Processing**: Extend `AdvancedLegalProcessor` class
2. **Query Types**: Add new reasoning templates in `LegalReasoningEngine`
3. **UI Components**: Modify the HTML template in `advanced_web_demo.py`

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=cbj-scraper tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📊 Performance

- **Response Time**: Typically 3-8 seconds for complex legal queries
- **Accuracy**: High precision legal information retrieval
- **Scalability**: Designed for concurrent users with session management
- **Language Support**: Optimized for Arabic legal texts with English query support

## 🆘 Troubleshooting

### Common Issues

**OpenAI API Errors:**
- Verify your API key is set correctly
- Check your OpenAI account billing status
- Monitor rate limits

**ChromaDB Issues:**
- Ensure write permissions for the `chroma_db` directory
- Check available disk space
- Restart the application to rebuild indices

**Memory Issues:**
- Consider reducing chunk size in document processing
- Monitor system resources during embedding generation

### Support
For issues and questions:
1. Check the error logs in `embedding_errors.log`
2. Review the console output for debugging information
3. Ensure all environment variables are properly set

## 📄 License

This project is for educational and research purposes. Please ensure compliance with OpenAI's usage policies and local regulations when deploying.

## 🤝 Acknowledgments

- **OpenAI**: For GPT-4 and embedding models
- **LangChain**: For RAG framework components
- **ChromaDB**: For vector database capabilities
- **Flask**: For web framework
- **Jordan Government**: For legal document availability

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Maintainer**: [Your Name/Organization] 