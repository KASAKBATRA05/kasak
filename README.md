# 🧠 GenAI NLP Document Assistant

A powerful, locally-running document assistant that uses advanced NLP techniques (not GPT) to analyze PDF and TXT documents. Built with Streamlit and state-of-the-art transformer models.

## ✨ Features

### 📄 Document Processing
- **Multi-format Support**: Upload PDF or TXT files
- **Smart Text Extraction**: Uses PyMuPDF for robust PDF text extraction
- **Document Statistics**: Get insights about your document's structure

### 📝 Intelligent Summarization
- **TextRank Algorithm**: Uses Sumy's TextRank for extractive summarization
- **Configurable Length**: Generates concise 150-word summaries
- **Multiple Algorithms**: Support for TextRank, LSA, and Luhn summarization

### ❓ Question-Answering System
- **Advanced QA Pipeline**: Powered by Haystack framework
- **RoBERTa Model**: Uses `deepset/roberta-base-squad2` for accurate answers
- **Semantic Search**: Sentence Transformers for finding relevant context
- **FAISS Integration**: Fast similarity search for large documents

### 🎯 Challenge Mode
- **Auto-Generated Questions**: Creates 3 types of questions from your document
  - Fill-in-the-blank questions
  - Yes/No questions  
  - Short answer questions
- **Smart Evaluation**: Fuzzy matching for answer assessment
- **Detailed Feedback**: Explanations with document references

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **PDF Processing**: PyMuPDF (fitz)
- **Summarization**: Sumy (TextRank, LSA, Luhn)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Question Answering**: Haystack + RoBERTa (`deepset/roberta-base-squad2`)
- **Similarity Search**: FAISS
- **NLP Processing**: NLTK
- **Answer Evaluation**: FuzzyWuzzy

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd genai-nlp-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
genai_nlp_assistant/
│
├── app.py                 # Main Streamlit application
├── backend/
│   ├── reader.py         # Document reading and text extraction
│   ├── summarizer.py     # Text summarization engine
│   ├── qna_engine.py     # Question-answering system
│   └── challenge_mode.py # Question generation and evaluation
├── utils/
│   └── embeddings.py     # Embedding management with FAISS
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎮 How to Use

### 1. Document Upload
- Navigate to "📄 Document Upload"
- Upload a PDF or TXT file
- View document preview and statistics

### 2. Generate Summary
- Go to "📝 Summary" 
- Click "Generate Summary"
- Get a concise 150-word summary of your document

### 3. Ask Questions
- Visit "❓ Ask Questions"
- Type any question about your document
- Get accurate answers with confidence scores and relevant context

### 4. Challenge Mode
- Access "🎯 Challenge Mode"
- Generate automatic questions from your document
- Answer questions and receive detailed feedback
- View your final score and explanations

## 🔧 Configuration

### Model Configuration
You can customize the models used by modifying the initialization parameters:

```python
# In qna_engine.py
self.retriever = EmbeddingRetriever(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Change model here
)

self.reader = FARMReader(
    model_name_or_path="deepset/roberta-base-squad2"  # Change QA model here
)
```

### Summarization Settings
Adjust summary length and algorithm in `summarizer.py`:

```python
# Change summarization algorithm
summarizer = DocumentSummarizer(algorithm='textrank')  # 'textrank', 'lsa', 'luhn'

# Adjust summary length
summary = summarizer.summarize(text, max_words=150)
```

## 🧪 Advanced Features

### Multiple Summarization Algorithms
```python
summaries = summarizer.generate_multiple_summaries(text)
# Returns summaries from TextRank, LSA, and Luhn algorithms
```

### Document Clustering
```python
embedding_manager = EmbeddingManager()
embedding_manager.build_index(documents)
clusters = embedding_manager.cluster_documents(num_clusters=5)
```

### Custom Question Types
Extend `challenge_mode.py` to add new question types:

```python
def _generate_custom_question(self, sentence: str) -> Dict[str, Any]:
    # Your custom question generation logic
    pass
```

## 📊 Performance

- **Processing Speed**: Handles documents up to 10MB efficiently
- **Memory Usage**: Optimized for local execution
- **Accuracy**: RoBERTa model provides high-quality answers
- **Scalability**: FAISS enables fast search even for large documents

## 🔒 Privacy & Security

- **100% Local**: No data sent to external APIs
- **No Internet Required**: All models run locally after initial download
- **Data Privacy**: Your documents never leave your machine
- **Open Source**: Full transparency in processing methods

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Errors**
   - Ensure stable internet connection for initial model downloads
   - Models are cached locally after first use

2. **Memory Issues**
   - For large documents, consider splitting into smaller chunks
   - Reduce `top_k` parameters in QA settings

3. **PDF Extraction Issues**
   - Some PDFs may have complex layouts
   - Try converting to TXT format if extraction fails

### Error Messages

- **"No text content found"**: PDF may be image-based or corrupted
- **"Error initializing Q&A system"**: Check model downloads and dependencies
- **"Index not built"**: Ensure document is uploaded and processed first

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Haystack**: For the excellent QA framework
- **Sentence Transformers**: For semantic embeddings
- **Sumy**: For text summarization algorithms
- **Streamlit**: For the beautiful web interface
- **Hugging Face**: For pre-trained transformer models

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error logs in the Streamlit interface
3. Open an issue on GitHub with detailed error information

---

**Built with ❤️ for local, privacy-focused document analysis**