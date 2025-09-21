# Developer Guide

This file provides comprehensive development guidance for the Resume Relevance Checker system.

## Development Commands

### Project Setup and Installation
```bash
# Initial setup (runs interactive setup)
python setup.py

# Install dependencies only (if venv already created)
python setup.py --install-deps

# Install dependencies directly
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm
```

### Running the Application
```bash
# Start the main Streamlit application
streamlit run src/app.py

# Run with specific port
streamlit run src/app.py --server.port 8502

# Run FastAPI backend (if needed for future development)
uvicorn src.app:app --reload --port 8000
```

### Development and Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_parser.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run single test function
python -m pytest tests/test_scorer.py::TestRelevanceScorer::test_skills_match -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/ --line-length 100

# Lint with flake8
flake8 src/ tests/ --max-line-length=100

# Type checking (if mypy is added)
mypy src/
```

### File Processing Testing
```bash
# Test resume parsing directly
python -c "
from src.resume_parser import ResumeParser
parser = ResumeParser()
result = parser.parse_resume('path/to/resume.pdf')
print(result)
"

# Test job analysis
python -c "
from src.job_analyzer import JobAnalyzer
analyzer = JobAnalyzer()
result = analyzer.analyze_job_description('Your job description text here')
print(result)
"
```

## Project Architecture

### Core Processing Pipeline
The application follows a modular pipeline architecture:

1. **Document Processing Layer** (`resume_parser.py`, `job_analyzer.py`)
   - Resume parsing: PDF/DOCX → structured data extraction
   - Job analysis: Text → requirement extraction and categorization
   - Handles multiple file formats with fallback mechanisms

2. **Analysis Engine** (`relevance_scorer.py`, `semantic_matcher.py`, `langchain_workflow.py`, `vector_store.py`)
   - **LangChain Orchestration**: Structured workflow using LangGraph for stateful processing
   - **Vector Store Integration**: ChromaDB/FAISS for embeddings and semantic search
   - **Traditional scoring**: Rule-based matching with weighted categories
   - **Enhanced semantic analysis**: LangChain-orchestrated LLM analysis
   - **Dual scoring system**: Combines hard keyword matching with vector similarity

3. **Web Interface** (`app.py`)
   - Streamlit-based UI with multiple analysis modes
   - Real-time processing with progress feedback
   - Export capabilities for analysis results

### Key Components Integration

**ResumeParser** → **JobAnalyzer** → **LangChainWorkflow** (LangGraph)
                                           ↓
                                    **VectorStore** (ChromaDB/FAISS)
                                           ↓
                                    **Results Aggregation** ← **RelevanceScorer**
                                           ↓
                                    **Streamlit UI Display**

### Data Flow Architecture
```
Input Files (PDF/DOCX) 
    ↓
Text Extraction (PyMuPDF/pdfplumber/docx2txt)
    ↓
NLP Processing (spaCy/NLTK)
    ↓
Structured Data Extraction
    ↓
LangChain Workflow Execution:
├── Hard Match Node (Keyword/Skill matching)
├── Semantic Match Node (LLM-based analysis) 
├── Score Combination Node (Weighted scoring)
├── Gap Analysis Node (Missing skills/experience)
└── Recommendations Node (LLM-generated advice)
    ↓
Vector Store Similarity (ChromaDB/FAISS)
    ↓
Final Score + Recommendations
```

### Scoring Algorithm Design
The relevance scoring uses a sophisticated weighted approach:

- **Skills Analysis**: Direct + fuzzy matching with skill categorization (programming languages, frameworks, tools, etc.)
- **Experience Evaluation**: Years of experience + seniority level alignment
- **Education Assessment**: Degree level hierarchy matching + field relevance
- **Projects Relevance**: Technology stack overlap analysis
- **Certification Matching**: Industry-standard certification recognition

### LangChain Workflow Architecture
The system implements a fully compliant LangChain-based architecture:

#### LangGraph Workflow (`langchain_workflow.py`):
- **Stateful Pipeline**: Uses LangGraph for structured, stateful resume processing
- **Node-based Processing**: Each step (hard match, semantic match, scoring) as separate nodes
- **LangSmith Traceability**: @traceable decorators for debugging and observability
- **Orchestrated LLM Calls**: Proper LangChain prompt templates and chains

#### Vector Store Integration (`vector_store.py`):
- **ChromaDB Primary**: Persistent vector storage with embedding management
- **FAISS Fallback**: High-performance similarity search when ChromaDB unavailable
- **Sentence Transformers**: Generate embeddings for semantic comparison
- **Cosine Similarity**: Proper embedding-based relevance scoring

### Environment Configuration
Critical environment variables (`.env` file):
- `OPENAI_API_KEY`: Required for enhanced semantic analysis
- `WEIGHT_*`: Configurable scoring weights for different categories
- `MIN_SCORE_*`: Customizable thresholds for High/Medium/Low classification
- `OPENAI_MODEL`: Language model selection for analysis

### File Processing Strategy
The parser implements a robust multi-library approach:
- **PDF**: PyMuPDF (primary) → pdfplumber (fallback)
- **DOCX**: docx2txt (primary) → python-docx (fallback)
- **Error Handling**: Graceful degradation with informative error messages

### Extensibility Points
- **Skill Databases**: Easily expandable skill pattern matching in `_load_skill_patterns()`
- **Scoring Weights**: Configurable via environment variables
- **Language Models**: Configurable models and parameters for semantic analysis
- **Export Formats**: JSON structure designed for easy extension

### Data Security Considerations
- Temporary file cleanup after processing
- No persistent storage of uploaded documents
- API keys managed through environment variables
- File size validation and sanitization

### Performance Optimizations
- Lazy loading of NLP models
- Cached skill pattern matching
- Efficient text processing with minimal memory footprint
- Async-ready architecture for future scaling

## Development Tips

- **Environment Setup**: Always use the provided `setup.py` script for initial configuration
- **Testing Strategy**: The application includes sample data generation (`utils.create_sample_data()`) for development testing
- **Error Handling**: Each module includes comprehensive error handling with fallback mechanisms
- **Logging**: Structured logging is available through `utils.setup_logging()`
- **File Management**: Use `utils.save_uploaded_file()` for consistent file handling

## Key Integration Points

When modifying the codebase:

1. **Adding New Skills**: Update `_load_skill_patterns()` in both `resume_parser.py` and `job_analyzer.py`
2. **Scoring Adjustments**: Modify weights in `relevance_scorer.py` or through environment variables
3. **UI Changes**: Streamlit components are modular in `app.py` with clear separation between modes
4. **LangChain Workflow**: Extend nodes in `langchain_workflow.py` for additional processing steps
5. **Vector Store**: Configure embedding models and vector databases in `vector_store.py`

## Dependencies Notes

- **Critical Dependencies**: PyMuPDF, pdfplumber, docx2txt, spacy, openai, streamlit
- **Optional Dependencies**: FAISS and ChromaDB are included for future vector search capabilities
- **Development Dependencies**: pytest, black, flake8 for code quality and testing
- **Model Dependencies**: en_core_web_sm spaCy model required for NLP processing