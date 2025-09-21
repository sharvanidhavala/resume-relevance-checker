# Resume Relevance Checker

A powerful system that automatically analyzes resumes against job descriptions, providing detailed relevance scoring and actionable feedback. Built for HR teams and placement professionals who need to efficiently screen large volumes of candidate applications.

## Features

- **Smart Resume Parsing**: Extract skills, education, experience, and projects from PDF/DOCX resumes
- **Job Description Analysis**: Parse and understand job requirements and preferred qualifications
- **Dual Matching System**:
  - **Hard Match**: Direct keyword and skill matching
  - **Semantic Match**: Advanced contextual relevance analysis using natural language processing
- **Comprehensive Scoring**: 0-100 relevance score with detailed breakdown
- **Gap Analysis**: Identify missing skills, certifications, and experience
- **Actionable Feedback**: Provide improvement suggestions for candidates
- **Interactive Dashboard**: Search, filter, and manage candidate applications

## Tech Stack

- **Backend**: Python with FastAPI/Streamlit
- **Text Processing**: PyMuPDF, pdfplumber, docx2txt
- **NLP**: spaCy, NLTK for text processing
- **NLP/Semantic Analysis**: Advanced language processing for contextual understanding
- **Vector Search**: FAISS for skill embeddings
- **Database**: SQLite (development) / PostgreSQL (production)
- **Frontend**: Streamlit for rapid prototyping

## Project Structure

```
resume-relevance-checker/
├── src/
│   ├── __init__.py
│   ├── app.py                 # Main Streamlit application
│   ├── resume_parser.py       # Resume text extraction and parsing
│   ├── job_analyzer.py        # Job description analysis
│   ├── relevance_scorer.py    # Scoring algorithms
│   ├── semantic_matcher.py    # Advanced semantic matching
│   └── utils.py               # Helper functions
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration settings
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   └── test_scorer.py
├── uploads/                   # Uploaded files storage
├── sample_data/              # Sample resumes and job descriptions
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Quick Start

### Simple Installation

1. **Download or clone** this project
2. **Run the installer**:
   ```bash
   python install.py
   ```
3. **Start the application**:
   ```bash
   streamlit run src/app.py
   ```
4. **Open your browser** to `http://localhost:8501`

### Manual Installation (Alternative)

1. **Prerequisites**: Python 3.8+ and pip
2. **Install core dependencies**:
   ```bash
   pip install streamlit pandas numpy scikit-learn spacy nltk
   pip install PyMuPDF pdfplumber python-docx docx2txt
   python -m spacy download en_core_web_sm
   ```
3. **For advanced features** (optional):
   ```bash
   pip install openai langchain langchain-openai sentence-transformers
   ```
4. **Create `.env` file** with your API key (if using advanced features):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## How It Works

### 1. Upload Process
- Recruiters upload job descriptions
- Candidates upload resumes (PDF/DOCX format)

### 2. Text Extraction & Parsing
- Extract text from documents using OCR and parsing libraries
- Identify key sections: skills, experience, education, projects
- Normalize and clean extracted data

### 3. Matching Algorithm

#### Hard Matching
- Direct keyword matching between resume and job description
- Skill-based matching using predefined skill databases
- Education and certification matching

#### Advanced Semantic Matching
- Natural language processing for deeper understanding
- Context-aware skill relevance assessment
- Intelligent project and experience relevance evaluation

### 4. Scoring & Output
- **Overall Score**: 0-100 relevance percentage
- **Category Breakdown**: Skills, Experience, Education scores
- **Gap Analysis**: Missing requirements and recommendations
- **Verdict**: High (80-100) / Medium (50-79) / Low (0-49) suitability

## Use Cases

### For Recruiters
- **Mass Screening**: Process hundreds of resumes quickly
- **Consistent Evaluation**: Standardized scoring across all candidates
- **Priority Ranking**: Focus on high-relevance candidates first
- **Gap Identification**: Understand market skill gaps

### For Candidates
- **Resume Optimization**: Get feedback on resume relevance
- **Skill Gap Analysis**: Identify areas for improvement
- **Application Targeting**: Apply to more relevant positions

### For Organizations
- **Hiring Efficiency**: Reduce time-to-hire significantly
- **Quality Improvement**: Better candidate-job matching
- **Data-Driven Decisions**: Analytics on hiring patterns

## Sample Output

```json
{
  "candidate_name": "John Doe",
  "overall_score": 85,
  "verdict": "High Suitability",
  "category_scores": {
    "technical_skills": 90,
    "experience": 80,
    "education": 85,
    "projects": 88
  },
  "matched_skills": ["Python", "Machine Learning", "SQL", "AWS"],
  "missing_skills": ["Docker", "Kubernetes"],
  "recommendations": [
    "Consider gaining experience with containerization technologies",
    "Cloud architecture certification would strengthen profile"
  ]
}
```

## Future Enhancements

- **Video Resume Analysis**: Process video introductions
- **Real-time Chat**: Interactive candidate Q&A system
- **Integration APIs**: Connect with ATS systems
- **Advanced Analytics**: Hiring trend analysis and predictions
- **Multi-language Support**: Process resumes in different languages

## Features Overview

### Core Functionality
- **Multi-format Support**: Processes PDF and DOCX resume files
- **Intelligent Parsing**: Extracts skills, experience, education, and projects
- **Smart Matching**: Compares candidates against job requirements
- **Detailed Scoring**: 0-100 relevance score with category breakdown
- **Gap Analysis**: Identifies missing skills and experience
- **Actionable Feedback**: Provides improvement recommendations

### Analysis Modes
- **Traditional Analysis**: Keyword and pattern-based matching (always available)
- **Enhanced Semantic**: Advanced language processing (requires API key)
- **Vector Similarity**: Embedding-based contextual matching (optional)

### User Interface
- **Single Analysis**: Process one resume at a time with detailed results
- **Batch Processing**: Handle multiple resumes against one job description
- **Export Results**: Download analysis results in JSON format

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB+ recommended for large document processing
- **Storage**: 1GB free space for dependencies and temporary files
- **Network**: Internet connection for advanced features

---

**Built to streamline recruitment and improve hiring efficiency**
