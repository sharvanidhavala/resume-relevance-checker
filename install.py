#!/usr/bin/env python3
"""
Resume Relevance Checker - Installation Script
Simple installation script that handles dependencies gracefully.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description, optional=False):
    """Run a command and handle errors gracefully"""
    print(f"\n[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"[WARNING] {description} failed (optional): {e}")
            return False
        else:
            print(f"[ERROR] {description} failed: {e}")
            return False

def install_basic_requirements():
    """Install core requirements that are essential"""
    basic_packages = [
        "streamlit>=1.28.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "PyMuPDF>=1.23.0",
        "pdfplumber>=0.9.0",
        "python-docx>=0.8.11",
        "docx2txt>=0.8",
        "spacy>=3.7.0",
        "nltk>=3.8.0",
        "openai>=1.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}", optional=True):
            print(f"[WARNING] Failed to install {package}, continuing...")

def install_advanced_requirements():
    """Install advanced/optional requirements"""
    advanced_packages = [
        "langchain>=0.1.20",
        "langchain-openai>=0.1.8", 
        "langchain-core>=0.1.52",
        "langgraph>=0.0.69",
        "langsmith>=0.1.85",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.15"
    ]
    
    print("\n[INFO] Installing advanced features (LangChain, Vector Store)...")
    print("[INFO] These are optional - the system will work without them")
    
    for package in advanced_packages:
        run_command(f"pip install {package}", f"Installing {package}", optional=True)

def setup_spacy_model():
    """Download spaCy model"""
    print("\n[INFO] Setting up spaCy language model...")
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model", optional=True):
        print("[WARNING] spaCy model download failed - some NLP features may be limited")

def create_directories():
    """Create necessary directories"""
    dirs = ['uploads', 'logs', 'temp', 'data']
    print(f"\n[INFO] Creating directories: {', '.join(dirs)}")
    
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"[INFO] Created: {directory}/")

def setup_env_file():
    """Setup environment file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n[INFO] Creating .env file from template...")
        try:
            env_content = env_example.read_text()
            env_file.write_text(env_content)
            print("[SUCCESS] .env file created")
            print("[IMPORTANT] Please update .env with your actual API key!")
        except Exception as e:
            print(f"[ERROR] Failed to create .env: {e}")

def test_installation():
    """Test basic functionality"""
    print("\n[INFO] Testing basic functionality...")
    
    try:
        # Test core imports
        import streamlit
        print("[SUCCESS] Streamlit available")
        
        import pandas
        print("[SUCCESS] Pandas available")
        
        from src.resume_parser import ResumeParser
        print("[SUCCESS] Resume parser available")
        
        from src.job_analyzer import JobAnalyzer
        print("[SUCCESS] Job analyzer available")
        
        from src.relevance_scorer import RelevanceScorer
        print("[SUCCESS] Relevance scorer available")
        
        from src.workflow_manager import WorkflowManager
        print("[SUCCESS] Workflow manager available")
        
    except ImportError as e:
        print(f"[ERROR] Import test failed: {e}")
        return False
    
    return True

def main():
    """Main installation function"""
    print("=" * 60)
    print("Resume Relevance Checker - Installation")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        print(f"[ERROR] Current version: {sys.version}")
        sys.exit(1)
    
    print(f"[SUCCESS] Python version: {sys.version.split()[0]}")
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_env_file()
    
    # Install requirements
    print("\n" + "=" * 40)
    print("INSTALLING DEPENDENCIES")
    print("=" * 40)
    
    install_basic_requirements()
    
    # Ask about advanced features
    response = input("\nInstall advanced features (LangChain, Vector Store)? (y/n): ").lower().strip()
    if response.startswith('y'):
        install_advanced_requirements()
    else:
        print("[INFO] Skipping advanced features - basic functionality will be available")
    
    # Setup spaCy
    setup_spacy_model()
    
    # Test installation
    print("\n" + "=" * 40)
    print("TESTING INSTALLATION")
    print("=" * 40)
    
    if test_installation():
        print("\n" + "=" * 60)
        print("INSTALLATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo run the application:")
        print("1. Update .env file with your API key (optional)")
        print("2. Run: streamlit run src/app.py")
        print("3. Open browser to: http://localhost:8501")
        print("\nThe system will work with traditional analysis even without API keys!")
        
    else:
        print("\n[WARNING] Installation completed with issues")
        print("Some features may not be available")

if __name__ == "__main__":
    main()