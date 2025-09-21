#!/usr/bin/env python3
"""
Setup Script for Automated Resume Relevance Check System
Installation and configuration script for the project
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed during {description}: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("[INFO] Setting up virtual environment...")
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    print("[SUCCESS] Virtual environment created successfully")
    print("\nTo activate the virtual environment:")
    print("   Windows: .\\venv\\Scripts\\activate")
    print("   Linux/Mac: source venv/bin/activate")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n[INFO] Installing project dependencies...")
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("[WARNING] Not in a virtual environment. Consider activating venv first.")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Download spaCy model
    print("\n[INFO] Downloading spaCy language model...")
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("[WARNING] spaCy model download failed. You can download it manually later.")
    
    return True

def create_environment_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n[INFO] Creating environment configuration file...")
        try:
            env_file.write_text(env_example.read_text())
            print("[SUCCESS] .env file created successfully")
    print("[NOTE] Please update .env file with your actual values, especially API_KEY for semantic analysis")
        except Exception as e:
            print(f"[ERROR] Failed to create .env file: {e}")

def setup_directories():
    """Create necessary directories"""
    directories = ['uploads', 'logs', 'sample_data']
    
    print("\n[INFO] Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[INFO] Created directory: {directory}")
    print("[SUCCESS] Project directories created")

def run_tests():
    """Run basic tests to verify setup"""
    print("\n[INFO] Running project tests...")
    if run_command("python -m pytest tests/ -v", "Running tests"):
        print("[SUCCESS] All tests passed!")
        return True
    else:
        print("[WARNING] Some tests failed. Please check the output above.")
        return False

def validate_python_version():
    """Check if Python version meets requirements"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"[ERROR] Current version: {sys.version}")
        sys.exit(1)
    
    print(f"[SUCCESS] Python version validated: {sys.version.split()[0]}")

def print_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nNext steps to run the application:")
    print("1. Update .env file with your API key for enhanced matching")
    print("2. Run: streamlit run src/app.py")
    print("3. Open browser to: http://localhost:8501")
    print("\nProject structure:")
    print("- src/app.py: Main Streamlit application")
    print("- src/resume_parser.py: Resume parsing functionality")
    print("- src/job_analyzer.py: Job description analysis")
    print("- src/relevance_scorer.py: Scoring algorithms")
    print("- src/semantic_matcher.py: Advanced semantic matching")
    print("- tests/: Unit tests for validation")
    print("\nFor detailed documentation, refer to README.md")

def main():
    """Main setup function"""
    print("Automated Resume Relevance Check System - Setup")
    print("="*50)
    print("Innomatics Research Labs - Project Setup")
    
    # Validate environment
    validate_python_version()
    
    # Setup project structure
    setup_directories()
    create_environment_file()
    
    # Handle virtual environment setup
    if "--skip-venv" not in sys.argv:
        response = input("\nCreate virtual environment? (y/n): ").lower().strip()
        if response.startswith('y'):
            if create_virtual_environment():
                print("\n[NOTE] Virtual environment created.")
                print("Please activate it and run: python setup.py --install-deps")
                return
    
    # Install dependencies
    if "--install-deps" in sys.argv:
        install_dependencies()
    else:
        response = input("\nInstall project dependencies? (y/n): ").lower().strip()
        if response.startswith('y'):
            install_dependencies()
    
    # Run tests
    if "--run-tests" in sys.argv:
        run_tests()
    else:
        response = input("\nRun validation tests? (y/n): ").lower().strip()
        if response.startswith('y'):
            run_tests()
    
    print_next_steps()

if __name__ == "__main__":
    main()