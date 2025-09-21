#!/usr/bin/env python3
"""
Run Script for Automated Resume Relevance Check System
Simple script to start the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met"""
    # Check if we're in the right directory
    if not Path("src/app.py").exists():
        print("[ERROR] Cannot find src/app.py. Please run from project root directory.")
        return False
    
    # Check if virtual environment is activated (optional but recommended)
    if sys.prefix == sys.base_prefix:
        print("[WARNING] Virtual environment not detected. Consider activating venv for better isolation.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if not response.startswith('y'):
            return False
    
    return True

def check_environment():
    """Check environment configuration"""
    env_file = Path(".env")
    if not env_file.exists():
        print("[WARNING] .env file not found. Some features may not work without proper configuration.")
        print("[INFO] Copy .env.example to .env and update with your settings.")
    
    return True

def start_application():
    """Start the Streamlit application"""
    print("[INFO] Starting Automated Resume Relevance Check System...")
    print("[INFO] Opening web interface at http://localhost:8501")
    print("[INFO] Press Ctrl+C to stop the application")
    
    try:
        # Start streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app.py",
            "--server.address", "localhost",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start application: {e}")
        print("[INFO] Make sure Streamlit is installed: pip install streamlit")
        return False
    except FileNotFoundError:
        print("[ERROR] Streamlit not found. Please install requirements first:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function"""
    print("Automated Resume Relevance Check System")
    print("=" * 50)
    
    # Pre-flight checks
    if not check_requirements():
        sys.exit(1)
    
    check_environment()
    
    # Start the application
    if not start_application():
        sys.exit(1)

if __name__ == "__main__":
    main()