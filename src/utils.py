"""
Utility Functions
Common helper functions for the resume relevance checker.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Any, Dict
import logging
from datetime import datetime
import hashlib

def save_uploaded_file(uploaded_file: Any, upload_dir: str) -> str:
    """
    Save uploaded file to specified directory
    
    Args:
        uploaded_file: Streamlit uploaded file object
        upload_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    safe_filename = f"{timestamp}_{hash_string(uploaded_file.name)}{file_extension}"
    
    file_path = os.path.join(upload_dir, safe_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_file_type(filename: str) -> str:
    """
    Get file type based on extension
    
    Args:
        filename: Name of the file
        
    Returns:
        File type as string
    """
    extension = Path(filename).suffix.lower()
    
    file_types = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.txt': 'txt'
    }
    
    return file_types.get(extension, 'unknown')

def hash_string(text: str, length: int = 8) -> str:
    """
    Generate a hash of the given text
    
    Args:
        text: Text to hash
        length: Length of the hash to return
        
    Returns:
        Hashed string
    """
    return hashlib.md5(text.encode()).hexdigest()[:length]

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing special characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove special characters and replace with underscore
    import re
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    cleaned = re.sub(r'_+', '_', cleaned)  # Replace multiple underscores with single
    return cleaned.strip('_')

def format_score(score: float) -> str:
    """
    Format score for display
    
    Args:
        score: Numerical score
        
    Returns:
        Formatted score string
    """
    return f"{score:.1f}%"

def get_score_color(score: float) -> str:
    """
    Get color code based on score
    
    Args:
        score: Numerical score (0-100)
        
    Returns:
        Color code or name
    """
    if score >= 80:
        return "#28a745"  # Green
    elif score >= 60:
        return "#ffc107"  # Yellow/Orange
    else:
        return "#dc3545"  # Red

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def format_list_for_display(items: list, max_items: int = 10, separator: str = ", ") -> str:
    """
    Format list of items for display
    
    Args:
        items: List of items
        max_items: Maximum number of items to display
        separator: Separator between items
        
    Returns:
        Formatted string
    """
    if not items:
        return "None"
    
    display_items = items[:max_items]
    result = separator.join(str(item) for item in display_items)
    
    if len(items) > max_items:
        remaining = len(items) - max_items
        result += f" and {remaining} more"
    
    return result

def validate_file_size(file_path: str, max_size_mb: int = 10) -> bool:
    """
    Validate file size
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum size in MB
        
    Returns:
        True if file is within size limit
    """
    if not os.path.exists(file_path):
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb <= max_size_mb

def cleanup_temp_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files cleaned up
    """
    if not os.path.exists(directory):
        return 0
    
    cleaned_count = 0
    current_time = datetime.now()
    max_age_seconds = max_age_hours * 3600
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time.timestamp() - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    cleaned_count += 1
                    logging.info(f"Cleaned up old file: {filename}")
    
    except Exception as e:
        logging.error(f"Error cleaning up files: {str(e)}")
    
    return cleaned_count

def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def create_sample_data(sample_dir: str) -> None:
    """
    Create sample data files for testing
    
    Args:
        sample_dir: Directory to create sample files
    """
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample job description
    job_description = """
Software Engineer Position

We are looking for a talented Software Engineer to join our team.

Required Skills:
- Python
- JavaScript
- React
- Node.js
- SQL
- Git

Preferred Skills:
- AWS
- Docker
- MongoDB

Requirements:
- Bachelor's degree in Computer Science or related field
- 3+ years of experience in software development
- Experience with web application development
- Strong problem-solving skills

Responsibilities:
- Develop and maintain web applications
- Collaborate with cross-functional teams
- Write clean, efficient code
- Participate in code reviews
"""
    
    job_file_path = os.path.join(sample_dir, "sample_job_description.txt")
    with open(job_file_path, 'w') as f:
        f.write(job_description)
    
    # Sample resume text (would normally be extracted from PDF/DOCX)
    resume_text = """
John Doe
Software Developer

Email: john.doe@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe

Professional Summary:
Experienced software developer with 4 years of experience in web application development.
Proficient in Python, JavaScript, and modern web technologies.

Technical Skills:
- Programming Languages: Python, JavaScript, Java
- Frameworks: React, Django, Flask
- Databases: PostgreSQL, MySQL
- Tools: Git, Docker, AWS
- Web Technologies: HTML, CSS, REST APIs

Work Experience:
Software Developer at TechCorp (2020-2024)
- Developed web applications using Python and React
- Implemented REST APIs and database designs
- Collaborated with team of 5 developers
- Improved application performance by 30%

Junior Developer at StartupXYZ (2019-2020)
- Built frontend components using React
- Worked with backend APIs
- Participated in agile development process

Education:
Bachelor of Science in Computer Science
State University (2015-2019)
GPA: 3.7/4.0

Projects:
E-commerce Platform
- Built full-stack web application using Python/Django and React
- Implemented payment processing and user authentication
- Deployed on AWS with containerized architecture

Task Management App
- Created React-based frontend with Node.js backend
- Integrated with MongoDB database
- Implemented real-time updates using WebSockets
"""
    
    resume_file_path = os.path.join(sample_dir, "sample_resume.txt")
    with open(resume_file_path, 'w') as f:
        f.write(resume_text)
    
    logging.info(f"Sample data created in {sample_dir}")

def get_app_version() -> str:
    """
    Get application version
    
    Returns:
        Version string
    """
    return "1.0.0"

def format_timestamp(timestamp: str) -> str:
    """
    Format ISO timestamp for display
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Formatted timestamp
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except:
        return ""
    
    return hash_md5.hexdigest()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries recursively
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dictionaries(result[key], value)
        else:
            result[key] = value
    
    return result

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> list:
    """
    Extract keywords from text
    
    Args:
        text: Input text
        min_length: Minimum length of keywords
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    import re
    from collections import Counter
    
    # Simple keyword extraction (can be enhanced with NLP)
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'are', 'for', 'with', 'will', 'you', 'your', 'our', 
        'this', 'that', 'have', 'has', 'can', 'was', 'were', 'been', 'from',
        'they', 'them', 'their', 'there', 'where', 'when', 'what', 'who',
        'how', 'why', 'which', 'should', 'would', 'could', 'may', 'might'
    }
    
    # Filter out stop words and get most common
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    return [word for word, count in word_counts.most_common(max_keywords)]