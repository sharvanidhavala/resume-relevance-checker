"""
Resume Parser Module
Extracts and parses text from resume files (PDF, DOCX) and structures the data.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# PDF processing
try:
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError:
    fitz = None
    pdfplumber = None

# DOCX processing
try:
    import docx2txt
    from docx import Document
except ImportError:
    docx2txt = None
    Document = None

# Text processing
import nltk
import spacy
from datetime import datetime
import json

class ResumeParser:
    """Extract and parse resume content from various file formats"""
    
    def __init__(self):
        """Initialize the resume parser with NLP models"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Load spaCy model (fallback to smaller model if large not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None
        
        # Skill databases and patterns
        self.skill_patterns = self._load_skill_patterns()
        self.section_patterns = self._load_section_patterns()
    
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a resume file and extract structured data
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary containing parsed resume data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text based on file type
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            text = self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Parse the extracted text
        parsed_data = self._parse_text(text)
        parsed_data['raw_text'] = text
        parsed_data['file_path'] = file_path
        parsed_data['parsed_at'] = datetime.now().isoformat()
        
        return parsed_data
    
    def parse_text(self, text: str) -> Dict[str, Any]:
        """Parse text directly without file processing"""
        parsed_data = self._parse_text(text)
        parsed_data['raw_text'] = text
        parsed_data['file_path'] = "direct_text"
        parsed_data['parsed_at'] = datetime.now().isoformat()
        return parsed_data
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try PyMuPDF first
        if fitz:
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                if text.strip():
                    return text
            except Exception as e:
                logging.warning(f"PyMuPDF failed: {e}")
        
        # Fallback to pdfplumber
        if pdfplumber:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                logging.warning(f"pdfplumber failed: {e}")
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF. Install PyMuPDF or pdfplumber.")
        
        return text
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        
        # Try docx2txt first (simpler)
        if docx2txt:
            try:
                text = docx2txt.process(file_path)
                if text.strip():
                    return text
            except Exception as e:
                logging.warning(f"docx2txt failed: {e}")
        
        # Fallback to python-docx
        if Document:
            try:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                logging.warning(f"python-docx failed: {e}")
        
        if not text.strip():
            raise ValueError("Could not extract text from DOCX. Install docx2txt or python-docx.")
        
        return text
    
    def _parse_text(self, text: str) -> Dict[str, Any]:
        """Parse extracted text and structure the data"""
        
        # Basic information extraction
        parsed_data = {
            'personal_info': self._extract_personal_info(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'projects': self._extract_projects(text),
            'certifications': self._extract_certifications(text),
            'summary': self._extract_summary(text),
            'contact_info': self._extract_contact_info(text),
            'languages': self._extract_languages(text),
            'sections': self._identify_sections(text)
        }
        
        # Calculate derived metrics
        parsed_data['metrics'] = self._calculate_metrics(parsed_data)
        
        return parsed_data
    
    def _extract_personal_info(self, text: str) -> Dict[str, str]:
        """Extract personal information"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        name = ""
        title = ""
        
        # Simple name extraction (first non-empty line that looks like a name)
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
                if '@' not in line and 'www.' not in line and len(line) > 3:
                    name = line
                    break
        
        # Look for job title/summary
        for line in lines[1:5]:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 100:
                if not any(char.isdigit() for char in line) and '@' not in line:
                    title = line
                    break
        
        return {
            'name': name,
            'title': title
        }
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': '',
            'location': ''
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone pattern
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # LinkedIn
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=)([A-Za-z0-9-_]+)'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_pattern = r'github\.com/([A-Za-z0-9-_]+)'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()
        
        return contact_info
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills"""
        skills = {
            'technical': [],
            'programming_languages': [],
            'frameworks': [],
            'tools': [],
            'soft_skills': [],
            'databases': [],
            'cloud_platforms': []
        }
        
        text_lower = text.lower()
        
        # Technical skills patterns
        for skill_type, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
                    if pattern not in skills[skill_type]:
                        skills[skill_type].append(pattern)
        
        return skills
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience"""
        experiences = []
        
        # Look for experience section
        experience_patterns = [
            r'(?:professional\s+)?experience',
            r'work\s+experience',
            r'employment\s+history',
            r'career\s+history'
        ]
        
        # Simple experience extraction (can be enhanced)
        lines = text.split('\n')
        in_experience_section = False
        current_experience = {}
        
        for line in lines:
            line = line.strip()
            
            # Check if we're entering experience section
            if any(re.search(pattern, line.lower()) for pattern in experience_patterns):
                in_experience_section = True
                continue
            
            # Check if we're leaving experience section
            if in_experience_section and any(keyword in line.lower() for keyword in ['education', 'skills', 'projects', 'certifications']):
                in_experience_section = False
            
            # Extract experience details (basic implementation)
            if in_experience_section and line:
                # This is a simplified extraction - can be much more sophisticated
                if re.search(r'\b\d{4}\b', line):  # Contains year
                    experiences.append({
                        'title': line,
                        'company': '',
                        'duration': '',
                        'description': ''
                    })
        
        return experiences
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract educational background"""
        education = []
        
        # Degree patterns
        degree_patterns = [
            r'bachelor.*(?:computer science|engineering|information technology)',
            r'master.*(?:computer science|engineering|information technology)',
            r'phd.*(?:computer science|engineering|information technology)',
            r'b\.?tech',
            r'm\.?tech',
            r'b\.?s\.?',
            r'm\.?s\.?',
            r'mba'
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in degree_patterns:
                if re.search(pattern, line.lower()):
                    education.append({
                        'degree': line,
                        'institution': '',
                        'year': '',
                        'gpa': ''
                    })
        
        return education
    
    def _extract_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract project information"""
        projects = []
        
        # Look for projects section
        project_section_found = False
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if 'project' in line.lower() and len(line) < 50:
                project_section_found = True
                continue
            
            if project_section_found and line and not line.startswith(' '):
                projects.append({
                    'name': line,
                    'description': '',
                    'technologies': '',
                    'link': ''
                })
        
        return projects
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        certifications = []
        
        cert_patterns = [
            r'aws\s+certified',
            r'google\s+cloud',
            r'microsoft\s+azure',
            r'oracle\s+certified',
            r'cisco\s+certified',
            r'comptia',
            r'pmp',
            r'scrum\s+master'
        ]
        
        for pattern in cert_patterns:
            if re.search(pattern, text.lower()):
                certifications.append(pattern.title())
        
        return certifications
    
    def _extract_summary(self, text: str) -> str:
        """Extract professional summary"""
        lines = text.split('\n')
        
        summary_keywords = ['summary', 'objective', 'profile', 'about']
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in summary_keywords):
                # Get next few lines as summary
                summary_lines = lines[i+1:i+5]
                summary = ' '.join([l.strip() for l in summary_lines if l.strip()])
                if len(summary) > 20:
                    return summary
        
        # Fallback: first paragraph after personal info
        meaningful_lines = [l.strip() for l in lines[3:8] if l.strip() and len(l) > 20]
        if meaningful_lines:
            return meaningful_lines[0]
        
        return ""
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract programming and spoken languages"""
        languages = []
        
        # Common programming languages
        prog_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin']
        
        for lang in prog_languages:
            if re.search(r'\b' + lang + r'\b', text.lower()):
                languages.append(lang.title())
        
        return languages
    
    def _identify_sections(self, text: str) -> Dict[str, bool]:
        """Identify which sections are present in the resume"""
        sections = {
            'experience': False,
            'education': False,
            'skills': False,
            'projects': False,
            'certifications': False,
            'summary': False,
            'contact': False
        }
        
        text_lower = text.lower()
        
        section_keywords = {
            'experience': ['experience', 'employment', 'work history'],
            'education': ['education', 'academic', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates'],
            'summary': ['summary', 'objective', 'profile'],
            'contact': ['contact', 'email', 'phone']
        }
        
        for section, keywords in section_keywords.items():
            sections[section] = any(keyword in text_lower for keyword in keywords)
        
        return sections
    
    def _calculate_metrics(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resume metrics"""
        metrics = {
            'total_skills': len([skill for category in parsed_data['skills'].values() for skill in category]),
            'experience_count': len(parsed_data['experience']),
            'education_count': len(parsed_data['education']),
            'project_count': len(parsed_data['projects']),
            'certification_count': len(parsed_data['certifications']),
            'sections_present': sum(parsed_data['sections'].values()),
            'completeness_score': 0
        }
        
        # Calculate completeness score
        max_sections = len(parsed_data['sections'])
        metrics['completeness_score'] = (metrics['sections_present'] / max_sections) * 100
        
        return metrics
    
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load skill patterns for matching"""
        return {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 
                'Swift', 'Kotlin', 'TypeScript', 'PHP', 'Scala', 'R', 'MATLAB'
            ],
            'frameworks': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'Django', 
                'Flask', 'Spring', 'Laravel', 'Rails', 'ASP.NET', 'jQuery'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle', 
                'SQL Server', 'Cassandra', 'DynamoDB', 'Elasticsearch'
            ],
            'cloud_platforms': [
                'AWS', 'Azure', 'Google Cloud', 'GCP', 'Heroku', 'Docker', 
                'Kubernetes', 'Jenkins', 'CI/CD'
            ],
            'tools': [
                'Git', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Slack', 
                'VS Code', 'IntelliJ', 'Eclipse', 'Postman'
            ],
            'technical': [
                'Machine Learning', 'Data Science', 'Artificial Intelligence', 
                'Web Development', 'Mobile Development', 'DevOps', 'Agile', 
                'REST API', 'GraphQL', 'Microservices'
            ]
        }
    
    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """Load section header patterns"""
        return {
            'experience': ['experience', 'work experience', 'professional experience', 'employment'],
            'education': ['education', 'academic background', 'qualifications'],
            'skills': ['skills', 'technical skills', 'core competencies'],
            'projects': ['projects', 'personal projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }