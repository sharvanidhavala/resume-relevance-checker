"""
Tests for Resume Parser
Basic test cases for the resume parser functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from resume_parser import ResumeParser

class TestResumeParser:
    """Test cases for ResumeParser class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.parser = ResumeParser()
    
    def test_init(self):
        """Test parser initialization"""
        assert self.parser is not None
        assert hasattr(self.parser, 'skill_patterns')
        assert hasattr(self.parser, 'section_patterns')
    
    def test_extract_personal_info(self):
        """Test personal information extraction"""
        sample_text = """
John Doe
Software Engineer

Email: john.doe@example.com
Phone: (555) 123-4567
        """.strip()
        
        personal_info = self.parser._extract_personal_info(sample_text)
        
        assert personal_info['name'] == 'John Doe'
        assert 'Software Engineer' in personal_info['title']
    
    def test_extract_contact_info(self):
        """Test contact information extraction"""
        sample_text = """
John Doe
Email: john.doe@example.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe
        """.strip()
        
        contact_info = self.parser._extract_contact_info(sample_text)
        
        assert contact_info['email'] == 'john.doe@example.com'
        assert '555' in contact_info['phone']
        assert 'linkedin.com/in/johndoe' in contact_info['linkedin']
        assert 'github.com/johndoe' in contact_info['github']
    
    def test_extract_skills(self):
        """Test skills extraction"""
        sample_text = """
Skills:
- Python
- JavaScript
- React
- Django
- AWS
- Docker
        """.strip()
        
        skills = self.parser._extract_skills(sample_text)
        
        # Check if some expected skills are found
        all_skills = []
        for category, skill_list in skills.items():
            all_skills.extend(skill_list)
        
        assert 'Python' in all_skills
        assert 'JavaScript' in all_skills
        assert 'React' in all_skills
    
    def test_extract_education(self):
        """Test education extraction"""
        sample_text = """
Education:
Bachelor of Science in Computer Science
University of Technology, 2019
        """.strip()
        
        education = self.parser._extract_education(sample_text)
        
        # Should find at least one education entry
        assert len(education) > 0
    
    def test_identify_sections(self):
        """Test section identification"""
        sample_text = """
John Doe
Software Engineer
Email: john@example.com

Skills:
Python, JavaScript

Experience:
Software Developer at TechCorp

Education:
Bachelor of Science

Projects:
E-commerce Platform
        """.strip()
        
        sections = self.parser._identify_sections(sample_text)
        
        assert sections['contact'] is True
        assert sections['skills'] is True
        assert sections['experience'] is True
        assert sections['education'] is True
        assert sections['projects'] is True
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        parsed_data = {
            'skills': {
                'programming_languages': ['Python', 'Java'],
                'frameworks': ['React'],
                'databases': ['MySQL']
            },
            'experience': [{'title': 'Software Developer'}],
            'education': [{'degree': 'BS Computer Science'}],
            'projects': [{'name': 'Web App'}],
            'certifications': ['AWS Certified'],
            'sections': {
                'skills': True,
                'experience': True,
                'education': True,
                'projects': True,
                'certifications': True,
                'summary': False,
                'contact': True
            }
        }
        
        metrics = self.parser._calculate_metrics(parsed_data)
        
        assert metrics['total_skills'] == 4
        assert metrics['experience_count'] == 1
        assert metrics['education_count'] == 1
        assert metrics['project_count'] == 1
        assert metrics['certification_count'] == 1
        assert metrics['sections_present'] == 6
        assert 0 <= metrics['completeness_score'] <= 100
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_parse_resume_file_not_found(self, mock_open):
        """Test handling of missing file"""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_resume('nonexistent_file.pdf')
    
    def test_parse_resume_unsupported_format(self):
        """Test handling of unsupported file format"""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp_file:
            tmp_file.write(b'test content')
            tmp_file.flush()
            
            try:
                with pytest.raises(ValueError, match="Unsupported file format"):
                    self.parser.parse_resume(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

if __name__ == '__main__':
    pytest.main([__file__])