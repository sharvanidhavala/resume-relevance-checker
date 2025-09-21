"""
Tests for Relevance Scorer
Basic test cases for the relevance scoring functionality.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from relevance_scorer import RelevanceScorer

class TestRelevanceScorer:
    """Test cases for RelevanceScorer class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.scorer = RelevanceScorer()
    
    def test_init(self):
        """Test scorer initialization"""
        assert self.scorer is not None
        assert hasattr(self.scorer, 'weights')
        assert hasattr(self.scorer, 'skill_importance')
        
        # Check weights sum to approximately 1.0
        total_weight = sum(self.scorer.weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_flatten_skills(self):
        """Test skills flattening functionality"""
        skills_dict = {
            'programming_languages': ['Python', 'Java'],
            'frameworks': ['React', 'Django'],
            'databases': ['MySQL']
        }
        
        flattened = self.scorer._flatten_skills(skills_dict)
        
        assert len(flattened) == 5
        assert 'Python' in flattened
        assert 'Java' in flattened
        assert 'React' in flattened
        assert 'Django' in flattened
        assert 'MySQL' in flattened
    
    def test_calculate_skills_match(self):
        """Test skills matching calculation"""
        resume_data = {
            'skills': {
                'programming_languages': ['Python', 'Java'],
                'frameworks': ['React'],
                'databases': ['MySQL']
            }
        }
        
        job_data = {
            'required_skills': ['Python', 'React', 'PostgreSQL'],
            'preferred_skills': ['Java', 'Docker']
        }
        
        result = self.scorer._calculate_skills_match(resume_data, job_data)
        
        assert 'score' in result
        assert 'matched' in result
        assert 'missing' in result
        assert 0 <= result['score'] <= 100
        
        # Check if Python and React are matched
        matched_lower = [skill.lower() for skill in result['matched']]
        assert 'python' in matched_lower
        assert 'react' in matched_lower
    
    def test_calculate_experience_match(self):
        """Test experience matching calculation"""
        resume_data = {
            'experience': [
                {'title': 'Software Developer'},
                {'title': 'Junior Developer'}
            ],
            'projects': [{'name': 'Web App'}]
        }
        
        job_data = {
            'experience_requirements': {
                'years_required': 2,
                'level': 'mid-level'
            },
            'seniority_level': 'mid-level'
        }
        
        result = self.scorer._calculate_experience_match(resume_data, job_data)
        
        assert 'score' in result
        assert 'resume_years' in result
        assert 'required_years' in result
        assert 'gap' in result
        assert 0 <= result['score'] <= 100
    
    def test_calculate_education_match(self):
        """Test education matching calculation"""
        resume_data = {
            'education': [
                {'degree': 'Bachelor of Science in Computer Science'}
            ]
        }
        
        job_data = {
            'education_requirements': {
                'degree_level': 'bachelor',
                'field_of_study': ['Computer Science'],
                'required': True
            }
        }
        
        result = self.scorer._calculate_education_match(resume_data, job_data)
        
        assert 'score' in result
        assert 'match' in result
        assert 0 <= result['score'] <= 100
    
    def test_calculate_projects_match(self):
        """Test projects matching calculation"""
        resume_data = {
            'projects': [
                {'name': 'E-commerce Platform', 'description': 'Built with Python and React'},
                {'name': 'Task Manager', 'description': 'Node.js and MongoDB'}
            ]
        }
        
        job_data = {
            'required_skills': ['Python', 'React', 'Node.js']
        }
        
        result = self.scorer._calculate_projects_match(resume_data, job_data)
        
        assert 'score' in result
        assert 'relevant_projects' in result
        assert 'project_skills' in result
        assert 0 <= result['score'] <= 100
    
    def test_calculate_certification_match(self):
        """Test certification matching calculation"""
        resume_data = {
            'certifications': ['AWS Certified', 'Scrum Master']
        }
        
        job_data = {
            'certifications': ['AWS Certified', 'Azure Certified']
        }
        
        result = self.scorer._calculate_certification_match(resume_data, job_data)
        
        assert 'score' in result
        assert 'matched' in result
        assert 'missing' in result
        assert 0 <= result['score'] <= 100
    
    def test_calculate_relevance_full(self):
        """Test full relevance calculation"""
        resume_data = {
            'skills': {
                'programming_languages': ['Python', 'Java'],
                'frameworks': ['React'],
                'databases': ['MySQL']
            },
            'experience': [{'title': 'Software Developer'}],
            'education': [{'degree': 'Bachelor of Science in Computer Science'}],
            'projects': [{'name': 'Web App', 'description': 'Python and React app'}],
            'certifications': ['AWS Certified']
        }
        
        job_data = {
            'required_skills': ['Python', 'React'],
            'preferred_skills': ['AWS'],
            'experience_requirements': {'years_required': 2, 'level': 'mid-level'},
            'education_requirements': {
                'degree_level': 'bachelor',
                'field_of_study': ['Computer Science'],
                'required': True
            },
            'certifications': ['AWS Certified'],
            'seniority_level': 'mid-level'
        }
        
        results = self.scorer.calculate_relevance(resume_data, job_data)
        
        assert 'overall_score' in results
        assert 'category_scores' in results
        assert 'matched_skills' in results
        assert 'missing_skills' in results
        assert 'recommendations' in results
        assert 'match_summary' in results
        
        assert 0 <= results['overall_score'] <= 100
        
        # Check category scores
        for category, score in results['category_scores'].items():
            assert 0 <= score <= 100
    
    def test_estimate_experience_years(self):
        """Test experience years estimation"""
        resume_data_with_exp = {
            'experience': [
                {'title': 'Senior Developer'},
                {'title': 'Junior Developer'},
                {'title': 'Intern'}
            ],
            'projects': [],
            'education': []
        }
        
        years = self.scorer._estimate_experience_years(resume_data_with_exp)
        assert years == 3
        
        resume_data_no_exp = {
            'experience': [],
            'projects': [{'name': 'Project 1'}, {'name': 'Project 2'}],
            'education': [{'degree': 'BS CS'}]
        }
        
        years = self.scorer._estimate_experience_years(resume_data_no_exp)
        assert years >= 1
    
    def test_estimate_seniority_level(self):
        """Test seniority level estimation"""
        resume_data = {
            'experience': [{'title': 'Senior Developer'}],
            'skills': {
                'programming_languages': ['Python', 'Java', 'C++'],
                'frameworks': ['React', 'Django']
            },
            'projects': [{'name': 'Project 1'}, {'name': 'Project 2'}]
        }
        
        level = self.scorer._estimate_seniority_level(resume_data)
        assert level in ['entry-level', 'mid-level', 'senior-level']
    
    def test_compare_seniority_levels(self):
        """Test seniority level comparison"""
        # Exact match
        multiplier = self.scorer._compare_seniority_levels('mid-level', 'mid-level')
        assert multiplier == 1.0
        
        # Higher than required
        multiplier = self.scorer._compare_seniority_levels('senior-level', 'mid-level')
        assert multiplier == 1.0
        
        # One level below
        multiplier = self.scorer._compare_seniority_levels('mid-level', 'senior-level')
        assert multiplier == 0.9
        
        # Two levels below
        multiplier = self.scorer._compare_seniority_levels('entry-level', 'senior-level')
        assert multiplier == 0.8

if __name__ == '__main__':
    pytest.main([__file__])