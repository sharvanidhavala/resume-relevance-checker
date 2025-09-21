#!/usr/bin/env python3
"""
Semantic Matcher - Basic implementation for resume relevance analysis
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

class SemanticMatcher:
    """Basic SemanticMatcher class for resume-job compatibility analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the semantic matcher
        
        Args:
            api_key: Optional API key for advanced features (not used in basic mode)
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
    def analyze_compatibility(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compatibility between resume and job description
        
        Args:
            resume_data: Parsed resume information
            job_data: Parsed job description information
            
        Returns:
            Dict containing compatibility analysis results
        """
        try:
            # Basic compatibility scoring
            skills_score = self._analyze_skills_compatibility(resume_data, job_data)
            experience_score = self._analyze_experience_compatibility(resume_data, job_data)
            education_score = self._analyze_education_compatibility(resume_data, job_data)
            
            # Calculate overall score
            overall_score = (skills_score * 0.5 + experience_score * 0.3 + education_score * 0.2)
            
            return {
                'overall_score': min(100, max(0, overall_score)),
                'semantic_insights': {
                    'relevance_explanation': f'Basic compatibility analysis completed with {overall_score:.1f}% match',
                    'key_alignments': self._find_key_alignments(resume_data, job_data),
                    'potential_concerns': self._identify_concerns(resume_data, job_data)
                },
                'skill_insights': {
                    'transferable_skills': self._find_transferable_skills(resume_data, job_data),
                    'skill_gaps': self._identify_skill_gaps(resume_data, job_data),
                    'recommendations': []
                },
                'experience_insights': {
                    'relevant_experience': self._find_relevant_experience(resume_data, job_data),
                    'experience_gaps': self._identify_experience_gaps(resume_data, job_data)
                },
                'recommendations': self._generate_recommendations(resume_data, job_data),
                'confidence_score': 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Compatibility analysis failed: {str(e)}")
            return self._get_fallback_analysis()
    
    def _analyze_skills_compatibility(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
        """Analyze skills compatibility between resume and job"""
        try:
            # Extract skills from resume
            resume_skills = set()
            for category, skills in resume_data.get('skills', {}).items():
                resume_skills.update([skill.lower().strip() for skill in skills])
            
            # Extract required skills from job
            required_skills = set()
            required_skills.update([skill.lower().strip() for skill in job_data.get('required_skills', [])])
            required_skills.update([skill.lower().strip() for skill in job_data.get('preferred_skills', [])])
            
            if not required_skills:
                return 70.0  # Default score if no requirements specified
            
            # Calculate overlap
            matching_skills = resume_skills.intersection(required_skills)
            score = (len(matching_skills) / len(required_skills)) * 100
            
            return min(100, score)
            
        except Exception as e:
            self.logger.error(f"Skills analysis failed: {str(e)}")
            return 50.0
    
    def _analyze_experience_compatibility(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
        """Analyze experience compatibility"""
        try:
            experiences = resume_data.get('experience', [])
            if not experiences:
                return 40.0
            
            # Simple scoring based on number of relevant experiences
            score = min(len(experiences) * 20, 100)
            return score
            
        except Exception as e:
            self.logger.error(f"Experience analysis failed: {str(e)}")
            return 50.0
    
    def _analyze_education_compatibility(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
        """Analyze education compatibility"""
        try:
            education = resume_data.get('education', [])
            if not education:
                return 60.0
            
            # Simple scoring based on education level
            score = 75.0 if education else 50.0
            return score
            
        except Exception as e:
            self.logger.error(f"Education analysis failed: {str(e)}")
            return 50.0
    
    def _find_key_alignments(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Find key alignments between resume and job"""
        alignments = []
        
        # Check for skill alignments
        resume_skills = set()
        for category, skills in resume_data.get('skills', {}).items():
            resume_skills.update([skill.lower().strip() for skill in skills])
        
        required_skills = set([skill.lower().strip() for skill in job_data.get('required_skills', [])])
        matching_skills = resume_skills.intersection(required_skills)
        
        if matching_skills:
            alignments.append(f"Matching skills: {', '.join(list(matching_skills)[:5])}")
        
        return alignments[:3]
    
    def _identify_concerns(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Identify potential concerns"""
        concerns = []
        
        # Check for missing skills
        resume_skills = set()
        for category, skills in resume_data.get('skills', {}).items():
            resume_skills.update([skill.lower().strip() for skill in skills])
        
        required_skills = set([skill.lower().strip() for skill in job_data.get('required_skills', [])])
        missing_skills = required_skills - resume_skills
        
        if missing_skills:
            concerns.append(f"Missing required skills: {', '.join(list(missing_skills)[:3])}")
        
        return concerns[:3]
    
    def _find_transferable_skills(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Find transferable skills"""
        resume_skills = []
        for category, skills in resume_data.get('skills', {}).items():
            resume_skills.extend(skills)
        
        return resume_skills[:5]
    
    def _identify_skill_gaps(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Identify skill gaps"""
        resume_skills = set()
        for category, skills in resume_data.get('skills', {}).items():
            resume_skills.update([skill.lower().strip() for skill in skills])
        
        required_skills = set([skill.lower().strip() for skill in job_data.get('required_skills', [])])
        gaps = required_skills - resume_skills
        
        return list(gaps)[:5]
    
    def _find_relevant_experience(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Find relevant experience"""
        experiences = resume_data.get('experience', [])
        relevant = [exp.get('title', 'Experience') for exp in experiences[:3]]
        return relevant
    
    def _identify_experience_gaps(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Identify experience gaps"""
        return ["Consider gaining more industry-specific experience"]
    
    def _generate_recommendations(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = [
            "Review job requirements in detail and tailor resume accordingly",
            "Highlight relevant experience and achievements",
            "Consider developing missing technical skills",
            "Quantify achievements with specific metrics where possible"
        ]
        return recommendations[:4]
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis when processing fails"""
        return {
            'overall_score': 60,
            'semantic_insights': {
                'relevance_explanation': 'Basic analysis completed with limited insights',
                'key_alignments': [],
                'potential_concerns': []
            },
            'skill_insights': {
                'transferable_skills': [],
                'skill_gaps': [],
                'recommendations': []
            },
            'experience_insights': {
                'relevant_experience': [],
                'experience_gaps': []
            },
            'recommendations': [
                'Consider reviewing job requirements in detail',
                'Focus on highlighting relevant experience',
                'Develop any missing technical skills'
            ],
            'confidence_score': 0.5
        }