"""
Relevance Scorer Module
Calculates relevance scores between resumes and job descriptions.
"""

import math
import re
from typing import Dict, List, Set, Any, Tuple
from collections import Counter
import logging

class RelevanceScorer:
    """Calculate relevance scores between resumes and job descriptions"""
    
    def __init__(self):
        """Initialize the relevance scorer"""
        # Scoring weights for different categories
        self.weights = {
            'skills_match': 0.35,
            'experience_match': 0.25,
            'education_match': 0.15,
            'projects_match': 0.15,
            'certification_match': 0.10
        }
        
        # Skill importance multipliers
        self.skill_importance = {
            'programming_languages': 1.2,
            'frameworks': 1.1,
            'databases': 1.0,
            'cloud_platforms': 1.1,
            'devops_tools': 1.0,
            'data_science': 1.2,
            'technical': 1.0,
            'soft_skills': 0.8
        }
    
    def calculate_relevance(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall relevance score between resume and job description
        
        Args:
            resume_data: Parsed resume data
            job_data: Analyzed job description data
            
        Returns:
            Dictionary containing relevance scores and analysis
        """
        # Calculate individual category scores
        skills_score = self._calculate_skills_match(resume_data, job_data)
        experience_score = self._calculate_experience_match(resume_data, job_data)
        education_score = self._calculate_education_match(resume_data, job_data)
        projects_score = self._calculate_projects_match(resume_data, job_data)
        certification_score = self._calculate_certification_match(resume_data, job_data)
        
        # Calculate weighted overall score
        overall_score = (
            skills_score['score'] * self.weights['skills_match'] +
            experience_score['score'] * self.weights['experience_match'] +
            education_score['score'] * self.weights['education_match'] +
            projects_score['score'] * self.weights['projects_match'] +
            certification_score['score'] * self.weights['certification_match']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            skills_score, experience_score, education_score, 
            projects_score, certification_score, resume_data, job_data
        )
        
        # Compile final results
        results = {
            'overall_score': round(overall_score, 1),
            'category_scores': {
                'technical_skills': round(skills_score['score'], 1),
                'experience': round(experience_score['score'], 1),
                'education': round(education_score['score'], 1),
                'projects': round(projects_score['score'], 1),
                'certifications': round(certification_score['score'], 1)
            },
            'matched_skills': skills_score['matched'],
            'missing_skills': skills_score['missing'],
            'skill_gaps': skills_score['gaps'],
            'experience_gap': experience_score['gap'],
            'recommendations': recommendations,
            'detailed_analysis': {
                'skills_analysis': skills_score,
                'experience_analysis': experience_score,
                'education_analysis': education_score,
                'projects_analysis': projects_score,
                'certification_analysis': certification_score
            },
            'match_summary': self._create_match_summary(overall_score, skills_score, experience_score)
        }
        
        return results
    
    def _calculate_skills_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate skills matching score"""
        resume_skills = self._flatten_skills(resume_data['skills'])
        required_skills = set(skill.lower() for skill in job_data.get('required_skills', []))
        preferred_skills = set(skill.lower() for skill in job_data.get('preferred_skills', []))
        
        resume_skills_lower = set(skill.lower() for skill in resume_skills)
        
        # Calculate matches
        required_matches = resume_skills_lower.intersection(required_skills)
        preferred_matches = resume_skills_lower.intersection(preferred_skills)
        
        # Calculate missing skills
        missing_required = required_skills - resume_skills_lower
        missing_preferred = preferred_skills - resume_skills_lower
        
        # Calculate score
        total_required = len(required_skills) if required_skills else 1
        total_preferred = len(preferred_skills) if preferred_skills else 0
        
        required_score = (len(required_matches) / total_required) * 100
        preferred_bonus = min((len(preferred_matches) / max(total_preferred, 1)) * 20, 20)
        
        final_score = min(required_score + preferred_bonus, 100)
        
        # Identify skill gaps by category
        skill_gaps = self._identify_skill_gaps(resume_data['skills'], job_data)
        
        return {
            'score': final_score,
            'matched': list(required_matches) + list(preferred_matches),
            'missing': list(missing_required) + list(missing_preferred),
            'required_matches': len(required_matches),
            'total_required': total_required,
            'preferred_matches': len(preferred_matches),
            'total_preferred': total_preferred,
            'gaps': skill_gaps
        }
    
    def _calculate_experience_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate experience matching score"""
        # Extract years of experience from resume
        resume_exp_years = self._estimate_experience_years(resume_data)
        
        # Get job requirements
        required_years = job_data.get('experience_requirements', {}).get('years_required', 0)
        required_level = job_data.get('seniority_level', 'mid-level')
        
        # Calculate experience score
        if required_years == 0:
            exp_score = 85  # Default score when no specific requirement
        elif resume_exp_years >= required_years:
            # Bonus for exceeding requirements, but cap at 100
            exp_score = min(90 + (resume_exp_years - required_years) * 2, 100)
        else:
            # Penalty for lacking experience
            gap = required_years - resume_exp_years
            exp_score = max(60 - gap * 10, 20)
        
        # Adjust for seniority level match
        resume_level = self._estimate_seniority_level(resume_data)
        level_match = self._compare_seniority_levels(resume_level, required_level)
        exp_score *= level_match
        
        return {
            'score': min(exp_score, 100),
            'resume_years': resume_exp_years,
            'required_years': required_years,
            'gap': max(0, required_years - resume_exp_years),
            'level_match': level_match,
            'resume_level': resume_level,
            'required_level': required_level
        }
    
    def _calculate_education_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate education matching score"""
        resume_education = resume_data.get('education', [])
        job_education_req = job_data.get('education_requirements', {})
        
        if not job_education_req.get('degree_level'):
            return {'score': 90, 'match': True, 'details': 'No specific education requirement'}
        
        required_level = job_education_req['degree_level']
        required_fields = job_education_req.get('field_of_study', [])
        is_required = job_education_req.get('required', False)
        
        # Education level hierarchy
        level_hierarchy = {
            'high_school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5
        }
        
        # Find highest education level in resume
        highest_resume_level = 0
        matching_fields = []
        
        for edu in resume_education:
            degree_text = edu.get('degree', '').lower()
            for level, hierarchy_value in level_hierarchy.items():
                if any(keyword in degree_text for keyword in self._get_degree_keywords(level)):
                    highest_resume_level = max(highest_resume_level, hierarchy_value)
                    
                    # Check field match
                    for field in required_fields:
                        if field.lower() in degree_text:
                            matching_fields.append(field)
        
        required_level_value = level_hierarchy.get(required_level, 3)
        
        # Calculate score
        if highest_resume_level >= required_level_value:
            base_score = 95
        elif highest_resume_level == required_level_value - 1:
            base_score = 80
        else:
            base_score = 60 if not is_required else 40
        
        # Bonus for field match
        if required_fields and matching_fields:
            field_bonus = min(len(matching_fields) / len(required_fields) * 20, 20)
            base_score = min(base_score + field_bonus, 100)
        
        return {
            'score': base_score,
            'match': highest_resume_level >= required_level_value,
            'resume_level': highest_resume_level,
            'required_level': required_level_value,
            'field_match': matching_fields,
            'is_required': is_required
        }
    
    def _calculate_projects_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate projects relevance score"""
        projects = resume_data.get('projects', [])
        job_skills = self._flatten_skills({'technical': job_data.get('required_skills', [])})
        
        if not projects:
            return {'score': 70, 'relevant_projects': [], 'project_skills': []}
        
        relevant_projects = []
        project_skills = set()
        
        for project in projects:
            project_text = f"{project.get('name', '')} {project.get('description', '')}".lower()
            
            # Check for skill mentions in project
            mentioned_skills = []
            for skill in job_skills:
                if skill.lower() in project_text:
                    mentioned_skills.append(skill)
                    project_skills.add(skill)
            
            if mentioned_skills:
                relevant_projects.append({
                    'name': project.get('name', ''),
                    'skills': mentioned_skills
                })
        
        # Calculate score based on relevance
        if not relevant_projects:
            score = 60
        else:
            skill_coverage = len(project_skills) / max(len(job_skills), 1)
            project_count_bonus = min(len(relevant_projects) * 5, 15)
            score = min(70 + skill_coverage * 30 + project_count_bonus, 100)
        
        return {
            'score': score,
            'relevant_projects': relevant_projects,
            'project_skills': list(project_skills),
            'total_projects': len(projects)
        }
    
    def _calculate_certification_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate certification matching score"""
        resume_certs = set(cert.lower() for cert in resume_data.get('certifications', []))
        job_certs = set(cert.lower() for cert in job_data.get('certifications', []))
        
        if not job_certs:
            return {'score': 90, 'matched': [], 'missing': [], 'bonus_certs': list(resume_certs)}
        
        matched_certs = resume_certs.intersection(job_certs)
        missing_certs = job_certs - resume_certs
        bonus_certs = resume_certs - job_certs
        
        # Calculate score
        if not job_certs:
            base_score = 90
        else:
            match_percentage = len(matched_certs) / len(job_certs)
            base_score = 60 + match_percentage * 40
        
        # Bonus for additional relevant certifications
        bonus = min(len(bonus_certs) * 5, 15)
        final_score = min(base_score + bonus, 100)
        
        return {
            'score': final_score,
            'matched': list(matched_certs),
            'missing': list(missing_certs),
            'bonus_certs': list(bonus_certs)
        }
    
    def _flatten_skills(self, skills_dict: Dict[str, List[str]]) -> List[str]:
        """Flatten skills dictionary to a list"""
        flattened = []
        for category, skills in skills_dict.items():
            flattened.extend(skills)
        return flattened
    
    def _identify_skill_gaps(self, resume_skills: Dict[str, List[str]], job_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify skill gaps by category"""
        gaps = {}
        
        required_skills = job_data.get('required_skills', [])
        preferred_skills = job_data.get('preferred_skills', [])
        
        all_resume_skills = set(skill.lower() for skill in self._flatten_skills(resume_skills))
        
        # Group missing skills by category
        skill_database = {
            'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#'],
            'frameworks': ['react', 'angular', 'vue.js', 'node.js', 'django', 'flask'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'google cloud'],
            'devops_tools': ['docker', 'kubernetes', 'jenkins']
        }
        
        for category, category_skills in skill_database.items():
            missing_in_category = []
            for skill in required_skills + preferred_skills:
                if (skill.lower() not in all_resume_skills and 
                    skill.lower() in [s.lower() for s in category_skills]):
                    missing_in_category.append(skill)
            if missing_in_category:
                gaps[category] = missing_in_category
        
        return gaps
    
    def _estimate_experience_years(self, resume_data: Dict[str, Any]) -> int:
        """Estimate years of experience from resume data"""
        experiences = resume_data.get('experience', [])
        
        if not experiences:
            # Fallback to project count and education
            project_count = len(resume_data.get('projects', []))
            education_count = len(resume_data.get('education', []))
            return max(1, project_count // 2 + education_count)
        
        # Simple heuristic: count unique experiences
        return max(len(experiences), 1)
    
    def _estimate_seniority_level(self, resume_data: Dict[str, Any]) -> str:
        """Estimate seniority level from resume data"""
        exp_years = self._estimate_experience_years(resume_data)
        skill_count = len(self._flatten_skills(resume_data.get('skills', {})))
        project_count = len(resume_data.get('projects', []))
        
        # Simple scoring system
        seniority_score = exp_years * 2 + skill_count * 0.1 + project_count * 0.5
        
        if seniority_score >= 10:
            return 'senior-level'
        elif seniority_score >= 5:
            return 'mid-level'
        else:
            return 'entry-level'
    
    def _compare_seniority_levels(self, resume_level: str, required_level: str) -> float:
        """Compare seniority levels and return multiplier"""
        level_values = {
            'entry-level': 1,
            'mid-level': 2,
            'senior-level': 3,
            'executive': 4
        }
        
        resume_value = level_values.get(resume_level, 2)
        required_value = level_values.get(required_level, 2)
        
        if resume_value >= required_value:
            return 1.0
        elif resume_value == required_value - 1:
            return 0.9
        else:
            return 0.8
    
    def _get_degree_keywords(self, level: str) -> List[str]:
        """Get keywords for degree levels"""
        keywords = {
            'high_school': ['high school', 'diploma', 'ged'],
            'associate': ['associate', 'a.a.', 'a.s.'],
            'bachelor': ['bachelor', 'b.s.', 'b.a.', 'b.tech', 'undergraduate'],
            'master': ['master', 'm.s.', 'm.a.', 'm.tech', 'graduate'],
            'phd': ['phd', 'ph.d.', 'doctorate', 'doctoral']
        }
        return keywords.get(level, [])
    
    def _generate_recommendations(self, skills_score: Dict, experience_score: Dict, 
                                education_score: Dict, projects_score: Dict, 
                                certification_score: Dict, resume_data: Dict, 
                                job_data: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Skills recommendations
        if skills_score['score'] < 70:
            missing_skills = skills_score['missing'][:5]  # Top 5 missing
            if missing_skills:
                recommendations.append(
                    f"Consider developing skills in: {', '.join(missing_skills)}"
                )
        
        # Experience recommendations
        if experience_score['gap'] > 0:
            recommendations.append(
                f"Gain {experience_score['gap']} more years of relevant experience"
            )
        
        # Education recommendations
        if education_score['score'] < 80 and education_score['is_required']:
            recommendations.append(
                "Consider pursuing the required educational qualification"
            )
        
        # Projects recommendations
        if projects_score['score'] < 70:
            recommendations.append(
                "Build projects showcasing relevant technologies and skills"
            )
        
        # Certification recommendations
        if certification_score['missing']:
            cert_list = ', '.join(certification_score['missing'][:3])
            recommendations.append(f"Consider obtaining certifications: {cert_list}")
        
        # Generic recommendations based on overall score
        overall = (skills_score['score'] + experience_score['score'] + 
                  education_score['score'] + projects_score['score'] + 
                  certification_score['score']) / 5
        
        if overall < 60:
            recommendations.append("Focus on developing core technical skills for this role")
        elif overall < 80:
            recommendations.append("Strengthen weak areas to become a stronger candidate")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _create_match_summary(self, overall_score: float, skills_score: Dict, 
                            experience_score: Dict) -> Dict[str, Any]:
        """Create a summary of the match"""
        if overall_score >= 80:
            verdict = "Excellent Match"
            description = "Strong alignment with job requirements"
        elif overall_score >= 65:
            verdict = "Good Match"
            description = "Good fit with some areas for improvement"
        elif overall_score >= 50:
            verdict = "Fair Match"
            description = "Moderate fit, several gaps to address"
        else:
            verdict = "Poor Match"
            description = "Significant gaps in requirements"
        
        return {
            'verdict': verdict,
            'description': description,
            'key_strengths': self._identify_strengths(skills_score, experience_score),
            'improvement_areas': self._identify_weaknesses(skills_score, experience_score)
        }
    
    def _identify_strengths(self, skills_score: Dict, experience_score: Dict) -> List[str]:
        """Identify candidate strengths"""
        strengths = []
        
        if skills_score['score'] >= 80:
            strengths.append("Strong technical skill alignment")
        
        if experience_score['score'] >= 80:
            strengths.append("Relevant experience level")
        
        if skills_score['matched']:
            top_skills = skills_score['matched'][:3]
            strengths.append(f"Proficiency in {', '.join(top_skills)}")
        
        return strengths
    
    def _identify_weaknesses(self, skills_score: Dict, experience_score: Dict) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []
        
        if skills_score['score'] < 60:
            weaknesses.append("Limited technical skill match")
        
        if experience_score['gap'] > 2:
            weaknesses.append("Insufficient relevant experience")
        
        if skills_score['missing']:
            key_missing = skills_score['missing'][:2]
            weaknesses.append(f"Missing key skills: {', '.join(key_missing)}")
        
        return weaknesses
    
    def combine_scores(self, traditional_scores: Dict[str, Any], 
                      semantic_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Combine traditional and semantic-based scores"""
        # Weight traditional vs semantic scores
        traditional_weight = 0.7
        semantic_weight = 0.3
        
        # Combine overall scores
        combined_overall = (traditional_scores['overall_score'] * traditional_weight + 
                          semantic_scores.get('overall_score', traditional_scores['overall_score']) * semantic_weight)
        
        # Update the traditional scores with semantic insights
        combined_scores = traditional_scores.copy()
        combined_scores['overall_score'] = round(combined_overall, 1)
        
        # Add semantic insights
        if 'semantic_insights' in semantic_scores:
            combined_scores['semantic_insights'] = semantic_scores['semantic_insights']
        
        # Enhance recommendations with semantic suggestions
        semantic_recommendations = semantic_scores.get('recommendations', [])
        combined_scores['recommendations'].extend(semantic_recommendations[:3])
        
        return combined_scores