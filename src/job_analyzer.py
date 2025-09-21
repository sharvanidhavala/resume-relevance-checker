"""
Job Analyzer Module
Analyzes job descriptions to extract requirements, skills, and qualifications.
"""

import re
import logging
from typing import Dict, List, Set, Any
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class JobAnalyzer:
    """Analyze job descriptions and extract structured requirements"""
    
    def __init__(self):
        """Initialize the job analyzer"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.skill_database = self._load_skill_database()
        self.seniority_levels = self._load_seniority_levels()
        
    def analyze_job_description(self, job_text: str) -> Dict[str, Any]:
        """
        Analyze a job description and extract structured data
        
        Args:
            job_text: Raw job description text
            
        Returns:
            Dictionary containing analyzed job data
        """
        if not job_text.strip():
            raise ValueError("Job description cannot be empty")
        
        # Clean and normalize text
        cleaned_text = self._clean_text(job_text)
        
        # Extract different components
        analysis = {
            'raw_text': job_text,
            'cleaned_text': cleaned_text,
            'job_title': self._extract_job_title(job_text),
            'company_info': self._extract_company_info(job_text),
            'required_skills': self._extract_required_skills(cleaned_text),
            'preferred_skills': self._extract_preferred_skills(cleaned_text),
            'education_requirements': self._extract_education_requirements(cleaned_text),
            'experience_requirements': self._extract_experience_requirements(cleaned_text),
            'certifications': self._extract_certifications(cleaned_text),
            'responsibilities': self._extract_responsibilities(cleaned_text),
            'benefits': self._extract_benefits(cleaned_text),
            'location': self._extract_location(job_text),
            'employment_type': self._extract_employment_type(job_text),
            'seniority_level': self._extract_seniority_level(cleaned_text),
            'industry': self._extract_industry(cleaned_text),
            'key_phrases': self._extract_key_phrases(cleaned_text),
            'requirements_summary': self._summarize_requirements(cleaned_text)
        }
        
        # Calculate analysis metrics
        analysis['metrics'] = self._calculate_job_metrics(analysis)
        
        return analysis
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize job description text"""
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove bullet point characters
        text = re.sub(r'[•·‣⁃]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from job description"""
        lines = text.split('\n')
        
        # Common job title patterns
        title_patterns = [
            r'(?:position|role|job title|title):\s*(.+)',
            r'we are looking for (?:an? |a )?(.+?) to',
            r'seeking (?:an? |a )?(.+?) to',
            r'^(.+?)(?:\s*-|\s*\||\s*at)'
        ]
        
        # Check first few lines for title
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 6:
                # Simple heuristic: short lines at the beginning might be titles
                if not any(word.lower() in line.lower() for word in ['company', 'location', 'posted', 'salary']):
                    return line
        
        # Try pattern matching
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_company_info(self, text: str) -> Dict[str, str]:
        """Extract company information"""
        company_info = {
            'name': '',
            'size': '',
            'industry': ''
        }
        
        # Company name patterns
        company_patterns = [
            r'company:\s*(.+)',
            r'about (.+?)[:.]',
            r'join (.+?) and',
            r'(.+?) is (?:a |an )?(?:leading|growing|innovative)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['name'] = match.group(1).strip()
                break
        
        return company_info
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills and technologies"""
        required_skills = set()
        
        # Look for required skills section
        required_sections = [
            r'required skills?:(.+?)(?:preferred|nice|responsibilities|qualifications|$)',
            r'must have:(.+?)(?:preferred|nice|responsibilities|qualifications|$)',
            r'requirements?:(.+?)(?:preferred|nice|responsibilities|qualifications|$)',
            r'essential skills?:(.+?)(?:preferred|nice|responsibilities|qualifications|$)'
        ]
        
        for pattern in required_sections:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                skills = self._extract_skills_from_text(section_text)
                required_skills.update(skills)
        
        # If no explicit required section, extract from entire text with higher confidence
        if not required_skills:
            all_skills = self._extract_skills_from_text(text)
            required_skills.update(all_skills[:10])  # Top 10 most mentioned
        
        return list(required_skills)
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred/nice-to-have skills"""
        preferred_skills = set()
        
        # Look for preferred skills section
        preferred_sections = [
            r'preferred skills?:(.+?)(?:required|responsibilities|qualifications|$)',
            r'nice to have:(.+?)(?:required|responsibilities|qualifications|$)',
            r'bonus:(.+?)(?:required|responsibilities|qualifications|$)',
            r'additional skills?:(.+?)(?:required|responsibilities|qualifications|$)'
        ]
        
        for pattern in preferred_sections:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                skills = self._extract_skills_from_text(section_text)
                preferred_skills.update(skills)
        
        return list(preferred_skills)
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from a given text section"""
        found_skills = []
        text_lower = text.lower()
        
        # Search for skills in our database
        for category, skills in self.skill_database.items():
            for skill in skills:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
        
        return found_skills
    
    def _extract_education_requirements(self, text: str) -> Dict[str, Any]:
        """Extract education requirements"""
        education = {
            'degree_level': '',
            'field_of_study': [],
            'required': False,
            'alternatives_accepted': []
        }
        
        # Degree level patterns
        degree_patterns = {
            'bachelor': r'bachelor\'?s?|b\.?s\.?|undergraduate',
            'master': r'master\'?s?|m\.?s\.?|graduate degree',
            'phd': r'phd|ph\.?d\.?|doctorate|doctoral',
            'associate': r'associate|a\.?a\.?|a\.?s\.?',
            'high_school': r'high school|diploma|ged'
        }
        
        text_lower = text.lower()
        for level, pattern in degree_patterns.items():
            if re.search(pattern, text_lower):
                education['degree_level'] = level
                break
        
        # Field of study
        field_patterns = [
            r'computer science', r'engineering', r'information technology',
            r'software engineering', r'data science', r'mathematics',
            r'business', r'marketing', r'finance'
        ]
        
        for field in field_patterns:
            if re.search(r'\b' + field + r'\b', text_lower):
                education['field_of_study'].append(field.title())
        
        # Check if education is required or preferred
        education['required'] = bool(re.search(r'require[sd].*degree', text_lower))
        
        return education
    
    def _extract_experience_requirements(self, text: str) -> Dict[str, Any]:
        """Extract experience requirements"""
        experience = {
            'years_required': 0,
            'years_preferred': 0,
            'level': '',
            'specific_experience': []
        }
        
        # Extract years of experience
        year_patterns = [
            r'(\d+)(?:\+|\s*or\s*more)?\s*years?\s*of\s*experience',
            r'(\d+)(?:\+|\s*or\s*more)?\s*years?\s*experience',
            r'minimum\s*of\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text.lower())
            if match:
                years = int(match.group(1))
                experience['years_required'] = max(experience['years_required'], years)
        
        # Extract seniority level
        for level, patterns in self.seniority_levels.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    experience['level'] = level
                    break
        
        return experience
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract mentioned certifications"""
        certifications = []
        
        cert_patterns = [
            r'aws\s+certified', r'azure\s+certified', r'google\s+cloud\s+certified',
            r'pmp', r'scrum\s+master', r'agile\s+certified',
            r'cissp', r'cisa', r'comptia', r'cisco\s+certified'
        ]
        
        text_lower = text.lower()
        for pattern in cert_patterns:
            if re.search(pattern, text_lower):
                certifications.append(pattern.replace(r'\\s\\+', ' ').title())
        
        return certifications
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities"""
        responsibilities = []
        
        # Look for responsibilities section
        resp_patterns = [
            r'responsibilities:(.+?)(?:requirements|qualifications|skills|benefits|$)',
            r'duties:(.+?)(?:requirements|qualifications|skills|benefits|$)',
            r'what you\'ll do:(.+?)(?:requirements|qualifications|skills|benefits|$)',
            r'role overview:(.+?)(?:requirements|qualifications|skills|benefits|$)'
        ]
        
        for pattern in resp_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                resp_text = match.group(1)
                # Split by bullet points or line breaks
                items = re.split(r'[•·‣⁃]|\n-|\n\*', resp_text)
                for item in items:
                    item = item.strip()
                    if len(item) > 10 and len(item) < 200:  # Filter reasonable responsibilities
                        responsibilities.append(item)
                break
        
        return responsibilities[:10]  # Limit to top 10
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract job benefits"""
        benefits = []
        
        benefit_keywords = [
            'health insurance', 'dental', 'vision', '401k', 'retirement',
            'vacation', 'pto', 'paid time off', 'flexible hours', 'remote work',
            'work from home', 'stock options', 'equity', 'bonus',
            'professional development', 'training', 'conference'
        ]
        
        text_lower = text.lower()
        for benefit in benefit_keywords:
            if benefit in text_lower:
                benefits.append(benefit.title())
        
        return benefits
    
    def _extract_location(self, text: str) -> str:
        """Extract job location"""
        # Location patterns
        location_patterns = [
            r'location:\s*(.+)',
            r'based in\s*(.+)',
            r'located in\s*(.+)',
            r'office in\s*(.+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for common location formats
        location_match = re.search(r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b', text)
        if location_match:
            return location_match.group(1)
        
        return ""
    
    def _extract_employment_type(self, text: str) -> str:
        """Extract employment type (full-time, part-time, contract, etc.)"""
        employment_types = {
            'full-time': [r'full.?time', r'permanent'],
            'part-time': [r'part.?time'],
            'contract': [r'contract', r'contractor', r'temporary'],
            'freelance': [r'freelance', r'freelancer'],
            'internship': [r'intern', r'internship'],
            'remote': [r'remote', r'work from home', r'distributed']
        }
        
        text_lower = text.lower()
        for emp_type, patterns in employment_types.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return emp_type
        
        return "full-time"  # Default assumption
    
    def _extract_seniority_level(self, text: str) -> str:
        """Extract seniority level"""
        text_lower = text.lower()
        
        for level, patterns in self.seniority_levels.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level
        
        return "mid-level"  # Default
    
    def _extract_industry(self, text: str) -> str:
        """Extract industry information"""
        industries = {
            'technology': ['tech', 'software', 'it', 'saas', 'startup'],
            'finance': ['bank', 'financial', 'fintech', 'investment'],
            'healthcare': ['health', 'medical', 'pharmaceutical', 'biotech'],
            'education': ['education', 'university', 'school', 'academic'],
            'retail': ['retail', 'ecommerce', 'shopping'],
            'consulting': ['consulting', 'advisory', 'professional services']
        }
        
        text_lower = text.lower()
        for industry, keywords in industries.items():
            if any(keyword in text_lower for keyword in keywords):
                return industry
        
        return ""
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from job description"""
        # Tokenize and remove stopwords
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Get most common words
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(20)]
        
        return common_words
    
    def _summarize_requirements(self, text: str) -> Dict[str, List[str]]:
        """Summarize key requirements"""
        summary = {
            'must_have': [],
            'nice_to_have': [],
            'key_responsibilities': []
        }
        
        # Extract must-have requirements
        must_have_patterns = [
            r'must have (.+?)(?:[.\n]|$)',
            r'required (.+?)(?:[.\n]|$)',
            r'essential (.+?)(?:[.\n]|$)'
        ]
        
        for pattern in must_have_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            summary['must_have'].extend([match.strip() for match in matches])
        
        # Extract nice-to-have requirements
        nice_patterns = [
            r'nice to have (.+?)(?:[.\n]|$)',
            r'preferred (.+?)(?:[.\n]|$)',
            r'bonus (.+?)(?:[.\n]|$)'
        ]
        
        for pattern in nice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            summary['nice_to_have'].extend([match.strip() for match in matches])
        
        return summary
    
    def _calculate_job_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate job analysis metrics"""
        metrics = {
            'total_requirements': len(analysis['required_skills']) + len(analysis['preferred_skills']),
            'required_skills_count': len(analysis['required_skills']),
            'preferred_skills_count': len(analysis['preferred_skills']),
            'has_education_req': bool(analysis['education_requirements']['degree_level']),
            'min_experience_years': analysis['experience_requirements']['years_required'],
            'certification_count': len(analysis['certifications']),
            'responsibility_count': len(analysis['responsibilities']),
            'benefit_count': len(analysis['benefits']),
            'complexity_score': 0
        }
        
        # Calculate complexity score
        complexity_factors = [
            metrics['total_requirements'] * 0.1,
            metrics['min_experience_years'] * 0.15,
            metrics['certification_count'] * 0.2,
            (1 if metrics['has_education_req'] else 0) * 0.1
        ]
        
        metrics['complexity_score'] = min(sum(complexity_factors), 10.0)
        
        return metrics
    
    def _load_skill_database(self) -> Dict[str, List[str]]:
        """Load comprehensive skill database"""
        return {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
                'Swift', 'Kotlin', 'PHP', 'Ruby', 'Scala', 'R', 'MATLAB', 'SQL'
            ],
            'web_technologies': [
                'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express',
                'Django', 'Flask', 'Spring Boot', 'ASP.NET', 'Laravel', 'Rails'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle',
                'SQL Server', 'Cassandra', 'DynamoDB', 'Elasticsearch'
            ],
            'cloud_platforms': [
                'AWS', 'Azure', 'Google Cloud Platform', 'GCP', 'Heroku',
                'DigitalOcean', 'IBM Cloud'
            ],
            'devops_tools': [
                'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions',
                'Terraform', 'Ansible', 'Chef', 'Puppet'
            ],
            'data_science': [
                'Machine Learning', 'Deep Learning', 'Data Science', 'Pandas',
                'NumPy', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Jupyter'
            ],
            'soft_skills': [
                'Communication', 'Leadership', 'Team work', 'Problem solving',
                'Critical thinking', 'Time management', 'Adaptability'
            ]
        }
    
    def _load_seniority_levels(self) -> Dict[str, List[str]]:
        """Load seniority level patterns"""
        return {
            'entry-level': [r'entry.?level', r'junior', r'graduate', r'trainee', r'intern'],
            'mid-level': [r'mid.?level', r'intermediate', r'regular', r'standard'],
            'senior-level': [r'senior', r'sr\.?', r'lead', r'principal', r'staff'],
            'executive': [r'director', r'vp', r'vice president', r'cto', r'ceo', r'head of']
        }