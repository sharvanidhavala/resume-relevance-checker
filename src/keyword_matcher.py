"""
Keyword Matching Module
Implements TF-IDF, BM25, and fuzzy matching for traditional keyword-based analysis.
"""

import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using fallback implementations")

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available, using basic string matching")

class KeywordMatcher:
    """Traditional keyword matching with TF-IDF, BM25, and fuzzy matching"""
    
    def __init__(self):
        """Initialize keyword matcher"""
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000,
                lowercase=True
            )
    
    def tfidf_similarity(self, resume_text: str, job_text: str) -> float:
        """Calculate TF-IDF similarity between resume and job description"""
        
        if not SKLEARN_AVAILABLE:
            # Fallback to simple word overlap
            return self._simple_word_overlap(resume_text, job_text)
        
        try:
            # Prepare documents
            documents = [resume_text, job_text]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logging.warning(f"TF-IDF calculation failed: {e}")
            return self._simple_word_overlap(resume_text, job_text)
    
    def bm25_score(self, resume_text: str, job_text: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 relevance score
        
        Notes:
        - Uses document frequency (number of documents the term appears in), not term frequency across all tokens.
        - Adds +1 inside the log to ensure the argument is positive and avoid math domain errors when df=N.
        """
        
        # Tokenize documents
        resume_tokens = self._tokenize(resume_text)
        job_tokens = self._tokenize(job_text)
        
        # Build document list and document frequency (df) per term
        documents = [set(resume_tokens), set(job_tokens)]
        total_docs = len(documents)
        df = {}
        for term in set(resume_tokens + job_tokens):
            df[term] = sum(1 for doc in documents if term in doc)
        
        # Calculate BM25 score
        score = 0.0
        resume_length = len(resume_tokens)
        avg_doc_length = (len(resume_tokens) + len(job_tokens)) / 2 if total_docs > 0 else 1
        
        for term in set(job_tokens):
            if term in resume_tokens:
                tf = resume_tokens.count(term)
                df_term = df.get(term, 0)
                # Standard BM25 IDF variant to avoid non-positive log args
                idf = math.log(((total_docs - df_term + 0.5) / (df_term + 0.5)) + 1.0)
                
                denom = tf + k1 * (1 - b + b * (resume_length / max(avg_doc_length, 1e-8)))
                term_score = (tf * (k1 + 1)) / denom * idf
                score += term_score
        
        # Normalize score to [0, 1]
        # Heuristic normalization by the number of unique job terms
        max_terms = max(len(set(job_tokens)), 1)
        normalized_score = score / (max_terms + 1e-8)
        
        return float(min(max(normalized_score, 0.0), 1.0))
    
    def fuzzy_match_skills(self, resume_skills: List[str], job_skills: List[str], threshold: int = 80) -> Dict[str, Any]:
        """Perform fuzzy matching on skills lists"""
        
        if not FUZZYWUZZY_AVAILABLE:
            # Fallback to exact matching
            return self._exact_skill_match(resume_skills, job_skills)
        
        try:
            matches = []
            match_scores = []
            unmatched_job_skills = []
            
            for job_skill in job_skills:
                # Find best match for each job skill
                best_match = process.extractOne(job_skill, resume_skills, scorer=fuzz.token_sort_ratio)
                
                if best_match and best_match[1] >= threshold:
                    matches.append({
                        'job_skill': job_skill,
                        'resume_skill': best_match[0],
                        'score': best_match[1]
                    })
                    match_scores.append(best_match[1])
                else:
                    unmatched_job_skills.append(job_skill)
            
            return {
                'matches': matches,
                'match_count': len(matches),
                'total_job_skills': len(job_skills),
                'match_percentage': (len(matches) / len(job_skills) * 100) if job_skills else 0,
                'average_score': sum(match_scores) / len(match_scores) if match_scores else 0,
                'unmatched_skills': unmatched_job_skills
            }
            
        except Exception as e:
            logging.warning(f"Fuzzy matching failed: {e}")
            return self._exact_skill_match(resume_skills, job_skills)
    
    def keyword_density_analysis(self, resume_text: str, job_keywords: List[str]) -> Dict[str, float]:
        """Analyze keyword density in resume"""
        
        resume_lower = resume_text.lower()
        resume_words = self._tokenize(resume_lower)
        total_words = len(resume_words)
        
        keyword_density = {}
        
        for keyword in job_keywords:
            keyword_lower = keyword.lower()
            
            # Count exact matches
            exact_count = resume_lower.count(keyword_lower)
            
            # Count word-level matches
            word_count = resume_words.count(keyword_lower)
            
            # Calculate density
            density = (word_count / total_words * 100) if total_words > 0 else 0
            
            keyword_density[keyword] = {
                'exact_matches': exact_count,
                'word_matches': word_count,
                'density_percentage': density
            }
        
        return keyword_density
    
    def weighted_score_combination(self, tfidf_score: float, bm25_score: float, 
                                  fuzzy_score: float, weights: Dict[str, float] = None) -> float:
        """Combine different matching scores with weights"""
        
        if weights is None:
            weights = {
                'tfidf': 0.4,
                'bm25': 0.4,
                'fuzzy': 0.2
            }
        
        combined_score = (
            tfidf_score * weights.get('tfidf', 0.4) +
            bm25_score * weights.get('bm25', 0.4) +
            fuzzy_score * weights.get('fuzzy', 0.2)
        )
        
        return min(max(combined_score, 0), 1)  # Clamp between 0 and 1
    
    def comprehensive_keyword_analysis(self, resume_data: Dict[str, Any], 
                                     job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive keyword analysis using all methods"""
        
        # Extract texts
        resume_text = self._extract_resume_text(resume_data)
        job_text = self._extract_job_text(job_data)
        
        # Extract skills
        resume_skills = self._extract_skills_list(resume_data)
        job_skills = self._extract_job_skills_list(job_data)
        
        # Run all analyses
        tfidf_score = self.tfidf_similarity(resume_text, job_text)
        bm25_score = self.bm25_score(resume_text, job_text)
        fuzzy_results = self.fuzzy_match_skills(resume_skills, job_skills)
        fuzzy_score = fuzzy_results['match_percentage'] / 100
        
        # Keyword density analysis
        keyword_density = self.keyword_density_analysis(resume_text, job_skills)
        
        # Combined score
        combined_score = self.weighted_score_combination(tfidf_score, bm25_score, fuzzy_score)
        
        return {
            'tfidf_similarity': tfidf_score,
            'bm25_score': bm25_score,
            'fuzzy_matching': fuzzy_results,
            'keyword_density': keyword_density,
            'combined_score': combined_score,
            'method_scores': {
                'tf_idf': tfidf_score * 100,
                'bm25': bm25_score * 100,
                'fuzzy_match': fuzzy_score * 100,
                'combined': combined_score * 100
            }
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def _simple_word_overlap(self, text1: str, text2: str) -> float:
        """Fallback word overlap similarity"""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _exact_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Fallback exact skill matching"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        matches = []
        for job_skill in job_skills_lower:
            if job_skill in resume_skills_lower:
                matches.append({
                    'job_skill': job_skill,
                    'resume_skill': job_skill,
                    'score': 100
                })
        
        unmatched = [skill for skill in job_skills_lower if skill not in resume_skills_lower]
        
        return {
            'matches': matches,
            'match_count': len(matches),
            'total_job_skills': len(job_skills),
            'match_percentage': (len(matches) / len(job_skills) * 100) if job_skills else 0,
            'average_score': 100 if matches else 0,
            'unmatched_skills': unmatched
        }
    
    def _extract_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Extract full text from resume data"""
        text_parts = []
        
        # Personal info
        personal = resume_data.get('personal_info', {})
        if personal.get('title'):
            text_parts.append(personal['title'])
        
        # Summary
        if resume_data.get('summary'):
            text_parts.append(resume_data['summary'])
        
        # Skills
        for category, skills in resume_data.get('skills', {}).items():
            text_parts.extend(skills)
        
        # Experience
        for exp in resume_data.get('experience', []):
            if exp.get('title'):
                text_parts.append(exp['title'])
            if exp.get('description'):
                text_parts.append(exp['description'])
        
        # Education
        for edu in resume_data.get('education', []):
            if edu.get('degree'):
                text_parts.append(edu['degree'])
        
        # Projects
        for proj in resume_data.get('projects', []):
            if proj.get('name'):
                text_parts.append(proj['name'])
            if proj.get('description'):
                text_parts.append(proj['description'])
        
        return ' '.join(text_parts)
    
    def _extract_job_text(self, job_data: Dict[str, Any]) -> str:
        """Extract full text from job data"""
        text_parts = []
        
        if job_data.get('job_title'):
            text_parts.append(job_data['job_title'])
        
        text_parts.extend(job_data.get('required_skills', []))
        text_parts.extend(job_data.get('preferred_skills', []))
        text_parts.extend(job_data.get('responsibilities', []))
        
        return ' '.join(text_parts)
    
    def _extract_skills_list(self, resume_data: Dict[str, Any]) -> List[str]:
        """Extract skills as list from resume"""
        skills = []
        for category, skill_list in resume_data.get('skills', {}).items():
            skills.extend(skill_list)
        return skills
    
    def _extract_job_skills_list(self, job_data: Dict[str, Any]) -> List[str]:
        """Extract skills as list from job data"""
        skills = []
        skills.extend(job_data.get('required_skills', []))
        skills.extend(job_data.get('preferred_skills', []))
        return skills