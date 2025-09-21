"""
Workflow Manager
Manages different workflow types and provides fallbacks when dependencies are unavailable.
"""

from typing import Dict, Any, Optional
import logging

class WorkflowManager:
    """Manages workflow execution with fallbacks"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize workflow manager"""
        self.api_key = api_key
        self.langchain_workflow = None
        self.semantic_matcher = None
        self.vector_store = None
        self.keyword_matcher = None
        
        self._initialize_workflows()
    
    def _initialize_workflows(self):
        """Initialize available workflows"""
        
        # Try to initialize LangChain workflow
        if self.api_key:
            try:
                from langchain_workflow import LangChainResumeWorkflow
                self.langchain_workflow = LangChainResumeWorkflow(api_key=self.api_key)
                logging.info("LangChain workflow initialized successfully")
            except Exception as e:
                logging.warning(f"LangChain workflow unavailable: {e}")
                
                # Fallback to semantic matcher
                try:
                    from semantic_matcher import SemanticMatcher
                    self.semantic_matcher = SemanticMatcher(api_key=self.api_key)
                    logging.info("Semantic matcher fallback initialized")
                except Exception as e2:
                    logging.warning(f"Semantic matcher also unavailable: {e2}")
        
        # Try to initialize vector store
        try:
            from vector_store import VectorStore
            self.vector_store = VectorStore()
            logging.info("Vector store initialized successfully")
        except Exception as e:
            logging.warning(f"Vector store unavailable: {e}")
        
        # Initialize keyword matcher (always available)
        try:
            from keyword_matcher import KeywordMatcher
            self.keyword_matcher = KeywordMatcher()
            logging.info("Keyword matcher initialized successfully")
        except Exception as e:
            logging.warning(f"Keyword matcher unavailable: {e}")
    
    def run_analysis(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using the best available method"""
        
        # Try LangChain workflow first (preferred)
        if self.langchain_workflow:
            try:
                result = self.langchain_workflow.run_analysis(resume_data, job_data)
                result['workflow_used'] = 'langchain'
                return result
            except Exception as e:
                logging.error(f"LangChain workflow failed: {e}")
        
        # Fallback to semantic matcher
        if self.semantic_matcher:
            try:
                from relevance_scorer import RelevanceScorer
                scorer = RelevanceScorer()
                
                # Traditional scoring
                traditional_scores = scorer.calculate_relevance(resume_data, job_data)
                
                # Enhanced with semantic analysis
                semantic_scores = self.semantic_matcher.analyze_compatibility(resume_data, job_data)
                combined_scores = scorer.combine_scores(traditional_scores, semantic_scores)
                
                combined_scores['workflow_used'] = 'semantic_fallback'
                return combined_scores
                
            except Exception as e:
                logging.error(f"Semantic matcher failed: {e}")
        
        # Final fallback to traditional scoring with keyword analysis
        try:
            from relevance_scorer import RelevanceScorer
            scorer = RelevanceScorer()
            scores = scorer.calculate_relevance(resume_data, job_data)
            
            # Add comprehensive keyword analysis if available
            if self.keyword_matcher:
                keyword_analysis = self.keyword_matcher.comprehensive_keyword_analysis(resume_data, job_data)
                scores['keyword_analysis'] = keyword_analysis
                scores['tf_idf_score'] = keyword_analysis['method_scores']['tf_idf']
                scores['bm25_score'] = keyword_analysis['method_scores']['bm25']
                scores['fuzzy_match_score'] = keyword_analysis['method_scores']['fuzzy_match']
                
                # Update overall score with keyword analysis
                traditional_score = scores.get('overall_score', 50)
                keyword_score = keyword_analysis['method_scores']['combined']
                combined_score = (traditional_score * 0.6) + (keyword_score * 0.4)
                scores['overall_score'] = combined_score
                scores['workflow_used'] = 'traditional_with_keywords'
            else:
                scores['workflow_used'] = 'traditional_only'
                
            return scores
            
        except Exception as e:
            logging.error(f"All workflows failed: {e}")
            return {
                'overall_score': 50.0,
                'workflow_used': 'emergency_fallback',
                'error': str(e),
                'verdict': 'Analysis Error - Manual Review Required'
            }
    
    def get_vector_similarity(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get vector similarity if available"""
        
        if self.vector_store:
            try:
                return self.vector_store.semantic_search(job_data, resume_data)
            except Exception as e:
                logging.warning(f"Vector similarity failed: {e}")
        
        return {"similarity_score": 70.0, "method": "unavailable"}
    
    def get_workflow_status(self) -> Dict[str, bool]:
        """Get status of available workflows"""
        return {
            'langchain_available': self.langchain_workflow is not None,
            'semantic_matcher_available': self.semantic_matcher is not None,
            'vector_store_available': self.vector_store is not None,
            'keyword_matcher_available': self.keyword_matcher is not None,
            'api_key_provided': self.api_key is not None
        }
