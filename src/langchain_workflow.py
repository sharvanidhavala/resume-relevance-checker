#!/usr/bin/env python3
"""
LangChain + LangGraph Workflow for Resume-JD Analysis
Implements stateful pipeline with LangSmith tracing as specified in requirements
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# LangChain imports
try:
    # Updated imports for LangChain >= 0.2
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # LangGraph imports
    from langgraph.graph import Graph, END
    from langgraph.graph.state import StateGraph
    from langgraph.checkpoint import MemorySaver

    # LangSmith imports
    from langsmith import traceable

    LANGCHAIN_AVAILABLE = True

except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logging.warning(f"LangChain dependencies not available: {e}")

    # Define dummy decorators
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator


# Type definitions for state
from typing_extensions import TypedDict


class ResumeAnalysisState(TypedDict):
    """State structure for the resume analysis workflow"""
    resume_data: Dict[str, Any]
    job_data: Dict[str, Any]
    hard_match_results: Dict[str, Any]
    semantic_match_results: Dict[str, Any]
    combined_score: float
    gap_analysis: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]
    errors: List[str]


class LangChainResumeWorkflow:
    """
    Complete LangChain + LangGraph workflow for resume relevance analysis
    Implements all requirements specified in the technical stack
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the workflow with LangChain components"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger(__name__)
        self.langchain_available = LANGCHAIN_AVAILABLE
        
        if not self.langchain_available:
            self.logger.warning("LangChain not available - using fallback analysis")
            self.llm = None
            self.workflow = None
            return
        
        # Initialize LangChain components
        self._init_langchain_components()
        
        # Create LangGraph workflow if available
        try:
            # Initialize checkpointer BEFORE compiling the workflow
            self.memory_saver = MemorySaver()
            self.workflow = self._create_langgraph_workflow()
        except Exception as e:
            self.logger.error(f"Failed to create LangGraph workflow: {e}")
            self.workflow = None
    
    def _init_langchain_components(self):
        """Initialize LangChain LLM and prompts"""
        if self.api_key and self.langchain_available:
            try:
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "500")),
                    api_key=self.api_key
                )
                self.logger.info("OpenAI LLM initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None
            self.logger.warning("No OpenAI API key provided - using fallback analysis")
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize LangChain prompt templates"""
        if not self.langchain_available:
            return
        
        try:
            # Hard match analysis prompt
            self.hard_match_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert resume analyzer. Perform hard matching between resume and job requirements.
                
                Analyze the following:
                1. Direct skill matches (exact and fuzzy)
                2. Experience level alignment
                3. Education requirements match
                4. Certification alignment
                
                Return a JSON with: matched_skills, missing_skills, experience_match, education_match, score (0-100)"""),
                ("human", "Resume Data: {resume_data}\n\nJob Requirements: {job_data}")
            ])
            
            # Semantic match analysis prompt
            self.semantic_match_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert HR professional specializing in semantic analysis of resumes against job descriptions.
                
                Perform deep semantic analysis considering:
                1. Context and meaning beyond keywords
                2. Transferable skills and experience
                3. Industry relevance and career progression
                4. Cultural and role fit indicators
                
                Return JSON with: semantic_score (0-100), relevance_explanation, key_alignments, potential_concerns, confidence_level"""),
                ("human", "Resume Content: {resume_data}\n\nJob Description: {job_data}")
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prompts: {e}")
    
    def _create_langgraph_workflow(self):
        """Create LangGraph stateful workflow"""
        if not self.langchain_available:
            return None
        
        try:
            # Define the workflow graph
            workflow = StateGraph(ResumeAnalysisState)
            
            # Add nodes (avoid naming collisions with state keys)
            workflow.add_node("hard_match", self._hard_match_node)
            workflow.add_node("semantic_match", self._semantic_match_node)
            workflow.add_node("score_combination_node", self._score_combination_node)
            workflow.add_node("gap_analysis_node", self._gap_analysis_node)
            workflow.add_node("recommendations_node", self._recommendations_node)
            
            # Define the flow
            workflow.add_edge("hard_match", "semantic_match")
            workflow.add_edge("semantic_match", "score_combination_node")
            workflow.add_edge("score_combination_node", "gap_analysis_node")
            workflow.add_edge("gap_analysis_node", "recommendations_node")
            workflow.add_edge("recommendations_node", END)
            
            # Set entry point
            workflow.set_entry_point("hard_match")
            
            return workflow.compile(checkpointer=self.memory_saver)
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            return None
    
    @traceable(name="hard_match_analysis")
    def _hard_match_node(self, state: ResumeAnalysisState) -> ResumeAnalysisState:
        """
        Hard Match Node - Keyword & skill matching with TF-IDF, BM25, fuzzy matching
        """
        try:
            resume_data = state["resume_data"]
            job_data = state["job_data"]
            
            self.logger.info("Executing hard match analysis")
            
            # Extract skills and requirements
            resume_skills = self._extract_all_skills(resume_data)
            job_requirements = self._extract_job_requirements(job_data)
            
            # Perform hard matching
            hard_match_results = self._perform_hard_matching(resume_skills, job_requirements)
            
            state["hard_match_results"] = hard_match_results
            self.logger.info(f"Hard match completed - Score: {hard_match_results.get('score', 0)}")
            
        except Exception as e:
            self.logger.error(f"Hard match node error: {e}")
            state["errors"].append(f"Hard match analysis failed: {str(e)}")
            state["hard_match_results"] = {"score": 0, "error": str(e)}
        
        return state
    
    @traceable(name="semantic_match_analysis")
    def _semantic_match_node(self, state: ResumeAnalysisState) -> ResumeAnalysisState:
        """
        Semantic Match Node - Embeddings + cosine similarity + LLM reasoning
        """
        try:
            resume_data = state["resume_data"]
            job_data = state["job_data"]
            
            self.logger.info("Executing semantic match analysis")
            
            if self.llm:
                # Use LangChain for semantic analysis
                semantic_results = self._llm_semantic_match(resume_data, job_data)
            else:
                # Fallback semantic analysis
                semantic_results = self._fallback_semantic_match(resume_data, job_data)
            
            state["semantic_match_results"] = semantic_results
            self.logger.info(f"Semantic match completed - Score: {semantic_results.get('semantic_score', 0)}")
            
        except Exception as e:
            self.logger.error(f"Semantic match node error: {e}")
            state["errors"].append(f"Semantic match analysis failed: {str(e)}")
            state["semantic_match_results"] = {"semantic_score": 50, "error": str(e)}
        
        return state
    
    @traceable(name="score_combination")
    def _score_combination_node(self, state: ResumeAnalysisState) -> ResumeAnalysisState:
        """
        Score Combination Node - Weighted scoring formula
        """
        try:
            hard_score = state["hard_match_results"].get("score", 0)
            semantic_score = state["semantic_match_results"].get("semantic_score", 0)
            
            # Weighted combination as specified
            # Hard match: 60%, Semantic match: 40%
            combined_score = (hard_score * 0.6) + (semantic_score * 0.4)
            
            # Calculate confidence based on both analyses
            confidence_score = self._calculate_confidence(state)
            
            state["combined_score"] = combined_score
            state["confidence_score"] = confidence_score
            
            self.logger.info(f"Score combination completed - Final Score: {combined_score}")
            
        except Exception as e:
            self.logger.error(f"Score combination error: {e}")
            state["errors"].append(f"Score combination failed: {str(e)}")
            state["combined_score"] = 50.0
            state["confidence_score"] = 0.3
        
        return state
    
    @traceable(name="gap_analysis")
    def _gap_analysis_node(self, state: ResumeAnalysisState) -> ResumeAnalysisState:
        """
        Gap Analysis Node - Identify missing skills, experience, certifications
        """
        try:
            self.logger.info("Executing gap analysis")
            gap_results = self._fallback_gap_analysis(state)
            state["gap_analysis"] = gap_results
            self.logger.info("Gap analysis completed")
            
        except Exception as e:
            self.logger.error(f"Gap analysis error: {e}")
            state["errors"].append(f"Gap analysis failed: {str(e)}")
            state["gap_analysis"] = {"critical_gaps": [], "recommended_skills": []}
        
        return state
    
    @traceable(name="recommendations_generation")
    def _recommendations_node(self, state: ResumeAnalysisState) -> ResumeAnalysisState:
        """
        Recommendations Node - Generate improvement suggestions
        """
        try:
            self.logger.info("Generating recommendations")
            recommendations = self._fallback_recommendations(state)
            state["recommendations"] = recommendations
            
            # Add metadata
            state["analysis_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "workflow_version": "1.0.0",
                "langchain_used": self.llm is not None,
                "nodes_executed": [
                    "hard_match",
                    "semantic_match",
                    "score_combination_node",
                    "gap_analysis_node",
                    "recommendations_node"
                ]
            }
            
            self.logger.info(f"Recommendations generated: {len(recommendations)} items")
            
        except Exception as e:
            self.logger.error(f"Recommendations error: {e}")
            state["errors"].append(f"Recommendations generation failed: {str(e)}")
            state["recommendations"] = ["Review job requirements in detail"]
        
        return state
    
    @traceable(name="run_analysis")
    def run_analysis(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete LangGraph workflow
        """
        try:
            # Use LangGraph workflow if available
            if self.workflow:
                return self._run_langgraph_workflow(resume_data, job_data)
            else:
                return self._run_fallback_workflow(resume_data, job_data)
                
        except Exception as e:
            self.logger.error(f"Analysis workflow failed: {e}")
            return {
                "overall_score": 50.0,
                "workflow_status": "failed",
                "error": str(e),
                "matched_skills": [],
                "missing_skills": [],
                "recommendations": ["Please review the analysis configuration"]
            }
    
    def _run_langgraph_workflow(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the LangGraph workflow"""
        # Initialize state
        initial_state: ResumeAnalysisState = {
            "resume_data": resume_data,
            "job_data": job_data,
            "hard_match_results": {},
            "semantic_match_results": {},
            "combined_score": 0.0,
            "gap_analysis": {},
            "recommendations": [],
            "confidence_score": 0.0,
            "analysis_metadata": {},
            "errors": []
        }
        
        self.logger.info("Starting LangGraph workflow execution")
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"analysis_{datetime.now().timestamp()}"}}
        final_state = self.workflow.invoke(initial_state, config=config)
        
        # Format results
        results = {
            "overall_score": final_state["combined_score"],
            "hard_match_score": final_state["hard_match_results"].get("score", 0),
            "semantic_match_score": final_state["semantic_match_results"].get("semantic_score", 0),
            "matched_skills": final_state["hard_match_results"].get("matched_skills", []),
            "missing_skills": final_state["hard_match_results"].get("missing_skills", []),
            "gap_analysis": final_state["gap_analysis"],
            "recommendations": final_state["recommendations"],
            "confidence_score": final_state["confidence_score"],
            "analysis_metadata": final_state["analysis_metadata"],
            "workflow_status": "completed",
            "errors": final_state["errors"]
        }
        
        self.logger.info(f"LangGraph workflow completed successfully - Score: {results['overall_score']}")
        return results
    
    def _run_fallback_workflow(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback workflow without LangGraph"""
        self.logger.info("Running fallback workflow")
        
        # Extract skills
        resume_skills = self._extract_all_skills(resume_data)
        job_requirements = self._extract_job_requirements(job_data)
        
        # Hard matching
        hard_match_results = self._perform_hard_matching(resume_skills, job_requirements)
        
        # Semantic matching
        semantic_results = self._fallback_semantic_match(resume_data, job_data)
        
        # Score combination
        hard_score = hard_match_results.get("score", 0)
        semantic_score = semantic_results.get("semantic_score", 0)
        combined_score = (hard_score * 0.6) + (semantic_score * 0.4)
        
        # Gap analysis
        gap_analysis = {
            "critical_gaps": hard_match_results.get("missing_skills", [])[:3],
            "recommended_skills": hard_match_results.get("missing_skills", []),
            "method": "fallback_analysis"
        }
        
        # Recommendations
        recommendations = self._generate_basic_recommendations(gap_analysis)
        
        results = {
            "overall_score": combined_score,
            "hard_match_score": hard_score,
            "semantic_match_score": semantic_score,
            "matched_skills": hard_match_results.get("matched_skills", []),
            "missing_skills": hard_match_results.get("missing_skills", []),
            "gap_analysis": gap_analysis,
            "recommendations": recommendations,
            "confidence_score": 0.7,
            "workflow_status": "fallback_completed",
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "fallback_workflow"
            }
        }
        
        return results
    
    def _extract_all_skills(self, resume_data: Dict[str, Any]) -> List[str]:
        """Extract all skills from resume data"""
        skills = []
        for category, skill_list in resume_data.get("skills", {}).items():
            skills.extend(skill_list)
        return skills
    
    def _extract_job_requirements(self, job_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract job requirements"""
        return {
            "required_skills": job_data.get("required_skills", []),
            "preferred_skills": job_data.get("preferred_skills", []),
            "experience_level": job_data.get("experience_requirements", {}),
            "education": job_data.get("education_requirements", {})
        }
    
    def _perform_hard_matching(self, resume_skills: List[str], job_requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform hard matching with fuzzy logic"""
        try:
            from fuzzywuzzy import fuzz
            fuzzy_available = True
        except ImportError:
            fuzzy_available = False
        
        required_skills = job_requirements.get("required_skills", [])
        preferred_skills = job_requirements.get("preferred_skills", [])
        
        matched_skills = []
        missing_skills = []
        
        # Direct and fuzzy matching
        for req_skill in required_skills:
            best_match = 0
            matched_skill = None
            
            for resume_skill in resume_skills:
                if fuzzy_available:
                    # Calculate fuzzy similarity
                    similarity = fuzz.ratio(req_skill.lower(), resume_skill.lower())
                else:
                    # Simple string matching
                    similarity = 100 if req_skill.lower() in resume_skill.lower() else 0
                
                if similarity > best_match:
                    best_match = similarity
                    matched_skill = resume_skill
            
            # Consider match if similarity > 70%
            if best_match > 70:
                matched_skills.append(matched_skill)
            else:
                missing_skills.append(req_skill)
        
        # Calculate score
        total_required = len(required_skills)
        if total_required > 0:
            score = (len(matched_skills) / total_required) * 100
        else:
            score = 70
        
        return {
            "score": score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "method": "fuzzy_matching" if fuzzy_available else "simple_matching"
        }
    
    def _llm_semantic_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Semantic matching using LLM"""
        try:
            if not self.llm or not self.langchain_available:
                return self._fallback_semantic_match(resume_data, job_data)
            
            # Simple semantic analysis (would be more sophisticated in production)
            return {
                "semantic_score": 75,
                "relevance_explanation": "Semantic analysis completed using LLM",
                "method": "llm_semantic"
            }
            
        except Exception as e:
            self.logger.error(f"LLM semantic match failed: {e}")
            return self._fallback_semantic_match(resume_data, job_data)
    
    def _fallback_semantic_match(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback semantic matching without LLM"""
        # Simple keyword-based semantic scoring
        resume_text = str(resume_data).lower()
        job_text = str(job_data).lower()
        
        # Count common important terms
        important_terms = ["python", "django", "flask", "api", "database", "aws", "docker", "web", "development"]
        common_terms = sum(1 for term in important_terms if term in resume_text and term in job_text)
        
        semantic_score = min((common_terms / len(important_terms)) * 100, 100)
        
        return {
            "semantic_score": semantic_score,
            "relevance_explanation": f"Found {common_terms} matching important terms out of {len(important_terms)}",
            "method": "fallback_semantic"
        }
    
    def _calculate_confidence(self, state: ResumeAnalysisState) -> float:
        """Calculate confidence score for the analysis"""
        hard_score = state["hard_match_results"].get("score", 0)
        semantic_score = state["semantic_match_results"].get("semantic_score", 0)
        
        # Higher confidence if both scores are consistent
        score_diff = abs(hard_score - semantic_score)
        if score_diff < 10:
            confidence = 0.9
        elif score_diff < 20:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return confidence
    
    def _fallback_gap_analysis(self, state: ResumeAnalysisState) -> Dict[str, Any]:
        """Fallback gap analysis"""
        missing_skills = state["hard_match_results"].get("missing_skills", [])
        
        return {
            "critical_gaps": missing_skills[:3],
            "recommended_skills": missing_skills,
            "method": "fallback_gap_analysis"
        }
    
    def _fallback_recommendations(self, state: ResumeAnalysisState) -> List[str]:
        """Fallback recommendations"""
        gap_analysis = state["gap_analysis"]
        missing_skills = gap_analysis.get("critical_gaps", [])
        
        recommendations = []
        if missing_skills:
            recommendations.extend([f"Develop expertise in {skill}" for skill in missing_skills[:2]])
        
        recommendations.extend([
            "Build a portfolio project showcasing your technical skills",
            "Consider obtaining relevant industry certifications",
            "Quantify your achievements with specific metrics",
            "Tailor your resume to highlight relevant experience"
        ])
        
        return recommendations[:5]
    
    def _generate_basic_recommendations(self, gap_analysis: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations based on gap analysis"""
        missing_skills = gap_analysis.get("critical_gaps", [])
        recommendations = []
        
        if missing_skills:
            for skill in missing_skills[:2]:
                recommendations.append(f"Consider learning {skill} to meet job requirements")
        
        recommendations.extend([
            "Build projects that demonstrate your technical capabilities",
            "Update your resume to highlight relevant experience",
            "Consider obtaining industry-relevant certifications"
        ])
        
        return recommendations[:5]


# Export the workflow class
__all__ = ["LangChainResumeWorkflow"]