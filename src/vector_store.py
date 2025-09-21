"""
Vector Store Implementation
Implements required vector storage and semantic search using FAISS and ChromaDB.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging

try:
    import faiss
    import chromadb
    from sentence_transformers import SentenceTransformer
    from chromadb.config import Settings
except ImportError as e:
    logging.warning(f"Vector store dependencies not available: {e}")
    faiss = None
    chromadb = None
    SentenceTransformer = None

class VectorStore:
    """Vector store for embeddings and semantic search"""
    
    def __init__(self, use_chroma: bool = True):
        """Initialize vector store"""
        self.use_chroma = use_chroma and chromadb is not None
        self.embedding_model = None
        self.faiss_index = None
        self.chroma_client = None
        self.chroma_collection = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector stores"""
        
        # Initialize sentence transformer for embeddings
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Loaded sentence transformer model")
            except Exception as e:
                logging.error(f"Failed to load embedding model: {e}")
        
        # Initialize ChromaDB if available
        if self.use_chroma:
            try:
                self.chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory="./chroma_db"
                ))
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name="resume_job_embeddings"
                )
                logging.info("Initialized ChromaDB")
            except Exception as e:
                logging.error(f"Failed to initialize ChromaDB: {e}")
                self.use_chroma = False
        
        # Initialize FAISS as fallback
        if faiss and not self.use_chroma:
            try:
                # Create FAISS index for 384-dimensional vectors (MiniLM output)
                self.faiss_index = faiss.IndexFlatIP(384)  # Inner product for cosine similarity
                logging.info("Initialized FAISS index")
            except Exception as e:
                logging.error(f"Failed to initialize FAISS: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for text list"""
        if not self.embedding_model:
            return None
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            return normalized_embeddings
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return None
    
    def add_resume_embeddings(self, resume_id: str, resume_data: Dict[str, Any]):
        """Add resume embeddings to vector store"""
        
        # Create text representations for embedding
        resume_texts = self._extract_resume_texts(resume_data)
        
        if not resume_texts:
            return False
        
        # Generate embeddings
        embeddings = self.generate_embeddings(resume_texts)
        if embeddings is None:
            return False
        
        # Store in ChromaDB
        if self.use_chroma:
            try:
                self.chroma_collection.add(
                    embeddings=embeddings.tolist(),
                    documents=resume_texts,
                    metadatas=[{"type": "resume", "id": resume_id} for _ in resume_texts],
                    ids=[f"resume_{resume_id}_{i}" for i in range(len(resume_texts))]
                )
                return True
            except Exception as e:
                logging.error(f"ChromaDB storage failed: {e}")
        
        # Fallback to FAISS
        elif self.faiss_index is not None:
            try:
                self.faiss_index.add(embeddings)
                return True
            except Exception as e:
                logging.error(f"FAISS storage failed: {e}")
        
        return False
    
    def semantic_search(self, job_data: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search between job and resume"""
        
        # Extract texts for comparison
        job_texts = self._extract_job_texts(job_data)
        resume_texts = self._extract_resume_texts(resume_data)
        
        if not job_texts or not resume_texts:
            return {"similarity_score": 0.7, "method": "fallback"}
        
        # Generate embeddings
        job_embeddings = self.generate_embeddings(job_texts)
        resume_embeddings = self.generate_embeddings(resume_texts)
        
        if job_embeddings is None or resume_embeddings is None:
            return {"similarity_score": 0.7, "method": "fallback"}
        
        # Calculate cosine similarity
        similarities = []
        for job_emb in job_embeddings:
            for resume_emb in resume_embeddings:
                similarity = np.dot(job_emb, resume_emb)
                similarities.append(similarity)
        
        # Average similarity score
        avg_similarity = np.mean(similarities)
        
        # Convert to 0-100 score
        score = (avg_similarity + 1) / 2 * 100  # Scale from [-1,1] to [0,100]
        
        return {
            "similarity_score": min(max(score, 0), 100),
            "method": "embedding_cosine_similarity",
            "num_comparisons": len(similarities),
            "max_similarity": max(similarities),
            "min_similarity": min(similarities)
        }
    
    def find_similar_resumes(self, job_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar resumes for a job using vector search"""
        
        if not self.use_chroma or not self.chroma_collection:
            return []
        
        # Extract job texts
        job_texts = self._extract_job_texts(job_data)
        if not job_texts:
            return []
        
        # Generate job embeddings
        job_embeddings = self.generate_embeddings(job_texts)
        if job_embeddings is None:
            return []
        
        try:
            # Query ChromaDB for similar resumes
            results = self.chroma_collection.query(
                query_embeddings=job_embeddings[0].tolist(),  # Use first job embedding
                n_results=top_k,
                where={"type": "resume"}
            )
            
            similar_resumes = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similar_resumes.append({
                    "resume_id": metadata.get("id", f"unknown_{i}"),
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "content_preview": doc[:200],
                    "metadata": metadata
                })
            
            return similar_resumes
            
        except Exception as e:
            logging.error(f"Similar resume search failed: {e}")
            return []
    
    def _extract_resume_texts(self, resume_data: Dict[str, Any]) -> List[str]:
        """Extract relevant texts from resume for embedding"""
        texts = []
        
        # Personal info and summary
        personal = resume_data.get('personal_info', {})
        if personal.get('title'):
            texts.append(f"Professional Title: {personal['title']}")
        
        summary = resume_data.get('summary', '')
        if summary:
            texts.append(f"Summary: {summary}")
        
        # Skills
        all_skills = []
        for category, skills in resume_data.get('skills', {}).items():
            all_skills.extend(skills)
        if all_skills:
            texts.append(f"Technical Skills: {', '.join(all_skills)}")
        
        # Experience
        for exp in resume_data.get('experience', []):
            if exp.get('title'):
                exp_text = f"Experience: {exp['title']}"
                if exp.get('description'):
                    exp_text += f" - {exp['description']}"
                texts.append(exp_text)
        
        # Education
        for edu in resume_data.get('education', []):
            if edu.get('degree'):
                texts.append(f"Education: {edu['degree']}")
        
        # Projects
        for proj in resume_data.get('projects', []):
            if proj.get('name'):
                proj_text = f"Project: {proj['name']}"
                if proj.get('description'):
                    proj_text += f" - {proj['description']}"
                texts.append(proj_text)
        
        return texts
    
    def _extract_job_texts(self, job_data: Dict[str, Any]) -> List[str]:
        """Extract relevant texts from job description for embedding"""
        texts = []
        
        # Job title
        if job_data.get('job_title'):
            texts.append(f"Position: {job_data['job_title']}")
        
        # Required skills
        required_skills = job_data.get('required_skills', [])
        if required_skills:
            texts.append(f"Required Skills: {', '.join(required_skills)}")
        
        # Preferred skills
        preferred_skills = job_data.get('preferred_skills', [])
        if preferred_skills:
            texts.append(f"Preferred Skills: {', '.join(preferred_skills)}")
        
        # Responsibilities
        responsibilities = job_data.get('responsibilities', [])
        if responsibilities:
            texts.append(f"Key Responsibilities: {'; '.join(responsibilities[:3])}")
        
        # Experience requirements
        exp_req = job_data.get('experience_requirements', {})
        if exp_req.get('years_required'):
            texts.append(f"Experience Required: {exp_req['years_required']} years")
        
        # Education requirements
        edu_req = job_data.get('education_requirements', {})
        if edu_req.get('degree_level'):
            texts.append(f"Education Required: {edu_req['degree_level']}")
        
        return texts
    
    def cleanup(self):
        """Cleanup vector store resources"""
        if self.chroma_client:
            try:
                # Persist ChromaDB data
                self.chroma_client.persist()
            except Exception as e:
                logging.error(f"ChromaDB cleanup failed: {e}")