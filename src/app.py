"""
Automated Resume Relevance Check System - Main Application
A Streamlit web application for analyzing resume relevance against job descriptions.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser import ResumeParser
from job_analyzer import JobAnalyzer
from relevance_scorer import RelevanceScorer
from workflow_manager import WorkflowManager
from utils import save_uploaded_file, get_file_type

# Helper functions for file processing
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except ImportError:
            raise Exception("PDF processing libraries not available. Install PyMuPDF or pdfplumber.")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        import docx2txt
        return docx2txt.process(file_path)
    except ImportError:
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            raise Exception("DOCX processing libraries not available. Install docx2txt or python-docx.")
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Resume Relevance Checker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.score-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1e3c72;
    margin: 1rem 0;
}
.high-score { border-left-color: #28a745; }
.medium-score { border-left-color: #ffc107; }
.low-score { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🎯 Resume Relevance Checker</h1><p>Intelligent candidate-job matching powered by AI</p></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        mode = st.radio("Choose Mode:", ["Single Analysis", "Batch Processing", "Analytics Dashboard"])
        
        st.header("⚙️ Settings")
        enable_advanced = st.checkbox("Enable Advanced Analysis", value=True, help="Use enhanced semantic analysis when available")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        
        if enable_advanced:
            api_key = st.text_input("API Key", type="password", help="Required for advanced semantic matching")
        else:
            api_key = None
    
    if mode == "Single Analysis":
        single_analysis_mode(enable_advanced, api_key, confidence_threshold)
    elif mode == "Batch Processing":
        batch_processing_mode()
    else:
        analytics_dashboard_mode()

def single_analysis_mode(enable_advanced, api_key, confidence_threshold):
    """Single resume analysis mode"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        job_input_method = st.radio("Input Method:", ["Text Input", "File Upload"])
        
        job_file = None  # Initialize job_file
        if job_input_method == "Text Input":
            job_description = st.text_area(
                "Paste job description here:",
                height=300,
                placeholder="Enter the complete job description including required skills, qualifications, and responsibilities..."
            )
        else:
            job_file = st.file_uploader("Upload Job Description", type=['txt', 'pdf', 'docx'])
            job_description = ""
            if job_file:
                try:
                    # Process job file based on type
                    if job_file.type == "text/plain":
                        # Text file
                        job_description = str(job_file.read(), "utf-8")
                    elif job_file.type == "application/pdf":
                        # PDF file - save temporarily and parse
                        temp_job_path = save_uploaded_file(job_file, "uploads")
                        job_description = extract_text_from_pdf(temp_job_path)
                        if os.path.exists(temp_job_path):
                            os.remove(temp_job_path)
                    elif job_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                        # DOCX file - save temporarily and parse
                        temp_job_path = save_uploaded_file(job_file, "uploads")
                        job_description = extract_text_from_docx(temp_job_path)
                        if os.path.exists(temp_job_path):
                            os.remove(temp_job_path)
                    else:
                        # Try to read as text
                        job_description = str(job_file.read(), "utf-8")
                    
                    st.success(f"✅ Job description loaded from {job_file.name}")
                    st.info(f"📄 Content preview: {job_description[:200]}...")
                    
                except Exception as e:
                    st.error(f"❌ Failed to process job file: {str(e)}")
                    job_description = ""
    
    with col2:
        st.subheader("📄 Resume Upload")
        resume_file = st.file_uploader("Upload Resume", type=['pdf', 'docx'], help="Supported formats: PDF, DOCX")
        
        if resume_file:
            st.success(f"✅ File uploaded: {resume_file.name}")
            file_type = get_file_type(resume_file.name)
            st.info(f"File type: {file_type.upper()}")
    
    # Analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Resume", type="primary"):
            if not job_description and not job_file:
                st.error("Please provide a job description!")
                return
            if not resume_file:
                st.error("Please upload a resume!")
                return
            if enable_advanced and not api_key:
                st.error("Please provide API key for advanced semantic matching!")
                return
            
            # Perform analysis
            with st.spinner("Analyzing resume relevance... This may take a few moments."):
                results = perform_analysis(job_description, job_file, resume_file, enable_advanced, api_key, confidence_threshold)
                st.session_state.analysis_results = results
    
    # Display results
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)

def perform_analysis(job_description, job_file, resume_file, enable_advanced=True, api_key=None, confidence_threshold=0.7):
    """Perform the complete analysis pipeline"""
    try:
        # Process job description if it comes from a file
        if not job_description and job_file:
            try:
                if job_file.type == "text/plain":
                    job_description = str(job_file.read(), "utf-8")
                elif job_file.type == "application/pdf":
                    temp_job_path = save_uploaded_file(job_file, "uploads")
                    job_description = extract_text_from_pdf(temp_job_path)
                    if os.path.exists(temp_job_path):
                        os.remove(temp_job_path)
                elif job_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                    temp_job_path = save_uploaded_file(job_file, "uploads")
                    job_description = extract_text_from_docx(temp_job_path)
                    if os.path.exists(temp_job_path):
                        os.remove(temp_job_path)
                else:
                    job_description = str(job_file.read(), "utf-8")
                st.success(f"✅ Job description processed from {job_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to process job file: {str(e)}")
                return None
        
        # Initialize components
        resume_parser = ResumeParser()
        job_analyzer = JobAnalyzer()
        
        # Initialize workflow manager with fallbacks
        workflow_manager = WorkflowManager(api_key if enable_advanced else None)
        
        # Display workflow status
        status = workflow_manager.get_workflow_status()
        if status['langchain_available']:
            st.success("✅ Advanced LangChain workflow available")
        elif status['semantic_matcher_available']:
            st.info("ℹ️ Semantic analysis available (LangChain fallback)")
        else:
            st.warning("⚠️ Using traditional analysis only")
        
        # Save uploaded file
        saved_resume_path = save_uploaded_file(resume_file, "uploads")
        
        # Parse resume
        resume_data = resume_parser.parse_resume(saved_resume_path)
        
        # Analyze job description
        job_data = job_analyzer.analyze_job_description(job_description)
        
        # Calculate relevance scores using workflow manager
        scores = workflow_manager.run_analysis(resume_data, job_data)
        
        # Add vector similarity if available
        vector_results = workflow_manager.get_vector_similarity(resume_data, job_data)
        scores['vector_similarity'] = vector_results.get('similarity_score', 0)
        scores['vector_method'] = vector_results.get('method', 'unavailable')
        
        # Prepare final results
        results = {
            'timestamp': datetime.now().isoformat(),
'resume_file': resume_file.name,
            'resume_path': saved_resume_path,
            'resume_data': resume_data,
            'job_data': job_data,
            'scores': scores,
            'advanced_enabled': enable_advanced,
            'verdict': get_verdict(scores.get('overall_score', 0))
        }
        
        # Persist analysis for admin dashboard
        try:
            save_dir = os.path.join('data', 'analyses')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            # Non-fatal: show a warning but do not block the flow
            st.warning(f"Could not save analysis to disk: {e}")
        
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def display_analysis_results(results):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Overall score card
    overall_score = results['scores'].get('overall_score', 0)
    # Compute verdict from the displayed score to avoid any mismatch
    verdict = get_verdict(overall_score)
    
    score_class = "high-score" if overall_score >= 80 else "medium-score" if overall_score >= 50 else "low-score"
    
    st.markdown(f"""
    <div class=\"score-card {score_class}\">
        <h2>Overall Relevance Score: {overall_score:.1f}/100</h2>
        <h3>Verdict: {verdict}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Analysis status (workflow and vector similarity)
    workflow_used = results['scores'].get('workflow_used', 'unknown')
    vector_method = results['scores'].get('vector_method', 'unavailable')
    vector_similarity = results['scores'].get('vector_similarity', None)
    if vector_similarity is not None:
        st.caption(f"Workflow: {workflow_used} • Vector: {vector_method} ({vector_similarity:.1f})")
    else:
        st.caption(f"Workflow: {workflow_used} • Vector: {vector_method}")

    # Surface errors if any to help diagnose fixed 50 scores
    if results['scores'].get('workflow_used') == 'emergency_fallback' or results['scores'].get('error'):
        st.error(f"Analysis fallback used. Details: {results['scores'].get('error', 'Unknown error')}")
    
    # Detailed scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Category Breakdown")
        
        category_scores = results['scores'].get('category_scores', {})
        for category, score in category_scores.items():
            st.metric(
                label=category.replace('_', ' ').title(),
                value=f"{score:.1f}%",
                delta=f"{score - 50:.1f}" if score != 50 else None
            )
    
    with col2:
        st.subheader("🎯 Skills Analysis")
        
        matched_skills = results['scores'].get('matched_skills', [])
        missing_skills = results['scores'].get('missing_skills', [])
        
        if matched_skills:
            st.success("✅ **Matched Skills:**")
            for skill in matched_skills[:10]:  # Show top 10
                st.write(f"• {skill}")
        
        if missing_skills:
            st.warning("⚠️ **Missing Skills:**")
            for skill in missing_skills[:10]:  # Show top 10
                st.write(f"• {skill}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = results['scores'].get('recommendations', [])
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.info(f"{i}. {rec}")
    else:
        st.info("No specific recommendations available.")
    
    # Export results
    st.markdown("---")
    if st.button("📥 Export Results", help="Download analysis results as JSON"):
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download JSON Report",
            data=results_json,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def batch_processing_mode():
    """Batch processing for multiple resumes"""
    st.subheader("📁 Batch Processing")
    st.info("Upload multiple resumes to analyze against a single job description.")
    
    # Job description input with upload option
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        job_input_method = st.radio("Input Method:", ["Text Input", "File Upload"], key="batch_job_input")
        
        job_file = None
        if job_input_method == "Text Input":
            job_description = st.text_area(
                "Paste job description here:",
                height=200,
                placeholder="Enter the complete job description...",
                key="batch_job_text"
            )
        else:
            job_file = st.file_uploader(
                "Upload Job Description", 
                type=['txt', 'pdf', 'docx'],
                key="batch_job_file"
            )
            job_description = ""
            if job_file:
                try:
                    if job_file.type == "text/plain":
                        job_description = str(job_file.read(), "utf-8")
                    elif job_file.type == "application/pdf":
                        temp_job_path = save_uploaded_file(job_file, "uploads")
                        job_description = extract_text_from_pdf(temp_job_path)
                        if os.path.exists(temp_job_path):
                            os.remove(temp_job_path)
                    elif job_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                        temp_job_path = save_uploaded_file(job_file, "uploads")
                        job_description = extract_text_from_docx(temp_job_path)
                        if os.path.exists(temp_job_path):
                            os.remove(temp_job_path)
                    else:
                        job_description = str(job_file.read(), "utf-8")
                    st.success(f"✅ Job description loaded from {job_file.name}")
                except Exception as e:
                    st.error(f"❌ Failed to process job file: {str(e)}")
                    job_description = ""
    
    with col2:
        st.subheader("📄 Resume Upload")
        resume_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Select multiple resume files for batch processing",
            key="batch_resume_files"
        )
        
        if resume_files:
            st.success(f"✅ {len(resume_files)} resume(s) uploaded")
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        enable_advanced = st.checkbox("Enable Advanced Analysis", value=False, key="batch_advanced")
    with col2:
        if enable_advanced:
            api_key = st.text_input("API Key", type="password", key="batch_api_key")
        else:
            api_key = None
    
    # Process batch
    if st.button("🔄 Process Batch") and (job_description or job_file) and resume_files:
        st.info(f"Processing {len(resume_files)} resumes...")
        
        # Process each resume with real analysis
        results = []
        progress_bar = st.progress(0)
        
        for i, resume_file in enumerate(resume_files):
            progress_bar.progress((i + 1) / len(resume_files))
            
            try:
                # Perform real analysis
                analysis_result = perform_analysis(
                    job_description, job_file, resume_file, 
                    enable_advanced, api_key, 0.7
                )
                
                if analysis_result:
                    overall = analysis_result['scores'].get('overall_score', 0)
                    verdict = get_verdict(overall)
                    results.append({
                        'Resume': resume_file.name,
                        'Overall Score': f"{overall:.1f}",
                        'Verdict': verdict,
                        'Matched Skills': len(analysis_result['scores'].get('matched_skills', [])),
                        'Missing Skills': len(analysis_result['scores'].get('missing_skills', []))
                    })
                else:
                    results.append({
                        'Resume': resume_file.name,
                        'Overall Score': '0.0',
                        'Verdict': 'Analysis Failed',
                        'Matched Skills': 0,
                        'Missing Skills': 0
                    })
            except Exception as e:
                st.error(f"Failed to process {resume_file.name}: {str(e)}")
                results.append({
                    'Resume': resume_file.name,
                    'Overall Score': '0.0',
                    'Verdict': 'Processing Error',
                    'Matched Skills': 0,
                    'Missing Skills': 0
                })
        
        # Display batch results
        st.subheader("📊 Batch Results")
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, width=800)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_score = sum(float(r['Overall Score']) for r in results) / len(results)
                st.metric("Average Score", f"{avg_score:.1f}")
            with col2:
                high_suitability = sum(1 for r in results if 'High' in r['Verdict'])
                st.metric("High Suitability", high_suitability)
            with col3:
                total_processed = len(results)
                st.metric("Total Processed", total_processed)
        else:
            st.error("No results to display")

def analytics_dashboard_mode():
    """Analytics and insights dashboard"""
    st.subheader("📊 Analytics Dashboard")
    st.info("Analytics features will be implemented in future versions.")
    
    # Mock analytics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes Processed", "1,234", "+12%")
    with col2:
        st.metric("Average Score", "67.8", "+2.1")
    with col3:
        st.metric("High Suitability Rate", "23%", "+5%")
    with col4:
        st.metric("Time Saved (hours)", "156", "+18")

def get_verdict(score):
    """Get verdict based on score"""
    if score >= 80:
        return "High Suitability"
    elif score >= 50:
        return "Medium Suitability"
    else:
        return "Low Suitability"

if __name__ == "__main__":
    main()