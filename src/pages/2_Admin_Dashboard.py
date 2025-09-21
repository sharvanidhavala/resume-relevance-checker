#!/usr/bin/env python3
"""
Admin Dashboard for Placement Team
- Simple login using environment variables ADMIN_USERNAME and ADMIN_PASSWORD
- Review saved analyses from data/analyses
- View summary metrics and download reports
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# Analysis imports
sys_path_added = False
try:
    # Ensure src/ is on path for imports when running as Streamlit page
    import sys
    SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
        sys_path_added = True
except Exception:
    pass

from resume_parser import ResumeParser
from job_analyzer import JobAnalyzer
from workflow_manager import WorkflowManager
from utils import save_uploaded_file

APP_ROOT = Path(__file__).resolve().parents[1]
ANALYSES_DIR = APP_ROOT / 'data' / 'analyses'


def load_analyses() -> List[Dict[str, Any]]:
    results = []
    if not ANALYSES_DIR.exists():
        return results
    for p in sorted(ANALYSES_DIR.glob('analysis_*.json')):
        try:
            data = json.loads(p.read_text(encoding='utf-8', errors='ignore'))
            results.append(data)
        except Exception:
            pass
    return results


def to_table_rows(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        scores = it.get('scores', {})
        rows.append({
            'Timestamp': it.get('timestamp', ''),
            'Resume': it.get('resume_file', ''),
            'Overall Score': scores.get('overall_score', 0),
            'Verdict': it.get('verdict', ''),
            'Workflow': scores.get('workflow_used', ''),
            'Matched Skills': len(scores.get('matched_skills', [])),
            'Missing Skills': len(scores.get('missing_skills', [])),
        })
    if not rows:
        return pd.DataFrame(columns=[
            'Timestamp','Resume','Overall Score','Verdict','Workflow','Matched Skills','Missing Skills'
        ])
    df = pd.DataFrame(rows)
    # Sort newest first
    return df.sort_values(by='Timestamp', ascending=False, ignore_index=True)


def login_form() -> bool:
    st.subheader('Admin Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    do_login = st.button('Login')

    if do_login:
        admin_user = os.getenv('ADMIN_USERNAME', 'admin')
        admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
        if username == admin_user and password == admin_pass:
            st.session_state.admin_authenticated = True
            st.success('Login successful')
            return True
        else:
            st.error('Invalid credentials')
    return st.session_state.get('admin_authenticated', False)


def extract_text_from_pdf(file_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                return text
        except Exception as e:
            raise Exception(f"PDF parsing failed: {e}")


def extract_text_from_docx(file_path: str) -> str:
    try:
        import docx2txt
        return docx2txt.process(file_path)
    except Exception:
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            return text
        except Exception as e:
            raise Exception(f"DOCX parsing failed: {e}")


def verdict_from_score(score: float) -> str:
    if score >= 80:
        return "High Suitability"
    elif score >= 50:
        return "Medium Suitability"
    return "Low Suitability"


def render_dashboard():
    st.title('Placement Team Dashboard')
    st.caption('Review analyses, scores, and reports')

    # --- Analyze New Resumes (Admin) ---
    st.markdown('---')
    st.subheader('Analyze New Resumes')
    colj, colr = st.columns(2)
    with colj:
        job_input_method = st.radio('Job Description Input', ['Text Input', 'File Upload'], key='admin_job_input')
        job_description = ''
        job_file = None
        if job_input_method == 'Text Input':
            job_description = st.text_area('Paste job description here:', height=200, key='admin_job_text')
        else:
            job_file = st.file_uploader('Upload Job Description', type=['txt', 'pdf', 'docx'], key='admin_job_file')
    with colr:
        resume_files = st.file_uploader('Upload Resumes (PDF/DOCX)', type=['pdf', 'docx'], accept_multiple_files=True, key='admin_resumes')
        st.caption('You can select multiple resume files.')

    colA, colB = st.columns(2)
    with colA:
        enable_advanced = st.checkbox('Enable Advanced Analysis', value=True, key='admin_adv')
    with colB:
        api_key = st.text_input('API Key (required when advanced is enabled)', type='password', key='admin_api_key') if enable_advanced else None

    if st.button('Run Analysis', type='primary'):
        # Validate inputs
        if not job_description and not job_file:
            st.error('Provide a job description (text or upload).')
        elif not resume_files:
            st.error('Upload at least one resume.')
        elif enable_advanced and not api_key:
            st.error('Provide API key or disable Advanced Analysis.')
        else:
            # Prepare job description text
            jd_text = ''
            if job_description:
                jd_text = job_description
            else:
                try:
                    # Save and parse JD file
                    temp_path = save_uploaded_file(job_file, str(APP_ROOT / 'uploads'))
                    if job_file.type == 'text/plain':
                        jd_text = Path(temp_path).read_text(encoding='utf-8', errors='ignore')
                    elif job_file.type == 'application/pdf':
                        jd_text = extract_text_from_pdf(temp_path)
                    else:
                        jd_text = extract_text_from_docx(temp_path)
                except Exception as e:
                    st.error(f'Failed to process job description file: {e}')
                    jd_text = ''

            if not jd_text.strip():
                st.error('Job description is empty after processing.')
            else:
                # Initialize components
                parser = ResumeParser()
                analyzer = JobAnalyzer()
                wfm = WorkflowManager(api_key if enable_advanced else None)

                progress = st.progress(0)
                results_summary = []
                total = len(resume_files) if resume_files else 0

                for i, rf in enumerate(resume_files or []):
                    try:
                        # Save resume to disk
                        saved_resume_path = save_uploaded_file(rf, str(APP_ROOT / 'uploads'))
                        # Parse resume
                        resume_data = parser.parse_resume(saved_resume_path)
                        # Analyze JD
                        job_data = analyzer.analyze_job_description(jd_text)
                        # Run analysis
                        scores = wfm.run_analysis(resume_data, job_data)
                        # Vector similarity (best-effort)
                        try:
                            vec = wfm.get_vector_similarity(resume_data, job_data)
                            scores['vector_similarity'] = vec.get('similarity_score', 0)
                            scores['vector_method'] = vec.get('method', 'unavailable')
                        except Exception:
                            pass
                        # Prepare result
                        result = {
                            'timestamp': pd.Timestamp.utcnow().isoformat(),
                            'resume_file': rf.name,
                            'resume_path': saved_resume_path,
                            'resume_data': resume_data,
                            'job_data': job_data,
                            'scores': scores,
                            'advanced_enabled': bool(enable_advanced),
                            'verdict': verdict_from_score(scores.get('overall_score', 0))
                        }
                        # Persist
                        ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
                        out_path = ANALYSES_DIR / f"analysis_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                        out_path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
                        results_summary.append((rf.name, scores.get('overall_score', 0)))
                    except Exception as e:
                        st.error(f"Failed to analyze {rf.name}: {e}")
                    finally:
                        progress.progress((i + 1) / max(total, 1))

                if results_summary:
                    st.success(f'Processed {len(results_summary)} resume(s).')
                else:
                    st.warning('No results generated.')

    # --- Backfill existing uploads (optional) ---
    st.markdown('---')
    with st.expander('Backfill Existing Uploads (scan uploads/ and create reports)', expanded=False):
        st.caption('Use this to generate reports for resumes already present in uploads/. Provide a JD to analyze against.')
        colx, coly = st.columns(2)
        with colx:
            bf_job_input = st.radio('Backfill JD Input', ['Text Input', 'File Upload'], key='bf_job_input')
            bf_job_text = ''
            bf_job_file = None
            if bf_job_input == 'Text Input':
                bf_job_text = st.text_area('Paste job description for backfill:', height=150, key='bf_job_text')
            else:
                bf_job_file = st.file_uploader('Upload JD for backfill', type=['txt','pdf','docx'], key='bf_job_file')
        with coly:
            bf_advanced = st.checkbox('Enable Advanced Analysis', value=False, key='bf_adv')
            bf_api_key = st.text_input('API Key (if advanced)', type='password', key='bf_api') if bf_advanced else None

        if st.button('Backfill from uploads/', key='bf_run'):
            # Build set of already analyzed resume files (by absolute path or name)
            analyzed = set()
            for it in load_analyses():
                p = it.get('resume_path') or ''
                n = it.get('resume_file') or ''
                if p:
                    analyzed.add(str(Path(p).resolve()).lower())
                if n:
                    analyzed.add(n.lower())

            uploads_dir = APP_ROOT / 'uploads'
            candidates = []
            if uploads_dir.exists():
                for p in uploads_dir.glob('*'):
                    if p.suffix.lower() in ['.pdf', '.docx']:
                        key1 = str(p.resolve()).lower()
                        key2 = p.name.lower()
                        if key1 not in analyzed and key2 not in analyzed:
                            candidates.append(p)

            if not candidates:
                st.info('No new files found in uploads/ to backfill.')
            else:
                # Prepare JD text
                jd_text = ''
                if bf_job_text:
                    jd_text = bf_job_text
                elif bf_job_file is not None:
                    try:
                        temp_path = save_uploaded_file(bf_job_file, str(APP_ROOT / 'uploads'))
                        if bf_job_file.type == 'text/plain':
                            jd_text = Path(temp_path).read_text(encoding='utf-8', errors='ignore')
                        elif bf_job_file.type == 'application/pdf':
                            jd_text = extract_text_from_pdf(temp_path)
                        else:
                            jd_text = extract_text_from_docx(temp_path)
                    except Exception as e:
                        st.error(f'Failed to process JD file for backfill: {e}')
                        jd_text = ''
                if not jd_text.strip():
                    st.error('Provide a valid job description to backfill against.')
                else:
                    parser = ResumeParser()
                    analyzer = JobAnalyzer()
                    wfm = WorkflowManager(bf_api_key if bf_advanced else None)

                    progress = st.progress(0)
                    created = 0
                    total = len(candidates)
                    for i, path in enumerate(candidates):
                        try:
                            resume_data = parser.parse_resume(str(path))
                            job_data = analyzer.analyze_job_description(jd_text)
                            scores = wfm.run_analysis(resume_data, job_data)
                            try:
                                vec = wfm.get_vector_similarity(resume_data, job_data)
                                scores['vector_similarity'] = vec.get('similarity_score', 0)
                                scores['vector_method'] = vec.get('method', 'unavailable')
                            except Exception:
                                pass
                            result = {
                                'timestamp': pd.Timestamp.utcnow().isoformat(),
                                'resume_file': path.name,
                                'resume_path': str(path),
                                'resume_data': resume_data,
                                'job_data': job_data,
                                'scores': scores,
                                'advanced_enabled': bool(bf_advanced),
                                'verdict': verdict_from_score(scores.get('overall_score', 0))
                            }
                            ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
                            out_path = ANALYSES_DIR / f"analysis_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                            out_path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
                            created += 1
                        except Exception as e:
                            st.error(f'Failed to backfill {path.name}: {e}')
                        finally:
                            progress.progress((i + 1) / max(total, 1))
                    if created:
                        st.success(f'Backfilled {created} file(s) from uploads/.')
                    else:
                        st.warning('No reports created.')

    # --- Existing analytics/review ---
    items = load_analyses()
    df = to_table_rows(items)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Analyses', len(df))
    with col2:
        avg = df['Overall Score'].mean() if not df.empty else 0
        st.metric('Average Score', f"{avg:.1f}")
    with col3:
        high = (df['Overall Score'] >= 80).sum() if not df.empty else 0
        st.metric('High Suitability', int(high))
    with col4:
        low = (df['Overall Score'] < 50).sum() if not df.empty else 0
        st.metric('Low Suitability', int(low))

    st.markdown('---')

    # Filters
    verdict_filter = st.multiselect(
        'Filter by Verdict', options=sorted(df['Verdict'].unique()) if not df.empty else [], default=None
    )
    workflow_filter = st.multiselect(
        'Filter by Workflow', options=sorted(df['Workflow'].unique()) if not df.empty else [], default=None
    )

    df_view = df.copy()
    if verdict_filter:
        df_view = df_view[df_view['Verdict'].isin(verdict_filter)]
    if workflow_filter:
        df_view = df_view[df_view['Workflow'].isin(workflow_filter)]

    st.dataframe(df_view, use_container_width=True)

    # Detailed review panel
    st.markdown('---')
    st.subheader('Detailed Review')
    if df_view.empty:
        st.info('No analyses available yet. Ask students to run an analysis on the homepage.')
    else:
        # Build selection list from loaded items to find the matching record
        items = load_analyses()
        options = [f"{it.get('timestamp','')} • {it.get('resume_file','')}" for it in items]
        selected = st.selectbox('Select an analysis to review', options=options, index=0)
        sel_idx = options.index(selected) if selected in options else None
        if sel_idx is not None:
            it = items[sel_idx]
            scores = it.get('scores', {})

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric('Overall Score', f"{scores.get('overall_score', 0):.1f}")
            with colB:
                st.metric('Workflow', scores.get('workflow_used', 'unknown'))
            with colC:
                st.metric('Verdict', it.get('verdict', ''))

            # Resume download / preview
            resume_path = it.get('resume_path')
            st.markdown('### Resume')
            if resume_path and Path(resume_path).exists():
                try:
                    with open(resume_path, 'rb') as rf:
                        st.download_button('Download Resume', data=rf.read(), file_name=Path(resume_path).name)
                except Exception as e:
                    st.warning(f"Unable to prepare download: {e}")
            else:
                st.caption('Stored resume file not found on disk.')

            # Resume text preview
            raw_text = it.get('resume_data', {}).get('raw_text', '')
            if raw_text:
                st.text_area('Resume Text (preview)', raw_text[:5000], height=200)

            # Skills details
            st.markdown('### Skills')
            matched = scores.get('matched_skills', [])
            missing = scores.get('missing_skills', [])
            col1, col2 = st.columns(2)
            with col1:
                st.success('Matched Skills')
                for s in matched[:30]:
                    st.write(f"• {s}")
            with col2:
                st.warning('Missing Skills')
                for s in missing[:30]:
                    st.write(f"• {s}")

            # Recommendations
            recs = scores.get('recommendations', []) or it.get('scores', {}).get('recommendations', [])
            if recs:
                st.markdown('### Recommendations')
                for i, r in enumerate(recs[:10], 1):
                    st.info(f"{i}. {r}")

    # Export
    st.markdown('---')
    csv_data = df_view.to_csv(index=False).encode('utf-8') if not df_view.empty else b''
    st.download_button('Download CSV', data=csv_data, file_name='analyses_report.csv', mime='text/csv')


# Page entry point
st.set_page_config(page_title='Admin Dashboard', page_icon='🛡️', layout='wide')

if not st.session_state.get('admin_authenticated'):
    # Logout resets state
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if login_form():
        st.experimental_rerun()
else:
    with st.sidebar:
        if st.button('Logout'):
            st.session_state.admin_authenticated = False
            st.experimental_rerun()
    render_dashboard()
