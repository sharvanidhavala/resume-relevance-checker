#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from resume_parser import ResumeParser
from job_analyzer import JobAnalyzer
from workflow_manager import WorkflowManager


def main():
    # Load inputs
    job_path = ROOT / 'test_job.txt'
    resume_path = ROOT / 'test_resume.txt'

    if not job_path.exists() or not resume_path.exists():
        print(json.dumps({
            'ok': False,
            'error': 'Missing test files',
            'expected_files': [str(job_path), str(resume_path)]
        }, indent=2))
        return

    job_text = job_path.read_text(encoding='utf-8', errors='ignore')
    resume_text = resume_path.read_text(encoding='utf-8', errors='ignore')

    parser = ResumeParser()
    analyzer = JobAnalyzer()

    # Parse inputs
    resume_data = parser.parse_text(resume_text)
    job_data = analyzer.analyze_job_description(job_text)

    # Use API key from env if present
    api_key = os.getenv('OPENAI_API_KEY')
    wfm = WorkflowManager(api_key=api_key)

    # Check workflow status
    status = wfm.get_workflow_status()

    # Run analysis
    scores = wfm.run_analysis(resume_data, job_data)

    # Try vector similarity (optional)
    try:
        vec = wfm.get_vector_similarity(resume_data, job_data)
    except Exception as e:
        vec = {'error': str(e)}

    out = {
        'ok': True,
        'workflow_status': status,
        'analysis': {
            'workflow_used': scores.get('workflow_used', 'unknown'),
            'overall_score': scores.get('overall_score'),
            'error': scores.get('error'),
            'keys': list(scores.keys())
        },
        'vector': vec
    }

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
