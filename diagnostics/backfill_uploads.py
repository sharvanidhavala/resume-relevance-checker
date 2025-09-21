#!/usr/bin/env python3
"""
Backfill uploads/: create analysis reports for resumes that don't have JSON reports yet.
Usage:
  python diagnostics/backfill_uploads.py [--jd-file PATH] [--advanced]

If --jd-file is not provided, tries sample_data/job_python_developer.txt, then test_job.txt.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resume_parser import ResumeParser
from job_analyzer import JobAnalyzer
from workflow_manager import WorkflowManager

ANALYSES_DIR = ROOT / 'data' / 'analyses'
UPLOADS_DIR = ROOT / 'uploads'


def load_existing_keys():
    keys = set()
    if not ANALYSES_DIR.exists():
        return keys
    for p in ANALYSES_DIR.glob('analysis_*.json'):
        try:
            data = json.loads(p.read_text(encoding='utf-8', errors='ignore'))
            rp = data.get('resume_path') or ''
            rf = data.get('resume_file') or ''
            if rp:
                keys.add(str(Path(rp).resolve()).lower())
            if rf:
                keys.add(rf.lower())
        except Exception:
            pass
    return keys


def read_jd_text(jd_file: Path | None) -> str:
    if jd_file and jd_file.exists():
        return jd_file.read_text(encoding='utf-8', errors='ignore')
    # Fallbacks
    for cand in [ROOT / 'sample_data' / 'job_python_developer.txt', ROOT / 'test_job.txt']:
        if cand.exists():
            return cand.read_text(encoding='utf-8', errors='ignore')
    raise FileNotFoundError('No JD provided and no sample JD found.')


def verdict_from_score(score: float) -> str:
    if score >= 80:
        return 'High Suitability'
    if score >= 50:
        return 'Medium Suitability'
    return 'Low Suitability'


def backfill(jd_text: str, enable_advanced: bool = False, api_key: str | None = None) -> dict:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    if not UPLOADS_DIR.exists():
        return {'created': 0, 'skipped': 0, 'message': 'uploads/ does not exist'}

    existing = load_existing_keys()
    parser = ResumeParser()
    analyzer = JobAnalyzer()
    wfm = WorkflowManager(api_key if enable_advanced else None)

    created = 0
    skipped = 0

    for p in sorted(UPLOADS_DIR.glob('*')):
        if p.suffix.lower() not in {'.pdf', '.docx'}:
            continue
        key1 = str(p.resolve()).lower()
        key2 = p.name.lower()
        if key1 in existing or key2 in existing:
            skipped += 1
            continue
        try:
            resume_data = parser.parse_resume(str(p))
            job_data = analyzer.analyze_job_description(jd_text)
            scores = wfm.run_analysis(resume_data, job_data)
            # Try vector similarity (best effort)
            try:
                vec = wfm.get_vector_similarity(resume_data, job_data)
                scores['vector_similarity'] = vec.get('similarity_score', 0)
                scores['vector_method'] = vec.get('method', 'unavailable')
            except Exception:
                pass
            result = {
                'timestamp': datetime.utcnow().isoformat(),
                'resume_file': p.name,
                'resume_path': str(p),
                'resume_data': resume_data,
                'job_data': job_data,
                'scores': scores,
                'advanced_enabled': bool(enable_advanced),
                'verdict': verdict_from_score(scores.get('overall_score', 0))
            }
            out_path = ANALYSES_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
            created += 1
        except Exception as e:
            print(f"[WARN] Failed to backfill {p.name}: {e}")
            skipped += 1
    return {'created': created, 'skipped': skipped, 'message': 'done'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jd-file', type=str, default=None, help='Path to job description text file')
    ap.add_argument('--advanced', action='store_true', help='Enable advanced analysis (requires OPENAI_API_KEY)')
    args = ap.parse_args()

    jd_file = Path(args.jd_file) if args.jd_file else None
    jd_text = read_jd_text(jd_file)
    api_key = os.getenv('OPENAI_API_KEY') if args.advanced else None

    result = backfill(jd_text, enable_advanced=args.advanced, api_key=api_key)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
