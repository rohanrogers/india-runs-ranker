import json
import gzip
import sqlite3
import argparse
import numpy as np
from datetime import date, datetime
from sentence_transformers import SentenceTransformer

# --- CONSTRAINTS & WEIGHTS ---
JD_SKILLS = {
    'embeddings': 3.0, 'faiss': 3.0, 'information retrieval': 3.0,
    'sentence transformers': 3.0, 'vector search': 2.5,
    'pinecone': 2.5, 'weaviate': 2.5, 'qdrant': 2.5,
    'elasticsearch': 2.0, 'opensearch': 2.0,
    'hugging face transformers': 2.0, 'mlflow': 2.0,
    'python': 1.5, 'pytorch': 1.5, 'recommendation systems': 2.0,
    'feature engineering': 1.5, 'mlops': 1.5, 'xgboost': 2.0, 'lightgbm': 2.0
}

PROFICIENCY_WEIGHT = {'expert': 1.5, 'advanced': 1.2, 'intermediate': 1.0, 'beginner': 0.4}
SERVICES = {'tcs', 'infosys', 'wipro', 'cognizant', 'hcl', 'tech mahindra', 'mindtree'}

# THE DETERMINISTIC ANCHOR: Guarantees Stage 3 reproducibility regardless of execution date.
ANCHOR_DATE = date(2026, 6, 14)


def setup_database(db_path="candidates.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS candidates")
    cursor.execute("""
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY,
            is_honeypot INTEGER,
            ai_depth_score REAL,
            behavioral_multiplier REAL,
            company_penalty REAL,
            location_bonus REAL,
            raw_yoe REAL,
            primary_reasoning_data TEXT,
            semantic_vector BLOB
        )
    """)
    conn.commit()
    return conn


def extract_primary_reasoning(profile, skills, signals):
    top_skills = sorted(
        [s for s in skills if s['name'].lower() in JD_SKILLS],
        key=lambda x: JD_SKILLS[x['name'].lower()] * x['duration_months'],
        reverse=True
    )[:2]

    return json.dumps({
        "current_title": profile.get('current_title', 'Engineer'),
        "current_company": profile.get('current_company', 'a company'),
        "top_skills": [{"name": s['name'], "dur": s['duration_months']} for s in top_skills],
        "location": profile.get('location', 'Unknown'),
        "yoe": profile.get('years_of_experience', 0),
        "notice_days": signals.get('notice_period_days', 90),
        "response_rate": signals.get('recruiter_response_rate', 0.0)
    })


def detect_honeypot(candidate, signals):
    """
    Multi-pattern honeypot detection.
    Checks for date inversions, salary inversions, impossible expert claims,
    and excessive expert skill counts.
    """
    # Pattern 1: last_active_date before signup_date (chronological impossibility)
    last_act_str = signals.get('last_active_date', '')
    signup_str = signals.get('signup_date', '')
    date_hp = (last_act_str and signup_str and last_act_str < signup_str)

    # Pattern 2: salary min exceeds salary max (inverted range)
    salary = signals.get('expected_salary_range_inr_lpa', {})
    sal_hp = salary.get('min', 0) > salary.get('max', 999)

    # Pattern 3: expert proficiency claimed with near-zero usage duration
    skills = candidate.get('skills', [])
    expert_zero_count = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) <= 2
    )
    skill_hp = expert_zero_count >= 3

    # Pattern 4: impossibly many expert-level skills (statistical outlier)
    total_expert = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert'
    )
    expert_count_hp = total_expert >= 10

    return 1 if (date_hp or sal_hp or skill_hp or expert_count_hp) else 0


def build_database(input_file="data/candidates.jsonl.gz", db_path="candidates.db"):
    print("Loading embedding model (BAAI/bge-small-en-v1.5) on CPU...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

    conn = setup_database(db_path)
    cursor = conn.cursor()

    processed = 0
    inserted = 0

    BATCH_SIZE = 2048
    batch_texts = []
    batch_metadata = []

    # Detect file format: gzipped JSONL vs plain JSON array
    if input_file.endswith('.json'):
        print(f"Reading plain JSON array from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as jf:
            candidates_list = json.load(jf)
        lines_iter = iter([json.dumps(c) for c in candidates_list])
    else:
        print(f"Reading gzipped JSONL from {input_file}...")
        lines_iter = gzip.open(input_file, 'rt', encoding='utf-8')

    print("Beginning Offline ETL Pipeline...")
    for line in lines_iter:
        if not line.strip():
            continue
        candidate = json.loads(line)
        processed += 1

        c_id = candidate['candidate_id']
        profile = candidate['profile']
        signals = candidate.get('redrob_signals', {})

        yoe = profile.get('years_of_experience', 0)
        if yoe < 4 or yoe > 12:
            continue

        # --- HONEYPOT DETECTION ---
        is_hp = detect_honeypot(candidate, signals)

        # --- DEPTH COMPILATION & ASSESSMENTS ---
        depth_score = 0.0
        for skill in candidate.get('skills', []):
            s_name = skill['name'].lower()
            if s_name in JD_SKILLS:
                depth_score += (JD_SKILLS[s_name] * PROFICIENCY_WEIGHT.get(skill['proficiency'], 1.0) * skill['duration_months'])

        assessments = signals.get('skill_assessment_scores', {})
        for skill_name, score in assessments.items():
            s_name = skill_name.lower()
            if s_name in JD_SKILLS:
                depth_score += (score / 100.0) * JD_SKILLS[s_name] * 10.0

        # --- MODIFIERS ---
        company_penalty = 0.8 if profile.get('current_company', '').lower() in SERVICES else 1.0
        loc_str = profile.get('location', '').lower()
        loc_bonus = 1.1 if ('pune' in loc_str or 'noida' in loc_str) else (1.05 if signals.get('willing_to_relocate') else 1.0)

        # --- BEHAVIORAL MATH ---
        response_rate = signals.get('recruiter_response_rate', 0.0)
        days_inactive = 0
        last_act_str = signals.get('last_active_date', '')
        if last_act_str:
            try:
                last_act = datetime.strptime(last_act_str, '%Y-%m-%d').date()
                days_inactive = (ANCHOR_DATE - last_act).days
            except ValueError:
                pass
        recency_score = max(0.0, 1.0 - (days_inactive / 180.0))
        notice_score = 1.0 if signals.get('notice_period_days', 90) <= 30 else 0.85
        otw_bonus = 1.1 if signals.get('open_to_work_flag') else 1.0

        behav_mult = ((response_rate * 0.5) + (recency_score * 0.5)) * notice_score * otw_bonus

        # --- BATCH PREPARATION ---
        career = candidate.get('career_history', [])
        recent_descs = " ".join([job.get('description', '') for job in career[:3]])
        semantic_text = f"{profile.get('summary', '')} {recent_descs}"

        r_data = extract_primary_reasoning(profile, candidate.get('skills', []), signals)

        batch_texts.append(semantic_text)
        batch_metadata.append((c_id, is_hp, depth_score, behav_mult, company_penalty, loc_bonus, yoe, r_data))

        # --- HARDWARE BATCH EXECUTION ---
        if len(batch_texts) >= BATCH_SIZE:
            vectors = model.encode(batch_texts, normalize_embeddings=True)
            for i, vec in enumerate(vectors):
                cursor.execute("""
                    INSERT INTO candidates
                    (candidate_id, is_honeypot, ai_depth_score, behavioral_multiplier, company_penalty, location_bonus, raw_yoe, primary_reasoning_data, semantic_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (*batch_metadata[i], vec.astype(np.float32).tobytes()))
            conn.commit()
            inserted += len(batch_texts)
            batch_texts.clear()
            batch_metadata.clear()
            print(f"Processed {processed} rows. {inserted} valid profiles loaded...")

    if batch_texts:
        vectors = model.encode(batch_texts, normalize_embeddings=True)
        for i, vec in enumerate(vectors):
            cursor.execute("""
                INSERT INTO candidates
                (candidate_id, is_honeypot, ai_depth_score, behavioral_multiplier, company_penalty, location_bonus, raw_yoe, primary_reasoning_data, semantic_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*batch_metadata[i], vec.astype(np.float32).tobytes()))
        conn.commit()
        inserted += len(batch_texts)

    # Close file handle if it was a gzip file object
    if hasattr(lines_iter, 'close'):
        lines_iter.close()

    conn.close()
    print(f"Pipeline complete. {inserted} highly contextualized records compiled securely to {db_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline ETL Compiler: ingest candidates into SQLite with embeddings")
    parser.add_argument("--input", default="data/candidates.jsonl.gz",
                        help="Path to candidates file (.jsonl.gz or .json)")
    parser.add_argument("--db", default="candidates.db",
                        help="Output SQLite database path")
    args = parser.parse_args()
    build_database(input_file=args.input, db_path=args.db)
