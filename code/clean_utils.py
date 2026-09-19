"""Cleaning & parsing utilities for The Corporate Heist dataset."""
import re
import numpy as np
import pandas as pd

CUR_YEAR = 2026

NUM_RE = re.compile(r'-?\d+\.?\d*')


def first_number(x):
    if pd.isna(x):
        return np.nan
    m = NUM_RE.search(str(x))
    return float(m.group()) if m else np.nan


def parse_experience(x):
    """Returns years of experience as float."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    n = first_number(s)
    if np.isnan(n):
        return np.nan
    if 'month' in s or ' mo' in s:
        return n / 12.0
    return n  # years, "8+ yrs", ">20" etc already captured by first_number


def parse_aptitude(x):
    """Normalize aptitude score to a 0-10 scale."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    n = first_number(s)
    if np.isnan(n):
        return np.nan
    if '%' in s:
        return n / 10.0
    if '/10' in s:
        return n
    return n  # already 0-10 scale


def parse_rating(x):
    """Normalize last_rating to a 1-5 scale."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    text_map = {
        'outstanding': 5.0,
        'exceeds expectations': 4.0,
        'meets expectations': 3.0,
        'needs improvement': 2.0,
        'unsatisfactory': 1.0,
        'new joiner - not rated': np.nan,
        'not rated': np.nan,
    }
    if s in text_map:
        return text_map[s]
    n = first_number(s)
    return n if not np.isnan(n) else np.nan


def parse_notice_days(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if 'immediate' in s or 'available now' in s:
        return 0.0
    n = first_number(s)
    if np.isnan(n):
        return np.nan
    if 'month' in s:
        return n * 30.0
    return n  # days


def parse_last_job_change(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == 'never':
        return 0.0
    if s.startswith('>'):
        n = first_number(s)
        return (n + 1) if not np.isnan(n) else 5.0
    n = first_number(s)
    return n


def parse_ctc_lakhs(x):
    """Normalize CTC to INR lakhs (1 lakh = 100,000 INR)."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(',', '').replace('₹', '')
    n = first_number(s)
    if np.isnan(n):
        return np.nan
    if 'cr' in s:
        return n * 100.0
    if 'lpa' in s or 'lakh' in s or s.strip().endswith('l'):
        return n
    # plain rupee figure, e.g. 1692000
    if n > 10000:
        return n / 100000.0
    return n


def parse_technical_assessment(x):
    """Normalize to 0-100 scale."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    n = first_number(s)
    if np.isnan(n):
        return np.nan
    if n <= 1.0:
        return n * 100.0
    return n


def parse_awards(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().lower()
    if s in ('-', '0', 'none', 'no'):
        return 0.0
    n = first_number(s)
    if not np.isnan(n):
        return max(n, 1.0) if 'award' in s or s.replace('.', '', 1).isdigit() else n
    return 1.0  # any textual mention like "Yes - Spot Award"


def parse_kpi_met(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    return 1.0 if s in ('y', 'yes', '1', 'true') else 0.0


def parse_binary_yesno(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    return 1.0 if s in ('y', 'yes', '1', 'true') else 0.0


def parse_code_contributions(x):
    """Returns numeric count of public code contributions, NaN if not tracked/missing."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if 'not tracked' in s:
        return np.nan
    n = first_number(s)
    return n


def canon_institute(x):
    """Canonicalize institute names to merge spelling/abbreviation variants."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    s = re.sub(r'[.\-,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # common abbreviation expansions
    repl = [
        (r'\bIIT\s*-?\s*D\b', 'IIT DELHI'),
        (r'\bIITD\b', 'IIT DELHI'),
        (r'\bIIT\s*-?\s*B\b', 'IIT BOMBAY'),
        (r'\bIITB\b', 'IIT BOMBAY'),
        (r'\bIIT\s*-?\s*K\b', 'IIT KANPUR'),
        (r'\bIITK\b', 'IIT KANPUR'),
        (r'\bIIT\s*-?\s*KGP\b', 'IIT KHARAGPUR'),
        (r'\bIIT\s*-?\s*M\b', 'IIT MADRAS'),
        (r'\bIITM\b', 'IIT MADRAS'),
        (r'\bI I T\b', 'IIT'),
        (r'INDIAN INSTITUTE OF TECHNOLOGY', 'IIT'),
    ]
    for pat, rep in repl:
        s = re.sub(pat, rep, s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


LEADERSHIP_COLLEGES = {
    'IIT DELHI', 'IIT BOMBAY', 'IIT KANPUR', 'IIT MADRAS',
}

SENIOR_TITLE_RE = re.compile(
    r'\b(?:head of|chief|vp\b|vice president|director)\b', re.IGNORECASE
)


def clean_dataframe(df):
    """Apply all parsers and return df with added *_clean numeric/derived columns."""
    df = df.copy()
    df['exp_years'] = df['total_experience'].apply(parse_experience)
    df['aptitude_clean'] = df['aptitude_score'].apply(parse_aptitude)
    df['rating_clean'] = df['last_rating'].apply(parse_rating)
    df['notice_days'] = df['notice_period'].apply(parse_notice_days)
    df['last_job_change_clean'] = df['last_job_change'].apply(parse_last_job_change)
    df['current_ctc_lakhs'] = df['current_ctc'].apply(parse_ctc_lakhs)
    df['expected_ctc_lakhs'] = df['expected_ctc'].apply(parse_ctc_lakhs)
    df['tech_assessment_clean'] = df['technical_assessment'].apply(parse_technical_assessment)
    df['awards_clean'] = df['awards'].apply(parse_awards)
    df['kpi_met_clean'] = df['kpi_met'].apply(parse_kpi_met)
    df['overtime_clean'] = df['overtime_history'].apply(parse_binary_yesno)
    df['enrolled_clean'] = (df['currently_enrolled'].fillna('no_enrollment') != 'no_enrollment').astype(float)
    df['institute_canon'] = df['institute'].apply(canon_institute)
    df['leadership_college'] = df['institute_canon'].isin(LEADERSHIP_COLLEGES).astype(float)

    df['years_since_grad'] = CUR_YEAR - df['graduation_year']
    df['age_at_grad'] = df['age'] - df['years_since_grad']
    df['exp_gap'] = df['exp_years'] - df['years_since_grad']

    df['title_inflated'] = (
        df['current_title'].fillna('').str.contains(SENIOR_TITLE_RE, regex=True)
        & (df['exp_years'] < 6)
    ).astype(float)

    if 'public_code_contributions' in df.columns:
        df['code_contrib_num'] = df['public_code_contributions'].apply(parse_code_contributions)
        df['sustained_coder'] = (df['code_contrib_num'] >= 24).astype(float)
        df['some_coder'] = ((df['code_contrib_num'] > 0) & (df['code_contrib_num'] < 24)).astype(float)

    ctc_ratio = df['expected_ctc_lakhs'] / df['current_ctc_lakhs'].replace(0, np.nan)
    df['ctc_ratio'] = ctc_ratio.clip(upper=10)

    return df
