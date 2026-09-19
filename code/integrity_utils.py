"""Detect likely-fabricated profiles and duplicate persons."""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def fabrication_score(df):
    """Composite red-flag count. Higher = more internally inconsistent / 'too good to be true'."""
    flags = pd.DataFrame(index=df.index)

    flags['age_grad_bad'] = ((df['age_at_grad'] < 18) | (df['age_at_grad'] > 32)).astype(int)
    flags['exp_exceeds_career'] = (df['exp_gap'] > 3).astype(int)
    flags['negative_exp'] = (df['exp_years'] < 0).astype(int)
    flags['too_many_employers'] = (df['num_employers'] > (df['exp_years'].clip(lower=1) / 0.8)).astype(int)

    # "looks amazing and doesn't add up": near-max on every single axis simultaneously
    near_perfect = (
        (df['tech_assessment_clean'].fillna(0) >= 95)
        & (df['aptitude_clean'].fillna(0) >= 9.5)
        & (df['rating_clean'].fillna(0) >= 5)
        & (df['kpi_met_clean'].fillna(0) == 1)
        & (df['awards_clean'].fillna(0) >= 1)
    )
    flags['too_perfect'] = near_perfect.astype(int)

    score = flags.sum(axis=1)
    return score


def normalize_name(name):
    if pd.isna(name):
        return ''
    s = str(name).lower().strip()
    s = re.sub(r'[^a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def find_duplicate_clusters(df, id_col='candidate_id'):
    """
    Group candidates that plausibly represent the same real person entered twice:
    same normalized name + same graduation_year + same institute (canonical),
    OR very high fuzzy name similarity with same graduation_year & institute & age.
    Returns dict: candidate_id -> cluster_id (candidates with the same cluster_id are
    considered the same person; -1 means unique / no cluster).
    """
    work = df[[id_col, 'full_name', 'graduation_year', 'institute_canon', 'age']].copy()
    work['name_norm'] = work['full_name'].apply(normalize_name)

    # exact-key blocking: normalized name + grad year + institute
    key = work['name_norm'] + '|' + work['graduation_year'].astype(str) + '|' + work['institute_canon'].astype(str)
    cluster_id = pd.Series(-1, index=work.index)
    next_cluster = 0
    for k, grp in work.groupby(key):
        if len(grp) > 1:
            cluster_id.loc[grp.index] = next_cluster
            next_cluster += 1

    # fuzzy pass within same grad_year+institute block for near-identical names
    remaining = work[cluster_id == -1]
    for (gy, inst), grp in remaining.groupby(['graduation_year', 'institute_canon']):
        if len(grp) < 2 or len(grp) > 60:
            continue
        names = grp['name_norm'].tolist()
        idxs = grp.index.tolist()
        used = set()
        for i in range(len(idxs)):
            if idxs[i] in used:
                continue
            group_members = [idxs[i]]
            for j in range(i + 1, len(idxs)):
                if idxs[j] in used:
                    continue
                if names[i] and names[j] and fuzz.ratio(names[i], names[j]) >= 90:
                    group_members.append(idxs[j])
                    used.add(idxs[j])
            if len(group_members) > 1:
                cluster_id.loc[group_members] = next_cluster
                next_cluster += 1
                used.add(idxs[i])

    return dict(zip(df[id_col], cluster_id.values))
