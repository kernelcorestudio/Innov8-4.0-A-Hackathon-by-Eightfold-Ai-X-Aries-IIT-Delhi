import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from clean_utils import clean_dataframe
from integrity_utils import fabrication_score, find_duplicate_clusters
import lightgbm as lgb

RANDOM_SEED = 42
DATA_DIR = '.'

# ---- Debiased feature set: signals the insider says the NEW panel still cares about.
# Deliberately EXCLUDES: institute prestige (kept only as a small 'leadership_college'
# flag), current_city, company_type/company_size, recruitment_channel, degree "properness".
CORE_FEATURES = [
    'tech_assessment_clean', 'aptitude_clean', 'rating_clean', 'kpi_met_clean',
    'awards_clean', 'trainings_last_year', 'training_hours', 'overtime_clean',
    'enrolled_clean', 'exp_years', 'num_employers', 'last_job_change_clean',
    'notice_days', 'ctc_ratio', 'tech_z', 'apt_z',
]


def add_role_normalized(df):
    df = df.copy()
    df['tech_z'] = df.groupby('applied_role')['tech_assessment_clean'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    df['apt_z'] = df.groupby('applied_role')['aptitude_clean'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return df


def load_clean(name):
    df = pd.read_csv(f'{DATA_DIR}/{name}')
    return add_role_normalized(clean_dataframe(df))


def train_model(train_df):
    X = train_df[CORE_FEATURES]
    y = train_df['post_hire_score']
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=-1,
    )
    model.fit(X, y)
    return model


def score_pool(model, df, has_code_col):
    scores = model.predict(df[CORE_FEATURES])
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    bonus = np.zeros(len(df))
    if has_code_col:
        bonus += 0.35 * df['sustained_coder'].fillna(0).values
        bonus += 0.10 * df['some_coder'].fillna(0).values
    bonus += 0.15 * df['leadership_college'].fillna(0).values
    final = z + bonus
    return final


def apply_hard_filters(df):
    fab = fabrication_score(df)
    dup_clusters = find_duplicate_clusters(df)
    dup_series = df['candidate_id'].map(dup_clusters)

    keep = pd.Series(True, index=df.index)
    keep &= (df['title_inflated'] == 0)
    keep &= (fab <= 1)  # allow at most 1 mild inconsistency (messy data is expected)
    keep &= (df['notice_days'].isna() | (df['notice_days'] <= 60))

    # within a duplicate cluster, keep only the single best-looking row later;
    # here just tag cluster id for downstream dedup after scoring.
    return keep, dup_series, fab


def dedup_keep_best(df, dup_series, score_col='final_score'):
    df = df.copy()
    df['_dupcluster'] = dup_series.values
    out_idx = []
    for cid, grp in df.groupby('_dupcluster'):
        if cid == -1:
            out_idx.extend(grp.index.tolist())
        else:
            out_idx.append(grp[score_col].idxmax())
    return df.loc[out_idx].drop(columns=['_dupcluster'])


def evaluate_on_dev(model):
    dev = load_clean('dev.csv')
    winners = pd.read_csv(f'{DATA_DIR}/dev_winners.csv')
    winner_ids = set(winners['candidate_id'])

    dev['final_score'] = score_pool(model, dev, has_code_col=False)
    keep, dup_series, fab = apply_hard_filters(dev)
    pool = dedup_keep_best(dev[keep], dup_series[keep])
    pool = pool.sort_values('final_score', ascending=False)

    top150 = set(pool.head(150)['candidate_id'])
    recall = len(top150 & winner_ids) / len(winner_ids)
    print(f'[DEV VALIDATION] recall@150 = {recall:.3f}  '
          f'(pool size after filters/dedup = {len(pool)} / {len(dev)})')

    # also report a no-filter baseline for comparison
    baseline = dev.sort_values('final_score', ascending=False)
    top150_base = set(baseline.head(150)['candidate_id'])
    recall_base = len(top150_base & winner_ids) / len(winner_ids)
    print(f'[DEV VALIDATION] recall@150 without hard filters/dedup = {recall_base:.3f}')
    return recall


def main():
    train = load_clean('train.csv')
    model = train_model(train)

    evaluate_on_dev(model)

    test = load_clean('test.csv')
    test['final_score'] = score_pool(model, test, has_code_col=True)
    keep, dup_series, fab = apply_hard_filters(test)
    n_title = int((test['title_inflated'] == 1).sum())
    n_fab = int((fab > 1).sum())
    n_notice = int((test['notice_days'] > 60).sum())

    pool = dedup_keep_best(test[keep], dup_series[keep])
    n_dup_removed = keep.sum() - len(pool)

    print(f'Excluded: title-inflated={n_title}, fabricated={n_fab}, '
          f'long-notice={n_notice}, duplicate-rows-removed={n_dup_removed}')
    print(f'Final eligible pool: {len(pool)} / {len(test)}')

    pool = pool.sort_values('final_score', ascending=False).head(500).reset_index(drop=True)
    submission = pd.DataFrame({
        'rank': range(1, len(pool) + 1),
        'candidate_id': pool['candidate_id'],
    })
    submission.to_csv('./submission.csv', index=False)
    print('Wrote submission.csv with', len(submission), 'rows')


if __name__ == '__main__':
    main()
