from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"

TARGET = "target_value"
ID_COL = "front_id"
DATE_COL = "decision_day"
RANDOM_STATE = 42

CAT_COLS = ["db_group_last", "fl_adminarea", "region_bucket"]

TOP_REGIONS = {
    "г. Москва", "Московская область", "г. Санкт - Петербург", "Краснодарский край",
    "Свердловская область", "Новосибирская область", "Республика Татарстан (Татарстан)",
    "Челябинская область", "Ростовская область", "Самарская область",
    "Республика Башкортостан", "Тюменская область", "Омская область", "Нижегородская область",
}


def load_data():
    train = pd.read_csv(DATA_DIR / "train_apps.csv", parse_dates=[DATE_COL])
    test = pd.read_csv(DATA_DIR / "test_apps.csv", parse_dates=[DATE_COL])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    r = df.copy()

    numeric_base = [
        c for c in r.columns
        if c not in [TARGET, ID_COL, DATE_COL, "db_group_last", "fl_adminarea"]
        and pd.api.types.is_numeric_dtype(r[c])
    ]
    r["missing_count"] = r[numeric_base + ["db_group_last", "fl_adminarea"]].isna().sum(axis=1)
    r["missing_share"] = r["missing_count"] / (len(numeric_base) + 2)

    # time
    r["decision_month"] = r[DATE_COL].dt.month
    r["decision_quarter"] = r[DATE_COL].dt.quarter
    r["decision_dayofweek"] = r[DATE_COL].dt.dayofweek
    r["decision_dayofmonth"] = r[DATE_COL].dt.day
    min_day = pd.Timestamp("2024-02-01")
    r["days_from_dataset_start"] = (r[DATE_COL] - min_day).dt.days
    r["month_sin"] = np.sin(2 * np.pi * r["decision_month"] / 12)
    r["month_cos"] = np.cos(2 * np.pi * r["decision_month"] / 12)

    # rate / limit business features (per case-study hints)
    r["rate_spread_to_cb"] = r["offered_rate"] - r["cb_rate"]
    r["rate_ratio_to_cb"] = safe_div(r["offered_rate"], r["cb_rate"])
    r["limit_width"] = r["overdraft_limit_max"] - r["overdraft_limit_min"]
    r["limit_midpoint"] = (r["overdraft_limit_max"] + r["overdraft_limit_min"]) / 2
    r["loan_minus_limit_max"] = r["loan_amount_last"] - r["overdraft_limit_max"]
    r["loan_minus_limit_min"] = r["loan_amount_last"] - r["overdraft_limit_min"]
    r["loan_minus_limit_midpoint"] = r["loan_amount_last"] - r["limit_midpoint"]
    r["loan_to_limit_max_ratio"] = safe_div(r["loan_amount_last"], r["overdraft_limit_max"])
    r["loan_to_limit_min_ratio"] = safe_div(r["loan_amount_last"], r["overdraft_limit_min"])
    r["loan_to_limit_mid_ratio"] = safe_div(r["loan_amount_last"], r["limit_midpoint"])
    r["limit_utilization_width_ratio"] = safe_div(r["limit_width"], r["limit_midpoint"])

    # financial activity ratios/diffs
    r["ul_sum_30_minus_90"] = r["sum_deb_ul_30"] - r["sum_deb_ul_90"]
    r["ul_sum_30_to_90_ratio"] = safe_div(r["sum_deb_ul_30"], r["sum_deb_ul_90"])
    r["ul_cnt_30_minus_90"] = r["cnt_deb_ul_ip_30"] - r["cnt_deb_ul_ip_90"]
    r["ul_cnt_30_to_90_ratio"] = safe_div(r["cnt_deb_ul_ip_30"], r["cnt_deb_ul_ip_90"])
    r["loan_activity_balance"] = r["cnt_cred_loan_90"] - r["cnt_deb_loan_90"]
    r["loan_activity_ratio"] = safe_div(r["cnt_cred_loan_90"], r["cnt_deb_loan_90"])
    r["avg_ul_amount_per_txn_90"] = safe_div(r["sum_deb_ul_90"], r["cnt_deb_ul_ip_90"])
    r["avg_ul_amount_per_txn_30"] = safe_div(r["sum_deb_ul_30"], r["cnt_deb_ul_ip_30"])
    r["balance_to_loan_ratio"] = safe_div(r["balance_rur_amt_30_min"], r["loan_amount_last"])
    r["investment_to_balance_ratio"] = safe_div(r["sum_deb_investment_90"], r["balance_rur_amt_30_min"])

    r["digital_activity_score"] = r[
        ["corp_credit_products", "corp_list", "count_all_corp_dashboard_events", "p75_time_spent_minutes"]
    ].mean(axis=1, skipna=True)
    r["debit_activity_score"] = r[
        ["sum_deb_ul_90", "sum_deb_ul_30", "cnt_deb_ul_ip_90", "cnt_deb_ul_ip_30"]
    ].mean(axis=1, skipna=True)
    r["credit_activity_score"] = r[["cnt_deb_loan_90", "cnt_cred_loan_90"]].mean(axis=1, skipna=True)

    # log transforms for skewed magnitude features (sign-preserving)
    for col in [
        "loan_amount_last", "overdraft_limit_min", "overdraft_limit_max", "sum_deb_ul_90", "sum_deb_ul_30",
        "balance_rur_amt_30_min", "sum_deb_investment_90", "limit_width", "limit_midpoint",
    ]:
        r[f"{col}_log1p_abs"] = np.sign(r[col]) * np.log1p(np.abs(r[col]))

    # categorical / product-type signals
    r["has_previous_product_type"] = r["db_group_last"].notna().astype(int)
    r["has_region"] = r["fl_adminarea"].notna().astype(int)
    r["is_overdraft_product"] = r["db_group_last"].eq("overdraft").astype(int)
    r["is_inn_scoring_product"] = r["db_group_last"].eq("inn_scoring").astype(int)
    r["is_moscow_region"] = r["fl_adminarea"].isin(["г. Москва", "Московская область"]).astype(int)
    r["is_spb"] = r["fl_adminarea"].eq("г. Санкт - Петербург").astype(int)

    r["region_bucket"] = r["fl_adminarea"].where(r["fl_adminarea"].isin(TOP_REGIONS), "OTHER_OR_MISSING")
    r.loc[r["fl_adminarea"].isna(), "region_bucket"] = "MISSING"

    # interactions: overdraft flag x rate/limit signals (overdraft segment behaves very differently)
    r["is_overdraft_x_rate_spread"] = r["is_overdraft_product"] * r["rate_spread_to_cb"]
    r["is_overdraft_x_loan_to_limit"] = r["is_overdraft_product"] * r["loan_to_limit_mid_ratio"].fillna(0)

    for col in CAT_COLS:
        r[col] = r[col].astype("object").where(r[col].notna(), "__MISSING__").astype(str)

    return r


@dataclass
class FittedTargetEncoder:
    columns: list[str]
    prior: float
    maps: dict[str, dict]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            out[f"{col}_te"] = df[col].map(self.maps[col]).fillna(self.prior)
        return out


def fit_target_encoding(train_df: pd.DataFrame, y: pd.Series, columns: list[str], smoothing: float = 80.0) -> FittedTargetEncoder:
    prior = float(y.mean())
    maps = {}
    tmp = train_df[columns].copy()
    tmp[TARGET] = y.to_numpy()
    for col in columns:
        stats = tmp.groupby(col, dropna=False)[TARGET].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
        maps[col] = smooth.to_dict()
    return FittedTargetEncoder(columns=columns, prior=prior, maps=maps)


def oof_target_encoding(df: pd.DataFrame, y: pd.Series, columns: list[str], n_splits: int = 5, smoothing: float = 80.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = df.index.to_numpy()
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    out = pd.DataFrame(index=df.index, columns=[f"{c}_te" for c in columns], dtype=float)
    for i in range(n_splits):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        enc = fit_target_encoding(df.loc[tr_idx], y.loc[tr_idx], columns, smoothing=smoothing)
        out.loc[val_idx, :] = enc.transform(df.loc[val_idx]).to_numpy()
    return out


def feature_columns(df: pd.DataFrame):
    excluded = {TARGET, ID_COL, DATE_COL}
    categorical = CAT_COLS
    numeric = [c for c in df.columns if c not in excluded and c not in categorical and pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy() / (len(values) + 1)


def make_lgbm_dataset(train_fe: pd.DataFrame, test_fe: pd.DataFrame, te_cols_present: list[str]):
    numeric_features, categorical_features = feature_columns(train_fe)
    features = numeric_features + categorical_features
    x_train = train_fe[features].copy()
    x_test = test_fe[features].copy()
    for col in categorical_features:
        x_train[col] = x_train[col].astype("category")
        x_test[col] = x_test[col].astype("category")
        cats = x_train[col].cat.categories
        x_test[col] = x_test[col].cat.set_categories(cats)
    return x_train, x_test, categorical_features


def train_lgbm(x_train, y, x_test, cat_features, seed, params_extra):
    base = dict(
        objective="binary",
        metric="auc",
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        random_state=seed,
    )
    base.update(params_extra)
    model = lgb.LGBMClassifier(**base)
    model.fit(x_train, y, categorical_feature=cat_features)
    return model.predict_proba(x_test)[:, 1]


def train_catboost(train_fe, y, test_fe, seed, params_extra):
    numeric_features, categorical_features = feature_columns(train_fe)
    features = numeric_features + categorical_features
    x_train = train_fe[features].copy()
    x_test = test_fe[features].copy()
    cat_idx = [x_train.columns.get_loc(c) for c in categorical_features]
    for col in categorical_features:
        x_train[col] = x_train[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    pool_train = cb.Pool(x_train, y, cat_features=cat_idx)
    pool_test = cb.Pool(x_test, cat_features=cat_idx)

    base = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )
    base.update(params_extra)
    model = cb.CatBoostClassifier(**base)
    model.fit(pool_train)
    return model.predict_proba(pool_test)[:, 1]


def time_cv_folds(train_fe: pd.DataFrame, n_folds: int = 5, val_months: int = 2):
    months = sorted(train_fe[DATE_COL].dt.to_period("M").unique())
    folds = []
    for i in range(n_folds):
        val_end_idx = len(months) - 1 - i * val_months
        val_start_idx = val_end_idx - val_months + 1
        if val_start_idx <= 2:
            break
        val_months_set = set(months[val_start_idx: val_end_idx + 1])
        train_cutoff_month = months[val_start_idx - 1]
        tr_mask = train_fe[DATE_COL].dt.to_period("M") <= train_cutoff_month
        val_mask = train_fe[DATE_COL].dt.to_period("M").isin(val_months_set)
        folds.append((tr_mask, val_mask))
    return list(reversed(folds))


def evaluate_cv(train_raw: pd.DataFrame):
    train_fe = add_features(train_raw)
    folds = time_cv_folds(train_fe, n_folds=5, val_months=2)

    lgbm_params = dict(
        boosting_type="goss",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        colsample_bytree=0.85,
        reg_lambda=20.0,
        reg_alpha=0.5,
        top_rate=0.2,
        other_rate=0.1,
        max_bin=127,
    )
    cb_params = dict(
        iterations=900,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10.0,
        bagging_temperature=0.3,
        border_count=128,
    )

    rows = []
    for fi, (tr_mask, val_mask) in enumerate(folds):
        tr = train_fe[tr_mask].copy()
        val = train_fe[val_mask].copy()
        y_tr = tr[TARGET].astype(int)
        y_val = val[TARGET].astype(int)

        te_train = oof_target_encoding(tr, y_tr, CAT_COLS)
        enc = fit_target_encoding(tr, y_tr, CAT_COLS)
        tr = pd.concat([tr, te_train], axis=1)
        val = pd.concat([val, enc.transform(val)], axis=1)

        x_tr, x_val, cat_feats = make_lgbm_dataset(tr, val, [f"{c}_te" for c in CAT_COLS])
        lgbm_pred = train_lgbm(x_tr, y_tr, x_val, cat_feats, seed=42, params_extra=lgbm_params)
        cb_pred = train_catboost(tr, y_tr, val, seed=42, params_extra=cb_params)

        lgbm_auc = roc_auc_score(y_val, lgbm_pred)
        cb_auc = roc_auc_score(y_val, cb_pred)
        blend = 0.5 * rank_normalize(lgbm_pred) + 0.5 * rank_normalize(cb_pred)
        blend_auc = roc_auc_score(y_val, blend)
        rows.append({"fold": fi, "lgbm_auc": lgbm_auc, "cb_auc": cb_auc, "blend_auc": blend_auc, "val_rows": len(val)})
        print(rows[-1], flush=True)

    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_data()

    cv_df = evaluate_cv(train)
    cv_df.to_csv(REPORT_DIR / "cv_metrics.csv", index=False)
    print(cv_df.to_string(index=False))
    print("mean blend auc:", cv_df["blend_auc"].mean())

    train_fe = add_features(train)
    test_fe = add_features(test)
    y = train_fe[TARGET].astype(int)

    te_train = oof_target_encoding(train_fe, y, CAT_COLS)
    enc_full = fit_target_encoding(train_fe, y, CAT_COLS)
    train_fe = pd.concat([train_fe, te_train], axis=1)
    test_fe = pd.concat([test_fe, enc_full.transform(test_fe)], axis=1)

    lgbm_params = dict(
        boosting_type="goss",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        colsample_bytree=0.85,
        reg_lambda=20.0,
        reg_alpha=0.5,
        top_rate=0.2,
        other_rate=0.1,
        max_bin=127,
    )
    cb_params = dict(
        iterations=900,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10.0,
        bagging_temperature=0.3,
        border_count=128,
    )

    x_train, x_test, cat_feats = make_lgbm_dataset(train_fe, test_fe, [f"{c}_te" for c in CAT_COLS])
    lgbm_seeds = [42, 142, 242]
    lgbm_preds = [train_lgbm(x_train, y, x_test, cat_feats, seed=s, params_extra=lgbm_params) for s in lgbm_seeds]
    lgbm_final = np.mean([rank_normalize(p) for p in lgbm_preds], axis=0)

    cb_seeds = [42, 142]
    cb_preds = [train_catboost(train_fe, y, test_fe, seed=s, params_extra=cb_params) for s in cb_seeds]
    cb_final = np.mean([rank_normalize(p) for p in cb_preds], axis=0)

    blend = 0.5 * lgbm_final + 0.5 * cb_final

    pred_by_id = pd.DataFrame({ID_COL: test_fe[ID_COL].to_numpy(), TARGET: np.clip(blend, 1e-6, 1 - 1e-6)})
    submission = sample[[ID_COL]].merge(pred_by_id, on=ID_COL, how="left")
    submission[TARGET] = submission[TARGET].fillna(float(np.nanmean(blend)))
    submission = submission[[ID_COL, TARGET]]

    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

    summary = {
        "cv_mean_blend_auc": float(cv_df["blend_auc"].mean()),
        "cv_mean_lgbm_auc": float(cv_df["lgbm_auc"].mean()),
        "cv_mean_cb_auc": float(cv_df["cb_auc"].mean()),
        "model": "LightGBM(GOSS, native categorical) + CatBoost blend, OOF target encoding, time-based CV",
        "output": str(SUBMISSION_DIR / "submission.csv"),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
