

import numpy as np
import pandas as pd
import pyreadstat  # to read .dta if needed, or you can use pd.read_stata
import random
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

################################################################################
# 1) Read the .dta file and prepare the data
################################################################################

# Example: read .dta with pandas directly
# df, meta = pyreadstat.read_dta("H:/data_rep.dta")  # or use pd.read_stata
df = pd.read_stata("H:/data_rep.dta")


df["y"] = df["loansamt_total"]
df.drop(columns=["loansamt_total"], inplace=True)

df["d"] = df["treatment"]
df.drop(columns=["treatment"], inplace=True)

# covariates (same as your R code)
covariates = [
    "members_resid_bl", "nadults_resid_bl", "head_age_bl", "act_livestock_bl", "act_business_bl",
    "borrowed_total_bl", "members_resid_d_bl", "nadults_resid_d_bl", "head_age_d_bl", "act_livestock_d_bl",
    "act_business_d_bl", "borrowed_total_d_bl", "ccm_resp_activ", "other_resp_activ", "ccm_resp_activ_d",
    "other_resp_activ_d", "head_educ_1", "nmember_age6_16"
]

# Keep only y, d, covariates
cols_to_keep = ["y", "d"] + covariates
df = df[cols_to_keep]

# Remove rows with missing values
df = df.dropna()

# ID variable from 1..N
df["ID"] = np.arange(1, len(df) + 1)

# Number of bootstrap repetitions
B = 400

print("Data columns:", df.columns)
print(f"Number of observations after cleaning: {len(df)}")

################################################################################
# 2) Define a helper to create a bootstrapped dataset (stratified by d)
################################################################################

def create_bootstrapped_data(df_source):
    """
    Sample with replacement *within d=0 and d=1* to preserve treatment ratio.
    This matches your R code's createbootstrappedData().
    """
    df0 = df_source[df_source["d"] == 0]
    df1 = df_source[df_source["d"] == 1]

    n0 = len(df0)
    n1 = len(df1)

    # sample indices from each stratum
    idx0 = np.random.choice(df0.index, size=n0, replace=True)
    idx1 = np.random.choice(df1.index, size=n1, replace=True)

    # combine them
    new_index = np.concatenate([idx0, idx1])
    return df_source.loc[new_index].copy()

################################################################################
# 3) Prepare for 5-fold sample splitting
################################################################################

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
folds = list(kf.split(df))

# We'll store results in these arrays: DR, IPW
# For each observation (row), for each bootstrap iteration (B),
# we store an estimate. So shape is (N, B).
n = len(df)
results_cate_DR = np.zeros((n, B))
results_cate_IPW = np.zeros((n, B))

################################################################################
# 4) Main loop over folds, then over B bootstrap reps
################################################################################

for f_idx, (train_index, test_index) in enumerate(folds, start=1):
    # data1 in R code
    data1 = df.iloc[train_index].copy()
    # df_main in R code
    df_main = df.iloc[test_index].copy()

    for b in range(1, B + 1):
        # set seed for reproducibility
        random.seed(1011 + b)
        np.random.seed(1011 + b)

        # 4.1) create bootstrapped data from data1
        df_aux = create_bootstrapped_data(data1)

        # 4.2) Fit a classification model to get the propensity scores, p_hat
        #     (R code used SuperLearner(..., family=binomial)). We'll use logistic regression for simplicity.

        X_aux = df_aux[covariates].values
        d_aux = df_aux["d"].values

        # logistic regression
        prop_model = LogisticRegression(solver="lbfgs", max_iter=1000)
        prop_model.fit(X_aux, d_aux)

        X_main = df_main[covariates].values
        # predicted probabilities of d=1
        p_hat = prop_model.predict_proba(X_main)[:, 1]

        # Overlap bounding: ifelse(p_hat < 0.025, 0.025, ifelse(p_hat > 0.975, 0.975, p_hat))
        p_hat = np.clip(p_hat, 0.025, 0.975)

        # 4.3) For DR, we also need outcome models for treated and control
        # Split df_aux by d
        df_aux_1 = df_aux[df_aux["d"] == 1]
        df_aux_0 = df_aux[df_aux["d"] == 0]

        # Fit random forests for outcome regression
        #   m1: E[Y|X, d=1]
        #   m0: E[Y|X, d=0]

        m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
        m1.fit(df_aux_1[covariates], df_aux_1["y"])

        m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
        m0.fit(df_aux_0[covariates], df_aux_0["y"])

        # predict on df_main
        m1_hat = m1.predict(df_main[covariates])
        m0_hat = m0.predict(df_main[covariates])

        # 4.4) DR: y_mo = (m1_hat - m0_hat) + [ d*(y - m1_hat)/p_hat - (1-d)*(y - m0_hat)/(1-p_hat) ]
        d_main = df_main["d"].values
        y_main = df_main["y"].values

        # first part
        first_part = m1_hat - m0_hat

        # second part
        second_part = (d_main * (y_main - m1_hat) / p_hat) - ((1 - d_main) * (y_main - m0_hat) / (1 - p_hat))

        y_mo = first_part + second_part

        # store DR results
        test_ids = df_main["ID"].values.astype(int) - 1  # if ID is 1..N, we shift by -1 for zero-based index
        results_cate_DR[test_ids, b - 1] = y_mo

        # 4.5) IPW: for each observation:
        # IPW_i = d_i * y_i / p_i - (1 - d_i)*y_i / (1 - p_i)
        # This yields an estimate of the "effect" for each i.
        ipw_i = (d_main * y_main / p_hat) - ((1 - d_main) * y_main / (1 - p_hat))
        results_cate_IPW[test_ids, b - 1] = ipw_i

        if b % 50 == 0:
            print(f"[Fold {f_idx}] Bootstrap iteration {b} / {B}")

################################################################################
# 5) Done. We have DR & IPW results for each row x B draws. Let's compute means
################################################################################

# We'll compute the average DR / IPW across B draws for each observation
mean_DR = np.mean(results_cate_DR, axis=1)  # shape (n,)
mean_IPW = np.mean(results_cate_IPW, axis=1)

# Attach them to df
df["cate_dr"] = mean_DR
df["cate_ipw"] = mean_IPW

print("\nDone! We stored final DR & IPW estimates in df['cate_dr'] and df['cate_ipw'].\n")

################################################################################
# 6) Optionally: Save results to a CSV or do further analysis
################################################################################

df.to_csv("python_DR_IPW_results.csv", index=False)
print("Wrote 'python_DR_IPW_results.csv' with columns ID, d, y, covariates, cate_dr, cate_ipw.")
