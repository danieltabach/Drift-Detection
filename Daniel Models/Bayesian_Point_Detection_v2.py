# %%
import pandas as pd
import numpy as np
import random
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
import numpy as np
from scipy.stats import norm
random.seed(1)
np.random.seed(1)

# %%
time_a_df = pd.read_csv('Data/time_a.csv')
time_b_df_decay = pd.read_csv('Data/time_b_df_decay.csv')
time_b_df_growth =pd.read_csv('Data/time_b_df_growth.csv')
time_b_df_normal = pd.read_csv('Data/time_b_df_normal.csv')

# %%
import numpy as np
from scipy.stats import norm

class DriftDetector:
    def __init__(self, baseline_data, window_size=7, threshold_drop=1):
        """
        Lightweight Bayesian drift detector using Gaussian likelihood scoring.

        Parameters:
        - baseline_data: np.array or pd.Series of engagement from Time A
        - window_size: number of days in each scoring window
        - threshold_drop: log-likelihood drop threshold to signal drift
        """
        self.window_size = window_size
        self.threshold_drop = threshold_drop
        self.mu = np.mean(baseline_data)
        self.sigma = np.std(baseline_data)
        self.fitted = True

    def compute_log_likelihood(self, window):
        return np.sum(norm.logpdf(window, loc=self.mu, scale=self.sigma))

    def detect(self, time_b_series):
        """
        Slides a window over Time B and returns the first changepoint (if any).
        """
        log_likelihoods = []
        changepoint = None

        for i in range(0, len(time_b_series) - self.window_size):
            window = time_b_series[i:i + self.window_size]
            ll = self.compute_log_likelihood(window)
            log_likelihoods.append(ll)

            if i >= 5:
                prev_avg = np.mean(log_likelihoods[max(0, i-5):i])
                if prev_avg - ll > self.threshold_drop:
                    changepoint = i + self.window_size
                    break

        return changepoint, log_likelihoods


# %%

def run_group_drift_detection(group_name, time_a_df, time_b_df, window_size=7, threshold_drop=1):
    group_a = time_a_df[time_a_df['Group'] == group_name]['Engagement'].values
    group_b = time_b_df[time_b_df['Group'] == group_name]['Engagement'].values

    detector = DriftDetector(baseline_data=group_a, window_size=window_size, threshold_drop=threshold_drop)
    changepoint, ll_trace = detector.detect(group_b)

    return changepoint, ll_trace

# %%
def evaluate_all_groups(time_a_df, time_b_df, time_b_start_date, window_size=7, threshold_drop=1):
    group_results = {}
    for group in ['Baseline', 'Drifters', 'Power Users']:
        cp_index, _ = run_group_drift_detection(group, time_a_df, time_b_df, window_size, threshold_drop)

        if cp_index is not None:
            cp_date = pd.to_datetime(time_b_start_date) + pd.Timedelta(days=cp_index)
        else:
            cp_date = "Not Detected"

        group_results[group] = {
            "changepoint_index": cp_index,
            "changepoint_date": cp_date
        }

    return group_results

# %%
def plot_log_likelihoods(ll_trace, title="Log-Likelihood Drift Detection"):
    plt.plot(ll_trace)
    plt.title(title)
    plt.xlabel("Window Index")
    plt.ylabel("Log-Likelihood")
    plt.axvline(np.argmax(np.diff(ll_trace)), color='red', linestyle='--')
    plt.grid(True)
    plt.show()

# %%
# group = "Power Users"  # or "Baseline", or "Drifters"

# # Extract engagement values for the selected group from Time A and Time B
# group_a = time_a_df[time_a_df['Group'] == group]['Engagement'].values
# group_b = time_b_df_decay[time_b_df_decay['Group'] == group]['Engagement'].values

# print(f"🔍 Analyzing group: {group}")
# print(f"Time A size: {len(group_a)}, Time B size: {len(group_b)}")

# %%
# detector = DriftDetector(baseline_data=group_a, window_size=7, threshold_drop=1)
# changepoint, ll_trace = detector.detect(group_b)

# print(f"Changepoint: {changepoint}")


# %%
# import matplotlib.pyplot as plt

# plt.plot(ll_trace)
# plt.axhline(np.mean(ll_trace[:5]), color='red', linestyle='--', label="Early Avg")
# plt.title(f"Log-Likelihood Drift Trace for {group}")
# plt.xlabel("Window Index")
# plt.ylabel("Log-Likelihood")
# plt.grid(True)
# plt.legend()
# plt.show()


# %%
# results = evaluate_all_groups(time_a_df, time_b_df_decay, "2024-07-01")
# for group, info in results.items():
#     print(f"{group} ➜ Detected changepoint: {info['changepoint_date']}")

# %%

def plot_group_drift_detection(time_b_df, results, time_b_start_date):
    time_b_df = time_b_df.copy()
    time_b_df['Date'] = pd.to_datetime(time_b_df['Date'])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, group in enumerate(['Baseline', 'Drifters', 'Power Users']):
        group_df = time_b_df[time_b_df['Group'] == group]
        group_df['Month'] = group_df['Date'].dt.to_period('M').astype(str)

        ax = axs[i]
        sns.boxplot(data=group_df, x='Month', y='Engagement', ax=ax)
        ax.set_title(f"Group = {group}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Mark true changepoint
        true_cp_date = pd.to_datetime(time_b_start_date)
        ax.axvline(x=0, color='green', linestyle='--', label='True Drift Start')

        # Mark detected changepoint
        cp_index = results[group]["changepoint_index"]
        if cp_index is not None:
            detected_cp_date = pd.to_datetime(time_b_start_date) + pd.Timedelta(days=cp_index)
            ax.axvline(
                x=(detected_cp_date.to_period('M') - true_cp_date.to_period('M')).n, 
                color='red', linestyle='-', label='Detected Drift'
            )

        if i == 0:
            ax.set_ylabel("Engagement")
        else:
            ax.set_ylabel("")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle("📈 Monthly Engagement + Drift Detection Lines", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
def plot_group_drift_detection_lineplot(time_b_df, results, time_b_start_date):
    time_b_df = time_b_df.copy()
    time_b_df['Date'] = pd.to_datetime(time_b_df['Date'])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, group in enumerate(['Baseline', 'Drifters', 'Power Users']):
        group_df = time_b_df[time_b_df['Group'] == group].copy()
        daily_engagement = group_df.groupby('Date')['Engagement'].mean()

        ax = axs[i]
        ax.plot(daily_engagement.index, daily_engagement.values, label="Engagement", color='blue')
        ax.set_title(f"Group: {group}")
        ax.set_xlabel("Date")
        if i == 0:
            ax.set_ylabel("Engagement")

        true_cp_date = pd.to_datetime(time_b_start_date)
        ax.axvline(x=true_cp_date, color='green', linestyle='--', label='True Drift Start')

        cp_index = results[group]["changepoint_index"]
        if cp_index is not None:
            detected_cp_date = true_cp_date + pd.Timedelta(days=cp_index)
            # Cap the detected date by the maximum date in the group data
            max_date = group_df['Date'].max()
            if detected_cp_date > max_date:
                detected_cp_date = max_date
            ax.axvline(x=detected_cp_date, color='red', linestyle='-', label='Detected Drift')

        ax.grid(True)
        ax.legend()

    fig.suptitle("📉 Daily Engagement with Drift Detection per Group", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
results_decay = evaluate_all_groups(time_a_df, time_b_df_decay, "2024-07-01", window_size=21, threshold_drop=4)
results_growth = evaluate_all_groups(time_a_df, time_b_df_growth, "2024-07-01", window_size=21, threshold_drop=4)
results_normal = evaluate_all_groups(time_a_df, time_b_df_normal, "2024-07-01", window_size=21, threshold_drop=4)

# Plot line plots for each:
plot_group_drift_detection_lineplot(time_b_df_decay, results_decay, "2024-07-01")
plot_group_drift_detection_lineplot(time_b_df_growth, results_growth, "2024-07-01")
plot_group_drift_detection_lineplot(time_b_df_normal, results_normal, "2024-07-01")



# %%


# %%
baseline_df = time_b_df_decay[
    (time_b_df_decay['Group'] == 'Baseline') &
    (time_b_df_decay['Date'] >= '2024-10-01')
]
print(baseline_df.shape)
print(baseline_df.head(20))


# %%
baseline_df['Date'].unique()

# %%



