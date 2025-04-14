import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
FUND_SIZE = 11_500_000  # $11.5M investible amount
FUND_SIZE_GROSS = 13_850_000  # $13.85M total fund size (reference only)
FUND_SIZE_LARGE = 13_850_000  # $13.85M investible amount for larger case
N_SIMULATIONS = 10_000  # Number of Monte Carlo runs

# "Initial Investment + Pro-Rata + Follow-On": 20 companies, $300K initial, $75K pro-rata, $800K follow-on in 5 (random from top 10)
INITIAL_INVESTMENT_PRF = 300_000  # $6M
PRO_RATA_INVESTMENT_PRF = 75_000  # $1.5M
FOLLOW_ON_INVESTMENT_PRF = 800_000  # $4M
N_COMPANIES_PRF = 20
N_FOLLOW_ON_PRF = 5  # Random 5 from top 10 companies
# Total: $6M + $1.5M + $4M = $11.5M

# "Initial Investment + Pro-Rata": 20 companies, $425K initial, $150K pro-rata
INITIAL_INVESTMENT_PR = 425_000  # $8.5M
PRO_RATA_INVESTMENT_PR = 150_000  # $3M
N_COMPANIES_PR = 20
# Total: $8.5M + $3M = $11.5M

# "Initial Investment": 20 companies, $575K initial
INITIAL_INVESTMENT_I = 575_000  # $11.5M
N_COMPANIES_I = 20
# Total: $11.5M

# Case 4: "Initial + Pro-Rata + Follow-On (10 Investments)": 20 companies, $300K initial, $75K pro-rata, $400K follow-on in 10
INITIAL_INVESTMENT_PRF_10 = 300_000  # $6M
PRO_RATA_INVESTMENT_PRF_10 = 75_000  # $1.5M
FOLLOW_ON_INVESTMENT_PRF_10 = 400_000  # $4M (same total as original, 10 investments)
N_COMPANIES_PRF_10 = 20
N_FOLLOW_ON_PRF_10 = 10  # All 10 from top 10
# Total: $6M + $1.5M + $4M = $11.5M

# Case 5: "Initial + Pro-Rata (12) + Follow-On (8)": 20 companies, $300K initial, $75K pro-rata for top 12, $575K follow-on in 8 (random from top 10)
INITIAL_INVESTMENT_PRF_12_8 = 300_000  # $6M
PRO_RATA_INVESTMENT_PRF_12_8 = 75_000  # $900K (12 companies)
FOLLOW_ON_INVESTMENT_PRF_12_8 = 575_000  # $4.6M (8 investments)
N_COMPANIES_PRF_12_8 = 20
N_PRO_RATA_PRF_12_8 = 12  # Top 12 companies
N_FOLLOW_ON_PRF_12_8 = 8  # Random 8 from top 10
# Total: $6M + $900K + $4.6M = $11.5M

# Define return multiples (random within ranges)
def generate_return_multiple(failure_rate, small_win_rate, big_win_rate, home_run_rate):
    rand = np.random.random()
    if rand < failure_rate:
        return np.random.uniform(0, 0.5)  # Fail: 0x–0.5x
    elif rand < failure_rate + small_win_rate:
        return np.random.uniform(1, 3)  # Small win: 1x–3x
    elif rand < failure_rate + small_win_rate + big_win_rate:
        return np.random.uniform(5, 10)  # Big win: 5x–10x
    else:
        return np.random.uniform(10, 20)  # Home run: 10x–20x

# Monte Carlo Simulation
def simulate_portfolio(n_companies, initial_investment, pro_rata_investment=0, n_pro_rata=None,
                       follow_on_investment=0, n_follow_on=0, failure_rate=0.65, small_win_rate=0.25,
                       big_win_rate=0.09, home_run_rate=0.01, fund_size=FUND_SIZE):
    returns = []
    for _ in range(N_SIMULATIONS):
        series_a_returns = [generate_return_multiple(failure_rate, small_win_rate, big_win_rate, home_run_rate)
                            for _ in range(n_companies)]
        
        total_return = sum(series_a_returns) * initial_investment
        
        if pro_rata_investment and n_pro_rata:
            sorted_indices = np.argsort(series_a_returns)[::-1]
            pro_rata_indices = sorted_indices[:n_pro_rata] if n_pro_rata is not None else range(n_companies)
            for idx in pro_rata_indices:
                series_a_multiple = series_a_returns[idx]
                step_up_factor = series_a_multiple ** (1/3) if series_a_multiple > 0 else 0
                pro_rata_multiple = series_a_multiple / step_up_factor if step_up_factor > 0 else 0
                total_return += pro_rata_multiple * pro_rata_investment
        
        if follow_on_investment and n_follow_on:
            sorted_indices = np.argsort(series_a_returns)[::-1]
            top_10_indices = sorted_indices[:10]
            follow_on_indices = np.random.choice(top_10_indices, size=n_follow_on, replace=False) if n_follow_on < 10 else top_10_indices
            for idx in follow_on_indices:
                series_a_multiple = series_a_returns[idx]
                step_up_factor = series_a_multiple ** (1/3) if series_a_multiple > 0 else 0
                follow_on_multiple = series_a_multiple / step_up_factor if step_up_factor > 0 else 0
                total_return += follow_on_multiple * follow_on_investment
        
        returns.append(total_return)
    return np.array(returns)

# Analyze results (in multiples of fund size)
def analyze_returns(returns, label, fund_size=FUND_SIZE):
    mean = np.mean(returns) / fund_size
    median = np.median(returns) / fund_size
    percentile_5 = np.percentile(returns, 5) / fund_size
    percentile_95 = np.percentile(returns, 95) / fund_size
    prob_3x = np.mean(returns >= 3 * fund_size) * 100
    
    print(f"\n{label} Results (x Fund Size of ${fund_size/1_000_000:.2f}M):")
    print(f"Mean Return: {mean:.2f}x")
    print(f"Median Return: {median:.2f}x")
    print(f"5th Percentile (Downside): {percentile_5:.2f}x")
    print(f"95th Percentile (Upside): {percentile_95:.2f}x")
    print(f"Probability of 3x Fund: {prob_3x:.1f}%")
    return mean, median, percentile_5, percentile_95, prob_3x

# Sensitivity Analysis on Return Distribution
def run_sensitivity_analysis():
    scenarios = [
        {"failure_rate": 0.5, "small_win_rate": 0.28, "big_win_rate": 0.2, "home_run_rate": 0.02},  # Base case
        {"failure_rate": 0.6, "small_win_rate": 0.18, "big_win_rate": 0.2, "home_run_rate": 0.02},  # Higher failure
        {"failure_rate": 0.4, "small_win_rate": 0.38, "big_win_rate": 0.2, "home_run_rate": 0.02},  # Lower failure
    ]
    
    results = {
        "Initial Investment": [],
        "Initial Investment + Pro-Rata": [],
        "Initial Investment + Pro-Rata + Follow-On": [],
        "Initial + Pro-Rata + Follow-On (10 Investments)": [],
        "Initial + Pro-Rata (12) + Follow-On (8)": []
    }
    all_returns = {
        "Initial Investment": [],
        "Initial Investment + Pro-Rata": [],
        "Initial Investment + Pro-Rata + Follow-On": [],
        "Initial + Pro-Rata + Follow-On (10 Investments)": [],
        "Initial + Pro-Rata (12) + Follow-On (8)": []
    }
    
    # Results for $13.85M Fund Size (all scenarios)
    results_large = {
        "Initial Investment": [],
        "Initial Investment + Pro-Rata": [],
        "Initial Investment + Pro-Rata + Follow-On": [],
        "Initial + Pro-Rata + Follow-On (10 Investments)": [],
        "Initial + Pro-Rata (12) + Follow-On (8)": []
    }
    all_returns_large = {
        "Initial Investment": [],
        "Initial Investment + Pro-Rata": [],
        "Initial Investment + Pro-Rata + Follow-On": [],
        "Initial + Pro-Rata + Follow-On (10 Investments)": [],
        "Initial + Pro-Rata (12) + Follow-On (8)": []
    }
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Sensitivity Scenario {i+1} ---")
        print(f"Failure: {scenario['failure_rate']}, Small Win: {scenario['small_win_rate']}, "
              f"Big Win: {scenario['big_win_rate']}, Home Run: {scenario['home_run_rate']}")
        
        # $11.5M Fund Size Simulations
        i_returns = simulate_portfolio(
            n_companies=N_COMPANIES_I, initial_investment=INITIAL_INVESTMENT_I, fund_size=FUND_SIZE, **scenario
        )
        pr_returns = simulate_portfolio(
            n_companies=N_COMPANIES_PR, initial_investment=INITIAL_INVESTMENT_PR, pro_rata_investment=PRO_RATA_INVESTMENT_PR,
            n_pro_rata=N_COMPANIES_PR, fund_size=FUND_SIZE, **scenario
        )
        prf_returns = simulate_portfolio(
            n_companies=N_COMPANIES_PRF, initial_investment=INITIAL_INVESTMENT_PRF, pro_rata_investment=PRO_RATA_INVESTMENT_PRF,
            n_pro_rata=N_COMPANIES_PRF, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF, n_follow_on=N_FOLLOW_ON_PRF,
            fund_size=FUND_SIZE, **scenario
        )
        prf_10_returns = simulate_portfolio(
            n_companies=N_COMPANIES_PRF_10, initial_investment=INITIAL_INVESTMENT_PRF_10, pro_rata_investment=PRO_RATA_INVESTMENT_PRF_10,
            n_pro_rata=N_COMPANIES_PRF_10, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF_10, n_follow_on=N_FOLLOW_ON_PRF_10,
            fund_size=FUND_SIZE, **scenario
        )
        prf_12_8_returns = simulate_portfolio(
            n_companies=N_COMPANIES_PRF_12_8, initial_investment=INITIAL_INVESTMENT_PRF_12_8, pro_rata_investment=PRO_RATA_INVESTMENT_PRF_12_8,
            n_pro_rata=N_PRO_RATA_PRF_12_8, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF_12_8, n_follow_on=N_FOLLOW_ON_PRF_12_8,
            fund_size=FUND_SIZE, **scenario
        )
        
        # Analyze and store $11.5M results
        i_stats = analyze_returns(i_returns, f"Initial Investment (Scenario {i+1})", fund_size=FUND_SIZE)
        pr_stats = analyze_returns(pr_returns, f"Initial Investment + Pro-Rata (Scenario {i+1})", fund_size=FUND_SIZE)
        prf_stats = analyze_returns(prf_returns, f"Initial Investment + Pro-Rata + Follow-On (Scenario {i+1})", fund_size=FUND_SIZE)
        prf_10_stats = analyze_returns(prf_10_returns, f"Initial + Pro-Rata + Follow-On (10 Investments) (Scenario {i+1})", fund_size=FUND_SIZE)
        prf_12_8_stats = analyze_returns(prf_12_8_returns, f"Initial + Pro-Rata (12) + Follow-On (8) (Scenario {i+1})", fund_size=FUND_SIZE)
        
        results["Initial Investment"].append(i_stats)
        results["Initial Investment + Pro-Rata"].append(pr_stats)
        results["Initial Investment + Pro-Rata + Follow-On"].append(prf_stats)
        results["Initial + Pro-Rata + Follow-On (10 Investments)"].append(prf_10_stats)
        results["Initial + Pro-Rata (12) + Follow-On (8)"].append(prf_12_8_stats)
        
        all_returns["Initial Investment"].append(i_returns)
        all_returns["Initial Investment + Pro-Rata"].append(pr_returns)
        all_returns["Initial Investment + Pro-Rata + Follow-On"].append(prf_returns)
        all_returns["Initial + Pro-Rata + Follow-On (10 Investments)"].append(prf_10_returns)
        all_returns["Initial + Pro-Rata (12) + Follow-On (8)"].append(prf_12_8_returns)
        
        # Statistical difference for $11.5M
        t_stat, p_value = ttest_ind(prf_returns, i_returns, equal_var=False)  # Welch's t-test
        print(f"\nStatistical Difference (Base Follow-On vs. No Follow-On/Pro-Rata) - $11.5M, Scenario {i+1}:")
        print(f"T-Statistic: {t_stat:.2f}")
        print(f"P-Value: {p_value:.4f} {'(Significant)' if p_value < 0.05 else '(Not Significant)'}")
        
        # $13.85M Fund Size Simulations
        i_returns_large = simulate_portfolio(
            n_companies=N_COMPANIES_I, initial_investment=INITIAL_INVESTMENT_I, fund_size=FUND_SIZE_LARGE, **scenario
        )
        pr_returns_large = simulate_portfolio(
            n_companies=N_COMPANIES_PR, initial_investment=INITIAL_INVESTMENT_PR, pro_rata_investment=PRO_RATA_INVESTMENT_PR,
            n_pro_rata=N_COMPANIES_PR, fund_size=FUND_SIZE_LARGE, **scenario
        )
        prf_returns_large = simulate_portfolio(
            n_companies=N_COMPANIES_PRF, initial_investment=INITIAL_INVESTMENT_PRF, pro_rata_investment=PRO_RATA_INVESTMENT_PRF,
            n_pro_rata=N_COMPANIES_PRF, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF, n_follow_on=N_FOLLOW_ON_PRF,
            fund_size=FUND_SIZE_LARGE, **scenario
        )
        prf_10_returns_large = simulate_portfolio(
            n_companies=N_COMPANIES_PRF_10, initial_investment=INITIAL_INVESTMENT_PRF_10, pro_rata_investment=PRO_RATA_INVESTMENT_PRF_10,
            n_pro_rata=N_COMPANIES_PRF_10, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF_10, n_follow_on=N_FOLLOW_ON_PRF_10,
            fund_size=FUND_SIZE_LARGE, **scenario
        )
        prf_12_8_returns_large = simulate_portfolio(
            n_companies=N_COMPANIES_PRF_12_8, initial_investment=INITIAL_INVESTMENT_PRF_12_8, pro_rata_investment=PRO_RATA_INVESTMENT_PRF_12_8,
            n_pro_rata=N_PRO_RATA_PRF_12_8, follow_on_investment=FOLLOW_ON_INVESTMENT_PRF_12_8, n_follow_on=N_FOLLOW_ON_PRF_12_8,
            fund_size=FUND_SIZE_LARGE, **scenario
        )
        
        # Analyze and store $13.85M results
        i_stats_large = analyze_returns(i_returns_large, f"Initial Investment (Scenario {i+1}, $13.85M)", fund_size=FUND_SIZE_LARGE)
        pr_stats_large = analyze_returns(pr_returns_large, f"Initial Investment + Pro-Rata (Scenario {i+1}, $13.85M)", fund_size=FUND_SIZE_LARGE)
        prf_stats_large = analyze_returns(prf_returns_large, f"Initial Investment + Pro-Rata + Follow-On (Scenario {i+1}, $13.85M)", fund_size=FUND_SIZE_LARGE)
        prf_10_stats_large = analyze_returns(prf_10_returns_large, f"Initial + Pro-Rata + Follow-On (10 Investments) (Scenario {i+1}, $13.85M)", fund_size=FUND_SIZE_LARGE)
        prf_12_8_stats_large = analyze_returns(prf_12_8_returns_large, f"Initial + Pro-Rata (12) + Follow-On (8) (Scenario {i+1}, $13.85M)", fund_size=FUND_SIZE_LARGE)
        
        results_large["Initial Investment"].append(i_stats_large)
        results_large["Initial Investment + Pro-Rata"].append(pr_stats_large)
        results_large["Initial Investment + Pro-Rata + Follow-On"].append(prf_stats_large)
        results_large["Initial + Pro-Rata + Follow-On (10 Investments)"].append(prf_10_stats_large)
        results_large["Initial + Pro-Rata (12) + Follow-On (8)"].append(prf_12_8_stats_large)
        
        all_returns_large["Initial Investment"].append(i_returns_large)
        all_returns_large["Initial Investment + Pro-Rata"].append(pr_returns_large)
        all_returns_large["Initial Investment + Pro-Rata + Follow-On"].append(prf_returns_large)
        all_returns_large["Initial + Pro-Rata + Follow-On (10 Investments)"].append(prf_10_returns_large)
        all_returns_large["Initial + Pro-Rata (12) + Follow-On (8)"].append(prf_12_8_returns_large)
        
        # Statistical difference for $13.85M
        t_stat_large, p_value_large = ttest_ind(prf_returns_large, i_returns_large, equal_var=False)
        print(f"\nStatistical Difference (Base Follow-On vs. No Follow-On/Pro-Rata) - $13.85M, Scenario {i+1}:")
        print(f"T-Statistic: {t_stat_large:.2f}")
        print(f"P-Value: {p_value_large:.4f} {'(Significant)' if p_value_large < 0.05 else '(Not Significant)'}")
    
    # Compute average returns across scenarios
    avg_returns = {key: np.mean(np.array(val), axis=0) for key, val in all_returns.items()}
    avg_returns_large = {key: np.mean(np.array(val), axis=0) for key, val in all_returns_large.items()}
    
    return results, results_large, scenarios, avg_returns, avg_returns_large

# Run sensitivity analysis
results, results_large, scenarios, avg_returns, avg_returns_large = run_sensitivity_analysis()

# Plotting for $11.5M fund size
plt.figure(figsize=(12, 6))
plot_cases = [
    "Initial Investment",
    "Initial Investment + Pro-Rata + Follow-On",
    "Initial + Pro-Rata (12) + Follow-On (8)"
]
for key in plot_cases:
    plt.hist(avg_returns[key] / FUND_SIZE, bins=50, alpha=0.5, label=key + " (Avg)", density=True)
plt.axvline(3, color='black', linestyle='--', label="3x Fund Goal")
plt.xlim(0, 4)
plt.xticks(np.arange(0, 5, 0.25))
plt.xlabel("Portfolio Return (Multiple of Fund Size)")
plt.ylabel("Density")
plt.title("Monte Carlo Simulation: Average Return Distribution Across 3 Scenarios ($11.5M Fund)")
plt.legend()
plt.show()

# Plotting for $13.85M fund size
plt.figure(figsize=(12, 6))
for key in plot_cases:
    plt.hist(avg_returns_large[key] / FUND_SIZE_LARGE, bins=50, alpha=0.5, label=key + " (Avg)", density=True)
plt.axvline(3, color='black', linestyle='--', label="3x Fund Goal")
plt.xlim(0, 4)
plt.xticks(np.arange(0, 5, 0.25))
plt.xlabel("Portfolio Return (Multiple of Fund Size)")
plt.ylabel("Density")
plt.title("Monte Carlo Simulation: Average Return Distribution Across 3 Scenarios ($13.85M Fund)")
plt.legend()
plt.show()

# Summarize sensitivity results in a table (in multiples)
for plan in results.keys():
    df = pd.DataFrame(results[plan], columns=["Mean", "Median", "5th Pctl", "95th Pctl", "Prob 3x"])
    print(f"\nSensitivity Summary - {plan} (x Fund Size of $11.5M):\n", df.to_string(index=False))

for plan in results_large.keys():
    df_large = pd.DataFrame(results_large[plan], columns=["Mean", "Median", "5th Pctl", "95th Pctl", "Prob 3x"])
    print(f"\nSensitivity Summary - {plan} (x Fund Size of $13.85M):\n", df_large.to_string(index=False))
