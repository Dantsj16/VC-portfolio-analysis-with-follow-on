# VC-portfolio-analysis-with-follow-on
Series A VC portfolio analysis with follow-on comparison
This repository contains a Python script for simulating and analyzing Series A venture capital portfolio strategies, with a focus on comparing fund returns with and without follow-on investment rounds. Using Monte Carlo simulations, the code evaluates five investment approaches for a fund with investible amounts of $11.5M and $13.85M, providing insights into how follow-on investments impact expected returns, risk, and upside potential.

**Key Features**

Follow-On Comparison: Directly compares Series A portfolio returns across strategies with and without follow-on investments:

No Follow-On: "Initial Investment" (20 companies at $575K each) and "Initial + Pro-Rata" (20 companies at $425K initial, $150K pro-rata).

With Follow-On: "Initial + Pro-Rata + Follow-On" (20 companies at $300K initial, $75K pro-rata, $800K follow-on for 5 random from top 10), "Initial + Pro-Rata + Follow-On (10 Investments)" (20 companies at $300K initial, $75K pro-rata, $400K follow-on for top 10), and "Initial + Pro-Rata (12) + Follow-On (8)" (20 companies at $300K initial, $75K pro-rata for top 12, $575K follow-on for 8 random from top 10).

Monte Carlo Simulations: Runs 10,000 simulations to model return distributions, capturing the effect of follow-on strategies on portfolio outcomes.

Sensitivity Analysis: Tests three scenarios with varying failure rates (40%, 50%, 60%) to assess the robustness of follow-on vs. non-follow-on approaches.

Statistical Testing: Uses Welch’s t-test to evaluate the statistical significance of return differences between the "Initial + Pro-Rata + Follow-On" strategy and the "Initial Investment" baseline.

Visualization: Generates histograms for three key strategies ("Initial Investment", "Initial + Pro-Rata + Follow-On", "Initial + Pro-Rata (12) + Follow-On (8)"), highlighting return distributions for $11.5M and $13.85M fund sizes.

Metrics: Outputs mean, median, 5th/95th percentiles, and probability of achieving a 3x fund return for each strategy.

**Dependencies**

Python 3.x
NumPy
Pandas
Matplotlib
SciPy

**Usage**

Install dependencies: pip install numpy pandas matplotlib scipy
Run the script: python portfolio_analysis.py
Review console output for metrics and statistical tests, and view plots comparing return distributions with and without follow-on investments.

**Outputs**

Console: Detailed metrics for each strategy across three scenarios, with t-test results comparing follow-on and non-follow-on approaches.
Plots: Two histograms ($11.5M and $13.85M) showing return distributions for "Initial Investment" (no follow-on), "Initial + Pro-Rata + Follow-On", and "Initial + Pro-Rata (12) + Follow-On (8)" (both with follow-on).
Tables: Summary tables for all five strategies, listing performance metrics to compare follow-on impacts.

**Notes**

Assumes a 20-company Series A portfolio with return multiples drawn from: failure (0–0.5x), small win (1–3x), big win (5–10x), home run (10–20x).
Random seed set for reproducibility (np.random.seed(42)).
Designed to analyze the value of follow-on investments in venture capital but adaptable to other portfolio models.

**License**

MIT License
