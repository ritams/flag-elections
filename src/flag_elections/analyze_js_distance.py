import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import linregress

# Ensure src is in path to import rvm
sys.path.append(os.path.join(os.getcwd(), 'src'))

from flag_elections import rvm

def analyze_and_plot():
    # Load data
    data_path = os.path.join('data', 'cleaned', 'cleaned_election_data.pkl')
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run clean_data.py first.")
        return

    results = []

    print("Calculating JS distances...")
    for country, country_data in data.items():
        # Extract data
        winners = np.array(country_data['winner'])
        runner_ups = np.array(country_data['runner_up'])
        turnouts = np.array(country_data['turnout'])
        
        # Calculate specific margin mu = (winner - runner_up) / turnout
        # Filter valid turnouts to avoid division by zero
        valid_indices = turnouts > 0
        mu = (winners[valid_indices] - runner_ups[valid_indices]) / turnouts[valid_indices]
        
        # Filter mu > 0 and mu < 1
        mu = mu[(mu > 0) & (mu < 1)]
        
        n_points = len(mu)
        if n_points < 10:  # Skip countries with very few data points
            continue

        # Calculate scaled specific margin x = mu / <mu>
        mean_mu_empirical = np.mean(mu)
        x = mu / mean_mu_empirical
        
        # Calculate JS distance
        # Using linear bins as per previous refinement
        try:
           js_dist = rvm.calculate_js_distance(x, rvm.f_x, bins=np.linspace(min(x), max(x), 100))
           results.append({
               'Country': country,
               'JS_Distance': js_dist,
               'N': n_points
           })
        except Exception as e:
            print(f"Skipping {country} due to error: {e}")

    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_dir = os.path.join('data', 'cleaned')
    csv_path = os.path.join(output_dir, 'js_distance_vs_n.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plotting
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Sort by N for connected line plot
    df_results = df_results.sort_values(by='N')
    
    # Linear regression in log-log space
    log_N = np.log10(df_results['N'])
    log_JS = np.log10(df_results['JS_Distance'])
    
    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_JS)
    print(f"Regression: slope={slope:.4f}, intercept={intercept:.4f}, r_value={r_value:.4f}")
    
    # Calculate fitted values for plotting line
    fit_log_JS = slope * log_N + intercept
    fit_JS = 10**fit_log_JS
    
    # Plotting points connected by dotted lines
    plt.loglog(df_results['N'], df_results['JS_Distance'], 'o--', 
               color='#2E86C1', 
               alpha=0.6, 
               markersize=6,
               markeredgewidth=0,
               linewidth=1,
               label='Data')
               
    # Plot regression line
    plt.loglog(df_results['N'], fit_JS, 'r-', 
               linewidth=2, 
               label=f'Empirical Fit (slope={slope:.2f})')
               
    # Add Simulated Slope Reference
    # Slope -0.48 from simulation
    sim_slope = -0.4816
    # Center the reference line on the data centroid
    mean_log_N = np.mean(log_N)
    mean_log_JS = np.mean(log_JS)
    # intercept = y_bar - m * x_bar
    sim_intercept = mean_log_JS - sim_slope * mean_log_N
    
    ref_fit_JS = 10**(sim_slope * log_N + sim_intercept)
    
    plt.loglog(df_results['N'], ref_fit_JS, 'g--', 
               linewidth=2, 
               label=f'Simulated Trend (slope={sim_slope:.2f})')
               
    # Outlier detection (residuals)
    residuals = log_JS - fit_log_JS
    std_resid = np.std(residuals)
    
    # Annotate outliers (> 1.5 std dev)
    outliers = df_results[np.abs(residuals) > 1.5 * std_resid]
    
    for _, row in outliers.iterrows():
        plt.annotate(row['Country'], 
                     (row['N'], row['JS_Distance']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=9,
                     alpha=0.8)
    
    plt.xlabel('Number of Data Points (N)')
    plt.ylabel('Jensen-Shannon Distance')
    plt.title('JS Distance vs Number of Data Points')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, 'js_distance_vs_n.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    analyze_and_plot()
