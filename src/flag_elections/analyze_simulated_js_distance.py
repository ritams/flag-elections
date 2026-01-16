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

def analyze_and_plot_simulated():
    # Load data
    data_path = os.path.join('data', 'cleaned', 'cleaned_election_data.pkl')
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run clean_data.py first.")
        return

    results = []

    print("Running simulations and calculating JS distances...")
    # Use exact same parameters as requested
    SIM_CANDIDATES = [3]  # "use the number of candidates to be three everywhere"
    SIM_ITERATIONS = 1    # "number of simulation would be just one"

    for country, country_data in data.items():
        # Get actual turnout data
        turnouts = np.array(country_data['turnout'])
        
        # Filter valid turnouts (must be > 0 and integer)
        valid_turnouts = turnouts[turnouts > 0].astype(int)
        
        n_points = len(valid_turnouts)
        if n_points < 10:
            continue
            
        # Simulate model
        # simulate_model returns (model_winner_arr, model_runner_up_arr, votes_arr, new_turnout_arr)
        # We process it as a list because simulate_model expects a list for turnout_arr
        winners_sim, runner_ups_sim, _, turnouts_sim = rvm.simulate_model(
            valid_turnouts, 
            SIM_CANDIDATES, 
            num_of_simulations=SIM_ITERATIONS
        )
        
        # Note: with 1 simulation, the output arrays are just flattened concatenations of that one run.
        # But verify shape. new_turnout_arr is a list of turnouts.
        
        # Calculate specific margin mu for simulated data
        turnouts_sim = np.array(turnouts_sim)
        
        # Avoid division by zero naturally, but double check
        valid_indices = turnouts_sim > 0
        mu_sim = (winners_sim[valid_indices] - runner_ups_sim[valid_indices]) / turnouts_sim[valid_indices]
        
        # Filter mu
        mu_sim = mu_sim[(mu_sim > 0) & (mu_sim < 1)]
        
        if len(mu_sim) < 10:
             continue
             
        # Calculate scaled specific margin x
        mean_mu_sim = np.mean(mu_sim)
        x_sim = mu_sim / mean_mu_sim
        
        # Calculate JS distance vs analytical
        try:
           js_dist = rvm.calculate_js_distance(x_sim, rvm.f_x, bins=np.linspace(min(x_sim), max(x_sim), 50))
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
    csv_path = os.path.join(output_dir, 'simulated_js_distance_vs_n.csv')
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
               color='#27AE60',  # Green for simulation
               alpha=0.6, 
               markersize=6,
               markeredgewidth=0,
               linewidth=1,
               label='Simulated Data (Candidate=3)')
               
    # Plot regression line
    plt.loglog(df_results['N'], fit_JS, 'r-', 
               linewidth=2, 
               label=f'Fit (slope={slope:.2f})')
               
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
    plt.ylabel('Jensen-Shannon Distance (Simulated)')
    plt.title('Simulated JS Distance vs Number of Data Points')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, 'simulated_js_distance_vs_n.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    analyze_and_plot_simulated()
