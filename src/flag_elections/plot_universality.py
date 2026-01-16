import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is in path to import rvm
sys.path.append(os.path.join(os.getcwd(), 'src'))

from flag_elections import rvm

def plot_universality(country_name='India'):
    # Load data
    data_path = os.path.join('data', 'cleaned', 'cleaned_election_data.pkl')
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run clean_data.py first.")
        return

    if country_name not in data:
        print(f"Error: {country_name} not found in data.")
        return

    country_data = data[country_name]
    
    # Extract data
    winners = np.array(country_data['winner'])
    runner_ups = np.array(country_data['runner_up'])
    turnouts = np.array(country_data['turnout'])
    
    # Calculate specific margin mu = (winner - runner_up) / turnout
    # Filter valid turnouts to avoid division by zero
    valid_indices = turnouts > 0
    mu = (winners[valid_indices] - runner_ups[valid_indices]) / turnouts[valid_indices]
    
    # Filter mu > 0 and mu < 1 for safety, though mu should be >= 0
    mu = mu[(mu > 0) & (mu < 1)]
    
    if len(mu) == 0:
        print(f"No valid data for {country_name}")
        return

    # Calculate scaled specific margin x = mu / <mu>
    # Use empirical mean for scaling to show universality
    mean_mu_empirical = np.mean(mu)
    x = mu / mean_mu_empirical
    
    print(f"Stats for {country_name}:")
    print(f"  Count: {len(x)}")
    print(f"  Mean mu: {mean_mu_empirical:.4f}")
    
    # Plotting
    # Use a modern style
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
    
    # Empirical Histogram
    # Using np.histogram to get density
    # Switching to linspace as requested
    counts, bin_edges = np.histogram(x, bins=np.linspace(min(x), max(x), 50), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter zero counts for log plot
    valid_bins = counts > 0
    # Use a nice blue color
    plt.plot(bin_centers[valid_bins], counts[valid_bins], 'o', 
             label=f'Empirical ({country_name})', 
             color='#2E86C1', 
             alpha=0.8, 
             markersize=8, 
             markeredgewidth=0)
    
    # Analytical Prediction
    # F(x)
    x_theoretical = np.logspace(np.log10(1e-3), np.log10(10), 1000)
    y_theoretical = [rvm.f_x(xi) for xi in x_theoretical]
    
    # Use a vivid red/orange
    plt.plot(x_theoretical, y_theoretical, '-', 
             label='Analytical Pred. RVM', 
             color='#E74C3C', 
             linewidth=3)
    
    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    
    # Limit range as requested
    plt.xlim(1e-2, 10)
    plt.ylim(1e-3, 2)
    
    plt.xlabel(r'Scaled Specific Margin $x = \mu / \langle \mu \rangle$')
    plt.ylabel(r'Probability Density $F(x)$')
    
    # Calculate JS distance
    js_dist = rvm.calculate_js_distance(x, rvm.f_x, bins=np.linspace(min(x), max(x), 50))
    print(f"  JS Distance: {js_dist:.4f}")
    
    plt.title(f'Universality of Margin Distribution: {country_name}\nJS Distance: {js_dist:.4f}')
    plt.legend(frameon=False)
    # Grid removed as requested
    
    plt.tight_layout()
    output_dir = 'docs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'universality_{country_name.lower()}.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    country = "India"
    if len(sys.argv) > 1:
        country = sys.argv[1]
    plot_universality(country)
