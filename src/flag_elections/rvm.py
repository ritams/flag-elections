import numpy as np
import tqdm
from scipy.spatial.distance import jensenshannon

def simulate_model(turnout_arr, candidates_arr, num_of_simulations=10):
    """
    Simulates the Random Vote Model (RVM) based on turnout and candidate counts.
    """
    num_constituencies = len(turnout_arr)
    num_of_voters_array = np.array(turnout_arr, dtype=int)
    
    model_winner_arr = []
    model_runner_up_arr = []
    votes_arr = []
    new_turnout_arr = []
    
    # Outer loop
    for k in tqdm.tqdm(range(num_of_simulations), desc="Simulation", position=0):
        # Inner loop
        for j in tqdm.tqdm(range(num_constituencies), desc="Constituency", position=1, leave=False):
            
            # Using random choice as per the snippet logic to sample from the distribution
            num_of_candidates = np.random.choice(candidates_arr)
            num_of_voters = np.random.choice(num_of_voters_array)
            
            prob = np.random.uniform(0, 1, num_of_candidates)
            prob = prob / np.sum(prob)
            
            votes_array = np.random.choice(num_of_candidates, num_of_voters, p=prob, replace=True)
            votes_candidates = np.zeros(num_of_candidates, dtype=int)
            
            for i in range(num_of_candidates):
                votes_candidates[i] = np.count_nonzero(votes_array == i)
            
            votes_sorted = np.sort(votes_candidates)[::-1]
            if len(votes_sorted) > 0:
                winner_votes = votes_sorted[0]
                model_winner_arr.append(winner_votes)
                
            if len(votes_sorted) > 1:
                runner_up_votes = votes_sorted[1]
                model_runner_up_arr.append(runner_up_votes)
            
            votes_arr.append(votes_sorted)
            new_turnout_arr.append(num_of_voters)
            
    return (np.array(model_winner_arr), 
            np.array(model_runner_up_arr), 
            votes_arr, 
            new_turnout_arr)

def get_mean_specific_margin():
    """
    Returns the analytical mean specific margin <mu>.
    <mu> = 1/2 + ln(9 * 3^0.25 / 16)
    """
    return 0.5 + np.log(9 * (3**0.25) / 16)

def p_mu(mu):
    """
    Analytical distribution P(mu) for specific margin mu.
    Eq (1): P(mu) = (1-mu)(5+7mu) / ((1+mu)^2 * (1+2mu)^2)
    """
    if mu <= 0 or mu >= 1:
        return 0.0
    return ((1 - mu) * (5 + 7 * mu)) / ((1 + mu)**2 * (1 + 2 * mu)**2)

def f_x(x):
    """
    Analytical distribution F(x) for scaled specific margin x = mu / <mu>.
    Eq (2): F(x) = <mu> * P(x * <mu>)
    """
    mean_mu = get_mean_specific_margin()
    mu = x * mean_mu
    # Since P(mu) is defined for 0 < mu < 1, we should handle boundaries if needed,
    # but for analytical plotting we usually pass valid ranges.
    return mean_mu * p_mu(mu)

def calculate_js_distance(empirical_data, analytical_func, bins=50):
    """
    Calculates the Jensen-Shannon distance between empirical data and an analytical PDF.
    
    Args:
        empirical_data: Array of empirical observations.
        analytical_func: Function that takes x and returns PDF value f(x).
        bins: Number of bins or array of bin edges.
        
    Returns:
        float: Jensen-Shannon distance.
    """
    # 1. Compute empirical distribution
    p, bin_edges = np.histogram(empirical_data, bins=bins, density=True)
    
    # 2. Compute analytical distribution on the same bins
    # We approximate the probability mass in each bin as f(center) * width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    
    q_unnormalized = np.array([analytical_func(x) for x in bin_centers])
    
    # Handle potential NaNs or infs in analytical function
    q_unnormalized = np.nan_to_num(q_unnormalized)
    
    # Normalize q to make it a probability distribution over the bins
    # Note: density=True in histogram makes sum(p * width) = 1.
    # But jensenshannon expects sum(p) = 1 if using base e, or just vectors.
    # Actually scipy.spatial.distance.jensenshannon checks if they sum to 1?
    # "If p and q sum to 1, this returns the Jensen-Shannon distance (metric)."
    # If we pass density values from histogram, they don't sum to 1 (they sum to 1/width).
    
    # Let's use probability mass instead of density for JSD calculation.
    p_mass, _ = np.histogram(empirical_data, bins=bin_edges, density=False)
    p_mass = p_mass / np.sum(p_mass)
    
    # Calculate q_mass
    q_mass = q_unnormalized * bin_widths
    # Normalize q_mass to ensure it sums to 1 (handling range truncation)
    if np.sum(q_mass) > 0:
        q_mass = q_mass / np.sum(q_mass)
    else:
        # Fallback if analytical is all zero in range (unlikely)
        q_mass = np.zeros_like(p_mass)
        
    return jensenshannon(p_mass, q_mass)
