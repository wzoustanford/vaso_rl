import matplotlib.pyplot as plt
import numpy as np
from fqe_gaussian_analysis import FQEGaussianAnalysis

import pickle
import numpy as np

# Assuming patient_q_values and clinician_q_values are already defined
# Example data (replace with your actual data):
# patient_q_values = np.array([...])
# clinician_q_values = np.array([...])

def save_q_values_to_pickle(model_q_values, clinician_q_values, filename='q_values.pkl'):
    """
    Save patient and clinician Q-values to a pickle file.
    
    Parameters:
    -----------
    patient_q_values : array-like
        Q-values from the patient model
    clinician_q_values : array-like
        Q-values from the clinician
    filename : str
        Name of the pickle file to save
    """
    data = {
        'patient_q_values': model_q_values,
        'clinician_q_values': clinician_q_values
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Q-values saved to {filename}")

# Load function for later use
def load_q_values_from_pickle(filename='q_values.pkl'):
    """
    Load Q-values from pickle file.
    
    Parameters:
    -----------
    filename : str
        Name of the pickle file to load
    
    Returns:
    --------
    dict : Dictionary containing patient_q_values and clinician_q_values
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    model_q_values = data['patient_q_values']
    clinician_q_values = data['clinician_q_values']

    return model_q_values, clinician_q_values

def plot_q_value_histograms(patient_q_values, clinician_q_values, bins=30, alpha=0.6):
    """
    Plot overlapping histograms of patient (model) and clinician Q-values.
    
    Parameters:
    -----------
    patient_q_values : array-like
        Q-values from the patient model (shown in red)
    clinician_q_values : array-like
        Q-values from the clinician (shown in blue)
    bins : int
        Number of bins for the histogram
    alpha : float
        Transparency level for overlapping histograms
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(patient_q_values, bins=bins, alpha=alpha, color='red', 
            label='Model Q-values', density=True, edgecolor='darkred')
    ax.hist(clinician_q_values, bins=bins, alpha=alpha, color='blue', 
            label='Clinician Q-values', density=True, edgecolor='darkblue')
    
    # Add labels and title
    ax.set_xlabel('Q-values', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Distribution of Q-values: Model vs Clinician', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    model_mean = np.mean(patient_q_values)
    model_std = np.std(patient_q_values)
    clinician_mean = np.mean(clinician_q_values)
    clinician_std = np.std(clinician_q_values)
    
    stats_text = f'Model: μ={model_mean:.3f}, σ={model_std:.3f}\n'
    stats_text += f'Clinician: μ={clinician_mean:.3f}, σ={clinician_std:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig

def save_histogram_plot(patient_q_values, clinician_q_values, 
                        filename='q_values_histogram.png', dpi=300, show_plot=False):
    """
    Plot and save Q-value histograms to a file.
    
    Parameters:
    -----------
    patient_q_values : array-like
        Q-values from the patient model
    clinician_q_values : array-like
        Q-values from the clinician
    filename : str
        Name of the file to save the plot
    dpi : int
        Resolution of the saved figure
    show_plot : bool
        Whether to display the plot (set False for remote servers)
    """
    fig = plot_q_value_histograms(patient_q_values, clinician_q_values)
    
    # Save the figure
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Histogram plot saved to {filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory on remote server
    
    return fig

# Example usage:
# fig = save_histogram_plot(patient_q_values, clinician_q_values, 
#                          filename='q_values_histogram.png', show_plot=False)

if __name__ == "__main__":
    model_type = 'stepwise'
    model_type = 'block_discrete'
    model_type = 'lsmt_BD'
    alpha=0.0001
    vp2_bins=10
    max_step=0.2
    file_str = f'q_values_model_{model_type}_alpha{alpha}_max_step{max_step}'
    file_str = f'q_values_model_{model_type}_alpha{alpha}_bins{vp2_bins}'
    
    all_model_q_values, all_clinician_q_values = load_q_values_from_pickle(file_str+'.pkl')
    save_q_values_to_pickle(all_model_q_values, all_clinician_q_values, file_str + '.pkl')
    save_histogram_plot(all_model_q_values, all_clinician_q_values, 
                        filename=file_str + '.png', dpi=300, show_plot=True)
    
    fqe_analyzer = FQEGaussianAnalysis(all_model_q_values, all_clinician_q_values)
    # Save comprehensive 4-panel plot
    fig = fqe_analyzer.save_analysis_plots(filename=file_str+'fqe_complete_analysis.png', show_plot=True)

    # Save individual plots
    fqe_analyzer.save_individual_plots(prefix=file_str+'_fqe', show_plots=True)

    # Get specific probability of improvement
    prob_improvement = fqe_analyzer.compute_probability_improvement()
    print(f"{file_str} Probability of model improvement over clinician mean: {prob_improvement:.3f}")
