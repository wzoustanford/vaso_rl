import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

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
    
    return data

# Example usage:
# save_q_values_to_pickle(patient_q_values, clinician_q_values)
# loaded_data = load_q_values_from_pickle()
# patient_q_values = loaded_data['patient_q_values']
# clinician_q_values = loaded_data['clinician_q_values']

import matplotlib.pyplot as plt
import numpy as np

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

class FQEGaussianAnalysis:
    """
    Fitted Q Evaluation with Gaussian fitting for Q-value distributions.
    """
    
    def __init__(self, patient_q_values, clinician_q_values):
        self.patient_q_values = np.array(patient_q_values)
        self.clinician_q_values = np.array(clinician_q_values)
        
        # Fit Gaussians to both distributions
        self.patient_params = self.fit_gaussian(patient_q_values)
        self.clinician_params = self.fit_gaussian(clinician_q_values)
        
    def fit_gaussian(self, q_values):
        """
        Fit a Gaussian distribution to Q-values using Maximum Likelihood Estimation.
        
        Parameters:
        -----------
        q_values : array-like
            Q-values to fit
            
        Returns:
        --------
        tuple : (mean, std) parameters of the fitted Gaussian
        """
        mean = np.mean(q_values)
        std = np.std(q_values, ddof=1)  # Use sample standard deviation
        return mean, std
    
    def gaussian_pdf(self, x, mean, std):
        """Calculate Gaussian PDF."""
        return stats.norm.pdf(x, mean, std)
    
    def gaussian_cdf(self, x, mean, std):
        """Calculate Gaussian CDF (sigmoid for Gaussian)."""
        return stats.norm.cdf(x, mean, std)
    
    def compute_probability_improvement(self, x_value=None):
        """
        Compute probability of improvement at a given Q-value threshold.
        
        Parameters:
        -----------
        x_value : float or None
            Q-value threshold. If None, uses clinician mean.
            
        Returns:
        --------
        float : Probability that model Q-value exceeds threshold
        """
        if x_value is None:
            x_value = self.clinician_params[0]  # Use clinician mean as threshold
        
        # Probability that model Q-value exceeds threshold
        prob_improvement = 1 - self.gaussian_cdf(x_value, 
                                                  self.patient_params[0], 
                                                  self.patient_params[1])
        return prob_improvement
    
    def plot_analysis(self):
        """
        Create comprehensive plots for FQE analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Fitted Gaussians overlaid on histograms
        ax1 = axes[0, 0]
        x_range = np.linspace(
            min(self.patient_q_values.min(), self.clinician_q_values.min()) - 1,
            max(self.patient_q_values.max(), self.clinician_q_values.max()) + 1,
            300
        )
        
        # Plot histograms
        ax1.hist(self.patient_q_values, bins=30, alpha=0.5, color='red', 
                density=True, label='Model Q-values (data)')
        ax1.hist(self.clinician_q_values, bins=30, alpha=0.5, color='blue', 
                density=True, label='Clinician Q-values (data)')
        
        # Plot fitted Gaussians
        model_pdf = self.gaussian_pdf(x_range, *self.patient_params)
        clinician_pdf = self.gaussian_pdf(x_range, *self.clinician_params)
        
        ax1.plot(x_range, model_pdf, 'r-', linewidth=2, 
                label=f'Model Gaussian (μ={self.patient_params[0]:.3f}, σ={self.patient_params[1]:.3f})')
        ax1.plot(x_range, clinician_pdf, 'b-', linewidth=2, 
                label=f'Clinician Gaussian (μ={self.clinician_params[0]:.3f}, σ={self.clinician_params[1]:.3f})')
        
        ax1.set_xlabel('Q-values')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Fitted Gaussian Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. CDF plots (Sigmoid for Gaussian)
        ax2 = axes[0, 1]
        model_cdf = self.gaussian_cdf(x_range, *self.patient_params)
        clinician_cdf = self.gaussian_cdf(x_range, *self.clinician_params)
        
        ax2.plot(x_range, model_cdf, 'r-', linewidth=2, label='Model CDF')
        ax2.plot(x_range, clinician_cdf, 'b-', linewidth=2, label='Clinician CDF')
        
        # Mark the means
        ax2.axvline(self.patient_params[0], color='red', linestyle='--', alpha=0.5, label='Model mean')
        ax2.axvline(self.clinician_params[0], color='blue', linestyle='--', alpha=0.5, label='Clinician mean')
        
        ax2.set_xlabel('Q-values')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Functions (CDFs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement Gap Visualization
        ax3 = axes[1, 0]
        
        # Calculate improvement probabilities across range
        improvement_probs = [1 - self.gaussian_cdf(x, *self.patient_params) 
                            for x in x_range]
        
        ax3.plot(x_range, improvement_probs, 'g-', linewidth=2)
        ax3.fill_between(x_range, 0, improvement_probs, alpha=0.3, color='green')
        
        # Mark clinician mean
        clinician_mean_prob = self.compute_probability_improvement()
        ax3.axvline(self.clinician_params[0], color='blue', linestyle='--', 
                   label=f'Clinician mean (P(improvement)={clinician_mean_prob:.3f})')
        ax3.axhline(clinician_mean_prob, color='gray', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Q-value Threshold')
        ax3.set_ylabel('P(Model Q > Threshold)')
        ax3.set_title('Probability of Model Improvement over Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Improvement Analysis Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate key metrics
        mean_diff = self.patient_params[0] - self.clinician_params[0]
        prob_improvement = self.compute_probability_improvement()
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((self.patient_params[1]**2 + self.clinician_params[1]**2) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Create summary text
        summary_text = "FQE Analysis Summary\n" + "="*30 + "\n\n"
        summary_text += f"Model Mean Q-value: {self.patient_params[0]:.4f}\n"
        summary_text += f"Model Std Dev: {self.patient_params[1]:.4f}\n\n"
        summary_text += f"Clinician Mean Q-value: {self.clinician_params[0]:.4f}\n"
        summary_text += f"Clinician Std Dev: {self.clinician_params[1]:.4f}\n\n"
        summary_text += "Improvement Metrics\n" + "-"*20 + "\n"
        summary_text += f"Mean Improvement Gap: {mean_diff:.4f}\n"
        summary_text += f"P(Model > Clinician Mean): {prob_improvement:.3f}\n"
        summary_text += f"Cohen's d (Effect Size): {cohens_d:.3f}\n\n"
        
        # Interpretation
        if cohens_d < 0.2:
            effect_interpretation = "Small effect"
        elif cohens_d < 0.5:
            effect_interpretation = "Medium effect"
        elif cohens_d < 0.8:
            effect_interpretation = "Large effect"
        else:
            effect_interpretation = "Very large effect"
        
        summary_text += f"Effect Size Interpretation: {effect_interpretation}\n"
        
        if mean_diff > 0:
            summary_text += "\n✓ Model shows higher average Q-values than clinician"
        else:
            summary_text += "\n✗ Clinician shows higher average Q-values than model"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontfamily='monospace')
        
        plt.suptitle('FQE Gaussian Analysis: Model vs Clinician Q-values', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_analysis_plots(self, filename='fqe_analysis.png', dpi=300, show_plot=False):
        """
        Create and save the comprehensive FQE analysis plots.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the plot
        dpi : int
            Resolution of the saved figure
        show_plot : bool
            Whether to display the plot (set False for remote servers)
        
        Returns:
        --------
        fig : matplotlib figure object
        """
        fig = self.plot_analysis()
        
        # Save the figure
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"FQE analysis plot saved to {filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory on remote server
        
        return fig
    
    def save_individual_plots(self, prefix='fqe', dpi=300, show_plots=False):
        """
        Create and save individual plots for each analysis component.
        
        Parameters:
        -----------
        prefix : str
            Prefix for saved file names
        dpi : int
            Resolution of the saved figures
        show_plots : bool
            Whether to display plots (set False for remote servers)
        """
        import matplotlib
        if not show_plots:
            matplotlib.use('Agg')  # Use non-interactive backend for remote servers
        
        # Plot 1: Fitted Gaussians
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x_range = np.linspace(
            min(self.patient_q_values.min(), self.clinician_q_values.min()) - 1,
            max(self.patient_q_values.max(), self.clinician_q_values.max()) + 1,
            300
        )
        
        ax1.hist(self.patient_q_values, bins=30, alpha=0.5, color='red', 
                density=True, label='Model Q-values (data)')
        ax1.hist(self.clinician_q_values, bins=30, alpha=0.5, color='blue', 
                density=True, label='Clinician Q-values (data)')
        
        model_pdf = self.gaussian_pdf(x_range, *self.patient_params)
        clinician_pdf = self.gaussian_pdf(x_range, *self.clinician_params)
        
        ax1.plot(x_range, model_pdf, 'r-', linewidth=2, 
                label=f'Model Gaussian (μ={self.patient_params[0]:.3f}, σ={self.patient_params[1]:.3f})')
        ax1.plot(x_range, clinician_pdf, 'b-', linewidth=2, 
                label=f'Clinician Gaussian (μ={self.clinician_params[0]:.3f}, σ={self.clinician_params[1]:.3f})')
        
        ax1.set_xlabel('Q-values')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Fitted Gaussian Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig(f'{prefix}_fitted_gaussians.png', dpi=dpi, bbox_inches='tight')
        print(f"Saved: {prefix}_fitted_gaussians.png")
        if not show_plots:
            plt.close(fig1)
        
        # Plot 2: CDFs
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        model_cdf = self.gaussian_cdf(x_range, *self.patient_params)
        clinician_cdf = self.gaussian_cdf(x_range, *self.clinician_params)
        
        ax2.plot(x_range, model_cdf, 'r-', linewidth=2, label='Model CDF')
        ax2.plot(x_range, clinician_cdf, 'b-', linewidth=2, label='Clinician CDF')
        ax2.axvline(self.patient_params[0], color='red', linestyle='--', alpha=0.5, label='Model mean')
        ax2.axvline(self.clinician_params[0], color='blue', linestyle='--', alpha=0.5, label='Clinician mean')
        
        ax2.set_xlabel('Q-values')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Functions (CDFs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(f'{prefix}_cdfs.png', dpi=dpi, bbox_inches='tight')
        print(f"Saved: {prefix}_cdfs.png")
        if not show_plots:
            plt.close(fig2)
        
        # Plot 3: Improvement Probability
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        improvement_probs = [1 - self.gaussian_cdf(x, *self.patient_params) 
                            for x in x_range]
        
        ax3.plot(x_range, improvement_probs, 'g-', linewidth=2)
        ax3.fill_between(x_range, 0, improvement_probs, alpha=0.3, color='green')
        
        clinician_mean_prob = self.compute_probability_improvement()
        ax3.axvline(self.clinician_params[0], color='blue', linestyle='--', 
                   label=f'Clinician mean (P(improvement)={clinician_mean_prob:.3f})')
        ax3.axhline(clinician_mean_prob, color='gray', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Q-value Threshold')
        ax3.set_ylabel('P(Model Q > Threshold)')
        ax3.set_title('Probability of Model Improvement over Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        fig3.savefig(f'{prefix}_improvement_probability.png', dpi=dpi, bbox_inches='tight')
        print(f"Saved: {prefix}_improvement_probability.png")
        if not show_plots:
            plt.close(fig3)
        
        if show_plots:
            plt.show()

# Example usage for remote servers:
# Initialize with your Q-values
# fqe_analyzer = FQEGaussianAnalysis(patient_q_values, clinician_q_values)

# Save comprehensive 4-panel plot
# fig = fqe_analyzer.save_analysis_plots(filename='fqe_complete_analysis.png', show_plot=False)

# Save individual plots
# fqe_analyzer.save_individual_plots(prefix='fqe', show_plots=False)

# Get specific probability of improvement
# prob_improvement = fqe_analyzer.compute_probability_improvement()
# print(f"Probability of model improvement over clinician mean: {prob_improvement:.3f}")