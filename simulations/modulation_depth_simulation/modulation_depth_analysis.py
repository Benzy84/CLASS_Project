import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import LinearSegmentedColormap
from utils.image_processing import center_crop, compute_similarity_score
from utils.visualization import display_field
nrm = lambda x: x/np.abs(x).max()



# Define the custom colormap
def get_custom_colormap():
    # Define the colors for the colormap in RGB format
    colors = [
        (0, 0, 0),  # Black
        (0, 0.2, 0),  # Dark Green
        (0, 0.5, 0),  # Green
        (0, 0.8, 0),  # Bright Green
        (0.7, 1, 0),  # Light Green-Yellow
        (1, 1, 1)  # White
    ]
    # Define the positions for the colors in the colormap (0 to 1)
    positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # Create the colormap using LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('greenish_hot', list(zip(positions, colors)))


# Get the custom colormap
new_cmap = get_custom_colormap()


# ----- Data Loading Functions -----

def select_result_directory():
    """Select a modulation results directory using a file dialog"""
    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(title="Select Modulation Depth Results Directory")
    if not selected_dir:
        print("No directory selected")
        return None
    return selected_dir


def extract_params_from_readme(output_dir):
    """Extract parameters from README.txt file"""
    readme_path = os.path.join(output_dir, 'README.txt')

    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()

        # Extract parameters
        alpha_values = extract_param(readme_content, 'alpha_values')
        strategies = extract_param(readme_content, 'strategies')
        num_trials = int(extract_param(readme_content, 'num_trials')[0])

        return {
            'alpha_values': alpha_values,
            'strategies': strategies,
            'num_trials': num_trials
        }
    except FileNotFoundError:
        print(f"README.txt not found in {output_dir}")
        return None


def extract_param(readme_content, param_name):
    """Extract specific parameter from README content"""
    param_match = re.search(rf'PARAM:{param_name}=([\w.,]+)', readme_content)
    if param_match:
        values = param_match.group(1).split(',')
        try:
            # Try to convert to float if possible
            if param_name == 'alpha_values':
                return [float(x) for x in values]
            else:
                return [float(x) if '.' in x else x for x in values]
        except ValueError:
            return values

    # Default values for known parameters
    if param_name == 'alpha_values':
        return [0, 0.25, 0.5, 0.75, 1.0]
    return []


def load_all_reference_data(output_dir, trial=1):
    """
    Load all reference data: widefield, PSF, and aberrated image

    Args:
        output_dir: Path to results directory
        trial: Trial number to use (default=1)

    Returns:
        dict: Dictionary containing reference data
    """
    # Initialize reference data dictionary
    ref_data = {}

    # Load widefield
    try:
        ref_data['widefield'] = np.load(os.path.join(output_dir, 'widefield.npy'))
        ref_data['widefield'] = ref_data['widefield'] / np.max(np.abs(ref_data['widefield']))
        print("Loaded widefield reference")
    except FileNotFoundError:
        print("Could not find widefield reference")
        return None

    # Load PSF
    try:
        ref_data['psf'] = np.load(os.path.join(output_dir, f'PSF_trial_{trial}.npy'))
        ref_data['psf'] = ref_data['psf'] / np.max(np.abs(ref_data['psf']))
        print("Loaded PSF reference")
    except FileNotFoundError:
        print("Could not find PSF, using placeholder")
        ref_data['psf'] = None

    # Load aberrated image example
    try:
        ref_data['aberrated_img'] = np.load(os.path.join(output_dir, f'aberrated_image_trial_{trial}.npy'))
        ref_data['aberrated_img'] = ref_data['aberrated_img'] / np.max(np.abs(ref_data['aberrated_img']))
        print("Loaded aberrated image example")
    except FileNotFoundError:
        print("Could not find aberrated image, using placeholder")
        ref_data['aberrated_img'] = None

    return ref_data


def load_data_and_compute_scores(output_dir, ref_data, params):
    """
    Load all reconstructions and compute similarity scores

    Args:
        output_dir: Path to results directory
        ref_data: Dictionary containing reference data
        params: Dictionary with parameters

    Returns:
        dict: scores_dict containing all scores
    """
    print("Loading data and computing scores...")

    if ref_data is None or 'widefield' not in ref_data:
        print("Widefield reference is required")
        return None

    widefield = ref_data['widefield']

    # Initialize dictionary for scores
    scores_dict = {}

    # Set up progress tracking
    total_combinations = len(params['strategies']) * len(params['alpha_values']) * params['num_trials']
    processed = 0

    # Load reconstructions and compute scores
    for strategy in params['strategies']:
        scores_dict[strategy] = {}

        for alpha in params['alpha_values']:
            alpha_str = f'{float(alpha):.2f}'
            scores_dict[strategy][alpha] = []

            for trial in range(1, params['num_trials'] + 1):
                # Load reconstruction
                recon_file = os.path.join(output_dir, f'O_est_{strategy}_alpha_{alpha_str}_trial_{trial}.npy')
                recon_step_file = os.path.join(output_dir, f'O_est_step_alpha_1.00_trial_1.npy')
                recon_lr_file = os.path.join(output_dir, f'O_est_linear_ramp_alpha_1.00_trial_1.npy')

                try:
                    # Load and normalize reconstruction
                    recon = np.load(recon_file)
                    recon = recon / np.max(np.abs(recon))

                    score = compute_similarity_score(widefield, recon)
                    scores_dict[strategy][alpha].append(score)

                    processed += 1
                    if processed % 10 == 0 or processed == total_combinations:
                        print(f"Processed {processed}/{total_combinations} reconstructions")

                except FileNotFoundError:
                    print(f"File not found: {recon_file}")

            # Calculate average score for this strategy and alpha
            if scores_dict[strategy][alpha]:
                avg_score = np.mean(scores_dict[strategy][alpha])
                scores_dict[strategy][alpha].append(avg_score)

    return scores_dict


# ----- Plotting and Analysis Functions -----

def find_best_alpha(scores_dict, strategy):
    """
    Find the alpha value with highest average score for a given strategy

    Args:
        scores_dict: Dictionary with all scores
        strategy: Strategy name

    Returns:
        tuple: (best_alpha, best_score)
    """
    best_score = -float('inf')
    best_alpha = None

    for alpha in scores_dict[strategy].keys():
        if scores_dict[strategy][alpha]:
            # Get average score (last element in the list)
            avg_score = scores_dict[strategy][alpha][-1]

            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

    return best_alpha, best_score


def load_reconstruction(output_dir, strategy, alpha, trial=1):
    """
    Load a reconstruction for any strategy, alpha value, and trial

    Args:
        output_dir: Path to results directory
        strategy: Strategy name
        alpha: Alpha value
        trial: Trial number to use (default=1)

    Returns:
        numpy.ndarray: Reconstruction
    """
    alpha_str = f'{float(alpha):.2f}'
    recon_file = os.path.join(output_dir, f'O_est_{strategy}_alpha_{alpha_str}_trial_{trial}.npy')

    try:
        recon = np.load(recon_file)
        recon = recon / np.max(np.abs(recon))
        return recon
    except FileNotFoundError:
        print(f"Could not find reconstruction file: {recon_file}")
        return None


def load_modulation_pattern(output_dir, strategy, alpha):
    """
    Load modulation pattern for a given strategy and alpha

    Args:
        output_dir: Path to results directory
        strategy: Strategy name
        alpha: Alpha value

    Returns:
        numpy.ndarray: Modulation pattern
    """
    alpha_str = f'{float(alpha):.2f}'
    mod_file = os.path.join(output_dir, f'modulation_{strategy}_alpha_{alpha_str}.npy')

    try:
        modulation = np.load(mod_file)
        return modulation
    except FileNotFoundError:
        print(f"Could not find modulation file: {mod_file}")
        return None


def plot_strategy_modulation(output_dir, strategy, alphas, ax=None):
    """
    Plot modulation patterns for a strategy at different alpha values

    Args:
        output_dir: Path to results directory
        strategy: Strategy name
        alphas: List of alpha values
        ax: Matplotlib axis to plot on (optional)

    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Color map for different alpha values
    cmap = plt.cm.viridis
    alpha_colors = cmap(np.linspace(0, 1, len(alphas)))

    for i, alpha in enumerate(alphas):
        modulation = load_modulation_pattern(output_dir, strategy, alpha)

        if modulation is not None:
            # Plot the pattern
            ax.plot(
                range(1, len(modulation) + 1),
                modulation,
                color=alpha_colors[i],
                label=f'α={float(alpha):.2f}',
                linewidth=2
            )

    # Set labels and grid
    ax.set_xlabel('Realization Index')
    ax.set_ylabel('Weight')
    ax.set_title(f'Modulation Pattern: {strategy}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_alpha_scores(scores_dict, strategy, ax=None):
    """
    Plot scores vs alpha for a given strategy

    Args:
        scores_dict: Dictionary with all scores
        strategy: Strategy name
        ax: Matplotlib axis to plot on (optional)

    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    alphas = sorted(scores_dict[strategy].keys())
    avg_scores = []

    for alpha in alphas:
        if scores_dict[strategy][alpha]:
            avg_scores.append(scores_dict[strategy][alpha][-1])  # Last element is the average
        else:
            avg_scores.append(np.nan)

    ax.plot(alphas, avg_scores, 'o-', linewidth=2, color='blue')
    ax.set_xlabel('Modulation Depth (α)')
    ax.set_ylabel('Average Score')
    ax.set_title(f'Score vs Alpha: {strategy}')
    ax.grid(True, alpha=0.3)

    return ax


def create_strategy_figure(output_dir, ref_data, scores_dict, strategy, params):
    """
    Create a figure for a specific strategy showing best reconstruction and metrics

    Args:
        output_dir: Path to results directory
        ref_data: Dictionary containing reference data
        scores_dict: Dictionary with all scores
        strategy: Strategy name
        params: Parameters dictionary

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get reference data
    widefield = ref_data['widefield']
    psf = ref_data.get('psf')
    aberrated_img = ref_data.get('aberrated_img')

    # Find best alpha for this strategy
    best_alpha, best_score = find_best_alpha(scores_dict, strategy)
    if best_alpha is None:
        print(f"No valid scores found for strategy: {strategy}")
        return None

    # Load best reconstruction
    best_recon = load_reconstruction(output_dir, strategy, best_alpha)
    best_recon -= np.min(best_recon)
    best_recon = nrm(best_recon)
    if best_recon is None:
        return None

    alphas = params['alpha_values']
    # alphas = alphas[::2]
    plt.figure()
    for idx, alpha in enumerate(alphas):
        img = load_reconstruction(output_dir, strategy, alpha)
        img -= np.min(img)
        img = nrm(img)
        score = np.round(scores_dict[strategy][alpha][-1], 4)
        plt.subplot(2,7, idx+1)
        plt.imshow(img, cmap=new_cmap)
        if idx == 0:
            plt.colorbar()
        plt.title(f'score:{score}')
    plt.show()


    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Strategy: {strategy} (Best α={best_alpha:.2f}, Score={best_score:.4f})', fontsize=16)

    # Row 1: Images
    # Widefield
    ax1 = plt.subplot(2, 3, 1)
    plt.title('Widefield Reference')
    im1 = plt.imshow(widefield, cmap=new_cmap)
    plt.colorbar(im1)
    plt.axis('off')

    # PSF
    ax2 = plt.subplot(2, 3, 2)
    plt.title('PSF')
    if psf is not None:
        im2 = plt.imshow(np.abs(psf), cmap='viridis')
        plt.colorbar(im2)
    else:
        ax2.text(0.5, 0.5, 'PSF not available', ha='center', va='center')
    plt.axis('off')

    # Aberrated Image
    ax3 = plt.subplot(2, 3, 3)
    plt.title('Aberrated Image')
    if aberrated_img is not None:
        im3 = plt.imshow(aberrated_img, cmap=new_cmap)
        plt.colorbar(im3)
    else:
        ax3.text(0.5, 0.5, 'Aberrated image not available', ha='center', va='center')
    plt.axis('off')

    # Row 2: Reconstruction, scores, modulation
    # Reconstruction
    ax4 = plt.subplot(2, 3, 4)
    plt.title(f'Best Reconstruction (α={best_alpha:.2f})')
    im4 = plt.imshow(best_recon, cmap=new_cmap)
    plt.colorbar(im4)
    plt.axis('off')

    # Scores vs Alpha
    ax5 = plt.subplot(2, 3, 5)
    plot_alpha_scores(scores_dict, strategy, ax=ax5)

    # Modulation pattern
    ax6 = plt.subplot(2, 3, 6)
    plot_strategy_modulation(output_dir, strategy, params['alpha_values'], ax=ax6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    # Save figure
    save_path = os.path.join(output_dir, f'{strategy}_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

    return fig


def find_all_best_reconstructions(output_dir, scores_dict, params):
    """
    Find best reconstructions for all strategies

    Args:
        output_dir: Path to results directory
        scores_dict: Dictionary with all scores
        params: Parameters dictionary

    Returns:
        tuple: (best_recons, best_scores, best_alphas)
    """
    best_recons = {}
    best_scores = {}
    best_alphas = {}

    for strategy in params['strategies']:
        best_alpha, best_score = find_best_alpha(scores_dict, strategy)
        if best_alpha is not None:
            best_recon = load_reconstruction(output_dir, strategy, best_alpha)
            best_recon -= np.min(best_recon)
            best_recon = nrm(best_recon)
            if best_recon is not None:
                best_recons[strategy] = best_recon
                best_scores[strategy] = best_score
                best_alphas[strategy] = best_alpha

    return best_recons, best_scores, best_alphas


def create_combined_figure(output_dir, widefield, best_recons, best_scores, best_alphas, params):
    """
    Create a figure showing the best reconstruction for each strategy
    and use alpha=0 from the first strategy as reference when 'none' is not present

    Args:
        output_dir: Path to results directory
        widefield: Widefield reference image
        best_recons: Dictionary of best reconstructions
        best_scores: Dictionary of best scores
        best_alphas: Dictionary of best alphas
        params: Parameters dictionary with strategies and alpha values

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Check if 'none' strategy is missing and alpha=0 is in the alpha values
    show_alpha_zero = 'none' not in params['strategies'] and 1 in params['alpha_values']

    # Determine grid dimensions
    n_strategies = len(best_recons)
    # If we're adding alpha=0 reference, add one more to the count
    n_total = n_strategies + 1 + (1 if show_alpha_zero else 0)  # +1 for widefield + optional alpha=0
    n_cols = min(5, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    # Create figure
    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    plt.suptitle('Widefield Reference and Best Reconstructions by Strategy', fontsize=16)

    # Plot widefield reference first
    ax1 = plt.subplot(n_rows, n_cols, 1)
    plt.title('Widefield Reference')
    im1 = plt.imshow(widefield, cmap=new_cmap)
    plt.colorbar(im1)
    plt.axis('off')

    plot_position = 2  # Start plotting reconstructions at position 2

    # Add alpha=0 reference if needed
    if show_alpha_zero:
        # Take the first strategy from the list
        first_strategy = params['strategies'][0]

        # Load alpha=0 reconstruction from the first strategy
        alpha_zero_recon = load_reconstruction(output_dir, first_strategy, 1, trial=1)

        if alpha_zero_recon is not None:
            alpha_zero_recon -= np.min(alpha_zero_recon)
            alpha_zero_recon = nrm(alpha_zero_recon)
            # Calculate score for this reconstruction
            alpha_zero_score = compute_similarity_score(widefield, alpha_zero_recon)

            # Plot it
            ax = plt.subplot(n_rows, n_cols, plot_position)
            plt.title(f'No Modulation (α=0)\nscore={alpha_zero_score:.4f}')
            im = plt.imshow(alpha_zero_recon, cmap=new_cmap)
            plt.colorbar(im)
            plt.axis('off')

            plot_position += 1  # Increment position for next plots

    # Plot each best reconstruction
    for i, strategy in enumerate(best_recons.keys()):
        ax = plt.subplot(n_rows, n_cols, plot_position + i)
        plt.title(f'{strategy}\nα={best_alphas[strategy]:.2f}, score={best_scores[strategy]:.4f}')
        im = plt.imshow(best_recons[strategy], cmap=new_cmap)
        plt.colorbar(im)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    # Save figure
    save_path = os.path.join(output_dir, 'best_reconstructions_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison figure to {save_path}")

    return plt.gcf()

def analyze_and_plot_all(output_dir, ref_data, scores_dict, params):
    """
    Run the full analysis and create all plots

    Args:
        output_dir: Path to results directory
        ref_data: Dictionary containing reference data
        scores_dict: Dictionary with all scores
        params: Parameters dictionary
    """
    # 1. Create individual strategy figures
    for strategy in params['strategies']:
        create_strategy_figure(output_dir, ref_data, scores_dict, strategy, params)

    # 2. Create combined figure with best reconstructions
    best_recons, best_scores, best_alphas = find_all_best_reconstructions(
        output_dir, scores_dict, params
    )
    # Pass params to the updated function
    create_combined_figure(output_dir, ref_data['widefield'], best_recons, best_scores, best_alphas, params)

    # Show all figures
    plt.show()

# ----- Main execution -----

def main():
    """Main function to run the entire analysis"""
    # Select results directory
    output_dir = select_result_directory()
    if not output_dir:
        return

    # Extract parameters
    params = extract_params_from_readme(output_dir)
    if not params:
        return

    print(f"Parameters: {len(params['strategies'])} strategies, "
          f"{len(params['alpha_values'])} alpha values, "
          f"{params['num_trials']} trials")

    # Load reference data
    ref_data = load_all_reference_data(output_dir)
    if ref_data is None:
        return

    # Load data and compute scores
    scores_dict = load_data_and_compute_scores(output_dir, ref_data, params)
    if not scores_dict:
        return

    # Print a summary of the scores
    print("\nScore Summary:")
    for strategy in scores_dict:
        print(f"\nStrategy: {strategy}")
        for alpha in sorted(scores_dict[strategy].keys()):
            if scores_dict[strategy][alpha]:
                avg_score = scores_dict[strategy][alpha][-1]  # Last element is the average
                print(f"  Alpha={alpha:.2f}: Average score={avg_score:.4f}")

    # Run analysis and create all plots
    analyze_and_plot_all(output_dir, ref_data, scores_dict, params)


if __name__ == "__main__":
    main()