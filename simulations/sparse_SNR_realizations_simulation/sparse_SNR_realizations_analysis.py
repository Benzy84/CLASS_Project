import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tkinter as tk
import re
from tkinter import filedialog
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.image_processing import compute_similarity_score


def select_result_directory():
    """Let user select a simulation results directory"""
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a directory
    selected_dir = filedialog.askdirectory(
        title="Select Simulation Results Directory"
    )

    if not selected_dir:
        print("No directory selected")
        return None

    return selected_dir


def detect_simulation_type(results_dir):
    """Detect if this is coherent or incoherent simulation based on directory name or files"""
    dir_name = os.path.basename(results_dir).lower()

    # Check directory name first - use exact matching to avoid substring issues
    if dir_name.startswith("coherent_") or "_coherent_" in dir_name:
        print(f"Detected coherent simulation based on directory name: {dir_name}")
        return True
    elif dir_name.startswith("incoherent_") or "_incoherent_" in dir_name:
        print(f"Detected incoherent simulation based on directory name: {dir_name}")
        return False

    # If directory name is ambiguous, check for file types
    snr_dirs = glob.glob(os.path.join(results_dir, "SNR_*"))
    if not snr_dirs:
        print("No SNR directories found!")
        return False  # Default to incoherent if cannot detect

    # Check the first SNR directory for E_ files (coherent) or I_ files (incoherent)
    e_files = glob.glob(os.path.join(snr_dirs[0], "E_*.npy"))
    i_files = glob.glob(os.path.join(snr_dirs[0], "I_*.npy"))

    if e_files and not i_files:
        print("Detected coherent simulation based on E_ files")
        return True
    elif i_files and not e_files:
        print("Detected incoherent simulation based on I_ files")
        return False

    # If still ambiguous, default to incoherent
    print("Could not definitively detect simulation type, defaulting to incoherent")
    return False


def extract_params_from_readme(results_dir):
    """Extract sz and padding_size from README.txt"""
    readme_path = os.path.join(results_dir, "README.txt")
    if not os.path.exists(readme_path):
        print(f"Warning: README.txt not found in {results_dir}")
        return 30, 10  # Default values if README doesn't exist

    try:
        with open(readme_path, 'r') as f:
            content = f.read()

        # Extract image size and padding size using regex
        sz_match = re.search(r'Image size: (\d+)x(\d+) pixels', content)
        if sz_match:
            sz = int(sz_match.group(1))  # Use the first number
        else:
            print("Warning: Could not find image size in README.txt")
            sz = 30  # Default value

        padding_match = re.search(r'Padding size: (\d+) pixels', content)
        if padding_match:
            padding_size = int(padding_match.group(1))
        else:
            print("Warning: Could not find padding size in README.txt")
            padding_size = 10  # Default value

        print(f"Extracted parameters: sz={sz}, padding_size={padding_size}")
        return sz, padding_size

    except Exception as e:
        print(f"Error reading README.txt: {e}")
        return 30, 10  # Default values


def center_crop(img, crop_size):
    """Apply center crop to image"""
    if img.shape[0] <= crop_size or img.shape[1] <= crop_size:
        return img  # Image already smaller than or equal to crop size

    start_y = (img.shape[0] - crop_size) // 2
    start_x = (img.shape[1] - crop_size) // 2

    return img[start_y:start_y + crop_size, start_x:start_x + crop_size]


def calculate_metrics(results_dir, is_coherent=False):
    """
    Calculate quality metrics for all SNRs, n and M values

    Args:
        results_dir: Main results directory
        is_coherent: Whether this is coherent data (affects how widefield is processed)

    Returns:
        Dictionary with metrics and parameters
    """
    print(f"Calculating metrics for {results_dir}...")

    sz, padding_size = extract_params_from_readme(results_dir)

    # Dictionary to store all results
    results = {}

    # Load simulation parameters
    try:
        ns = np.load(os.path.join(results_dir, "ns.npy"))
        mask = ns != 65536
        ns_filtered = ns[mask]
        ns = ns_filtered
        Ms = np.load(os.path.join(results_dir, "Ms.npy"))
        snrs = np.load(os.path.join(results_dir, "SNRs.npy"))
        print(f"Loaded parameters: {len(ns)} n values, {len(Ms)} M values, {len(snrs)} SNR values")
    except FileNotFoundError as e:
        print(f"Error: Missing parameter files in {results_dir}: {e}")
        return None

    # Store parameters in results
    results['ns'] = ns
    results['Ms'] = Ms
    results['SNRs'] = snrs

    # Get SNR directories
    snr_dirs = []
    for snr in snrs:
        # Format SNR for directory name
        if np.isinf(snr):
            snr_name = "SNR_inf"
        else:
            snr_name = f"SNR_{int(snr) if snr.is_integer() else snr}"

        snr_dir = os.path.join(results_dir, snr_name)
        if os.path.isdir(snr_dir):
            snr_dirs.append(snr_dir)
        else:
            print(f"Warning: Directory not found for {snr_name}")
            # Create a placeholder in results for this SNR
            snr_dirs.append(None)

    results['SNR_dirs'] = [d for d in snr_dirs if d is not None]

    # Process each SNR directory
    metrics_by_snr = {}
    for i, snr in enumerate(snrs):
        snr_dir = snr_dirs[i]
        if snr_dir is None:
            continue

        print(f"Processing SNR = {snr if not np.isinf(snr) else 'inf'}")

        # Initialize metric arrays
        ccs_array = np.zeros((len(ns), len(Ms)))

        # Find all trials
        unique_trials = set()
        for file_path in glob.glob(os.path.join(snr_dir, "O_est_*_*_*.npy")):
            file_name = os.path.basename(file_path)
            parts = file_name.replace("O_est_", "").replace(".npy", "").split("_")
            if len(parts) == 3:
                try:
                    trial = int(parts[0])
                    unique_trials.add(trial)
                except ValueError:
                    pass

        trials = sorted(list(unique_trials))
        if not trials:
            print(f"Warning: No valid trials found in {snr_dir}")
            continue

        print(f"Found {len(trials)} trials: {trials}")

        # Calculate metrics for each parameter combination
        for n_idx, n in enumerate(ns):
            for m_idx, m in enumerate(Ms):
                # Process all trials for this combination
                cc_values = []

                for trial in trials:
                    try:
                        # Load widefield reference and reconstruction
                        widefield_path = os.path.join(snr_dir, f"widefield_{trial}_{n}.npy")
                        recon_path = os.path.join(snr_dir, f"O_est_{trial}_{n}_{m}.npy")

                        if not os.path.exists(widefield_path) or not os.path.exists(recon_path):
                            continue

                        widefield = np.load(widefield_path)
                        recon = np.load(recon_path)

                        # Normalize images for better comparison
                        widefield = widefield / np.max(widefield)
                        recon = recon / np.max(recon)

                        crop_zize = sz - 2 * padding_size + 10
                        scoring_widefield = center_crop(widefield, crop_zize)
                        scoring_recon = center_crop(recon, crop_zize)

                        # For coherent data, we need to square the amplitude
                        if is_coherent:
                            scoring_widefield = np.abs(scoring_widefield) ** 2
                            scoring_recon = np.abs(scoring_recon) ** 2

                        # Ensure same shape
                        if widefield.shape != recon.shape:
                            min_h = min(widefield.shape[0], recon.shape[0])
                            min_w = min(widefield.shape[1], recon.shape[1])
                            widefield = widefield[:min_h, :min_w]
                            recon = recon[:min_h, :min_w]

                        # Calculate cross-correlation
                        cc = compute_similarity_score(scoring_widefield, scoring_recon)
                        cc_values.append(cc)
                    except Exception as e:
                        print(f"Error processing n={n}, M={m}, trial={trial}: {e}")
                        continue

                # Store average metrics
                if cc_values:
                    ccs_array[n_idx, m_idx] = np.mean(cc_values)
                else:
                    ccs_array[n_idx, m_idx] = np.nan

        # Store metrics for this SNR
        metrics_by_snr[snr] = {'ccs': ccs_array}

        # Save metrics to SNR directory
        np.save(os.path.join(snr_dir, "ccs.npy"), ccs_array)

    results['metrics'] = metrics_by_snr
    return results


def create_heatmaps(results, results_dir):
    """
    Create heatmaps of CC scores for each SNR arranged in a grid

    Args:
        results: Dictionary with metrics from calculate_metrics()
        results_dir: Directory to save plots to
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    snrs = results['SNRs']
    metrics_by_snr = results['metrics']
    ns = results['ns']
    Ms = results['Ms']

    # Count how many SNRs we'll actually plot (those with data)
    valid_snrs = [snr for snr in snrs if snr in metrics_by_snr]
    num_snrs = len(valid_snrs)

    if num_snrs == 0:
        print("No valid SNR data to plot")
        return

    # Determine grid dimensions based on number of SNRs
    if num_snrs <= 3:
        # For 1-3 SNRs, use a single row
        nrows, ncols = 1, num_snrs
    elif num_snrs <= 6:
        # For 4-6 SNRs, use 2 rows
        nrows, ncols = 2, (num_snrs + 1) // 2
    else:
        # For more than 6, use 3 rows
        nrows, ncols = 3, (num_snrs + 2) // 3

    # Create the figure
    fig_width = min(15, 5 * ncols)  # Cap width at 15 inches
    fig_height = min(12, 4 * nrows)  # Cap height at 12 inches

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    # Handle case where grid has only one element
    if num_snrs == 1:
        axes = np.array([axes])

    # Make axes accessible as a flattened array
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Title the figure
    fig.suptitle('Cross-Correlation Scores Across Parameters', fontsize=16)

    # Set common color range
    vmin, vmax = 0, 1  # CC scores are between 0 and 1

    # Create heatmaps
    for i, snr in enumerate(valid_snrs):
        if i < len(axes_flat):  # Make sure we don't exceed available axes
            ax = axes_flat[i]

            cc_data = metrics_by_snr[snr]['ccs']

            # Check for NaN values
            if np.isnan(cc_data).any():
                print(f"Warning: NaN values found in CC data for SNR={snr}")
                # Replace NaNs with 0 for visualization
                cc_data = np.nan_to_num(cc_data, nan=0.0)

            # Create heatmap
            im = ax.imshow(cc_data, cmap='hot', aspect='auto',
                           interpolation='nearest', origin='lower',
                           vmin=vmin, vmax=vmax)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Set labels
            ax.set_xlabel('Number of Realizations (M)')
            ax.set_ylabel('Number of Points (n)')

            # Set title for this SNR
            snr_label = 'SNR = ∞' if np.isinf(snr) else f'SNR = {snr}'
            ax.set_title(f'{snr_label}', fontsize=12)

            # Set ticks
            ax.set_xticks(range(len(Ms)))
            ax.set_xticklabels(Ms, rotation=45, fontsize=9)

            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels([f"$2^{{{int(np.log2(n))}}}$" if (n & (n - 1) == 0) and n != 0 else f"{n}" for n in ns],
                               fontsize=9)
    # Hide any unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # Save figure
    base_name = os.path.basename(results_dir)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for title
    plt.savefig(os.path.join(results_dir, f"{base_name}_cc_heatmaps.png"),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_cc_vs_realizations(results, results_dir):
    """
    Plot CC scores vs number of realizations for each n-value in a grid layout

    Args:
        results: Dictionary with metrics from calculate_metrics()
        results_dir: Directory to save plots to
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    snrs = results['SNRs']
    metrics_by_snr = results['metrics']
    ns = results['ns']
    Ms = results['Ms']

    valid_snrs = [snr for snr in snrs if snr in metrics_by_snr]
    if not valid_snrs:
        print("No valid SNR data to plot")
        return

    # If only one n value, create a single plot
    if len(ns) == 1:
        plt.figure(figsize=(10, 6))

        # Use a qualitative colormap
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(valid_snrs)))

        # Plot CC vs M for each SNR
        for i, snr in enumerate(valid_snrs):
            cc_data = metrics_by_snr[snr]['ccs']

            # Extract data for the selected n value
            cc_vs_M = cc_data[0, :]  # Only one n value

            # Create label
            snr_label = 'SNR = ∞' if np.isinf(snr) else f'SNR = {snr}'

            # Plot
            plt.plot(Ms, cc_vs_M, 'o-', color=colors[i], label=snr_label, linewidth=2, markersize=6)

        # Add labels and title
        plt.xlabel('Number of Realizations (M)', fontsize=12)
        plt.ylabel('Cross-Correlation Score', fontsize=12)
        plt.title(f'Cross-Correlation vs Number of Realizations (n = {ns[0]})', fontsize=14)

        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Save figure
        base_name = os.path.basename(results_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{base_name}_cc_vs_realizations.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()
        return

    # For multiple n values, choose a subset and plot in a grid

    # Select n values to plot - if many, select a representative subset
    if len(ns) <= 6:
        # Use all n values if 6 or fewer
        plot_n_indices = list(range(len(ns)))
    else:
        # Select first, last, and up to 4 equally spaced in between
        plot_n_indices = [0]
        step = len(ns) // 4
        for i in range(step, len(ns) - step, step):
            plot_n_indices.append(i)
        plot_n_indices.append(len(ns) - 1)

    # Determine grid dimensions
    num_plots = len(plot_n_indices)
    if num_plots <= 3:
        nrows, ncols = 1, num_plots
    elif num_plots <= 6:
        nrows, ncols = 2, (num_plots + 1) // 2
    else:
        nrows, ncols = 3, (num_plots + 2) // 3

    # Create figure with subplots in a grid
    fig_width = min(15, 5 * ncols)
    fig_height = min(12, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)

    # Make axes accessible as a flattened array
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Use a qualitative colormap
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(valid_snrs)))

    # Plot each n value in its own subplot
    for plot_idx, n_idx in enumerate(plot_n_indices):
        if plot_idx >= len(axes_flat):
            break

        ax = axes_flat[plot_idx]
        n_value = ns[n_idx]

        # Plot CC vs M for each SNR on this subplot
        for i, snr in enumerate(valid_snrs):
            cc_data = metrics_by_snr[snr]['ccs']

            # Extract data for this n value (ensure we don't go out of bounds)
            valid_n_idx = min(n_idx, cc_data.shape[0] - 1)
            cc_vs_M = cc_data[valid_n_idx, :]

            # Create label (only include in the first subplot)
            snr_label = 'SNR = ∞' if np.isinf(snr) else f'SNR = {snr}'
            label = snr_label if plot_idx == 0 else None

            # Plot
            ax.plot(Ms, cc_vs_M, 'o-', color=colors[i], label=label, linewidth=2, markersize=6)

        # Add title and grid
        ax.set_title(f'n = {n_value}', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add y-label only for leftmost subplots
        if plot_idx % ncols == 0:
            ax.set_ylabel('Cross-Correlation Score', fontsize=12)

        # Add x-label only for bottom subplots
        if plot_idx >= len(plot_n_indices) - ncols:
            ax.set_xlabel('Number of Realizations (M)', fontsize=12)

    # Hide any unused axes
    for j in range(plot_idx + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # Add a single legend for the entire figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=min(5, len(valid_snrs)), fontsize=10)

    # Add a super title
    fig.suptitle('Cross-Correlation vs Number of Realizations', fontsize=16, y=0.99)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the legend

    # Save figure
    base_name = os.path.basename(results_dir)
    plt.savefig(os.path.join(results_dir, f"{base_name}_cc_vs_realizations_grid.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
def main():
    """Main function to run the analysis"""
    # Ask user to select a results directory
    results_dir = select_result_directory()
    if not results_dir:
        print("No directory selected. Exiting.")
        return

    # Detect if this is coherent or incoherent simulation
    is_coherent = detect_simulation_type(results_dir)

    # Calculate metrics
    results = calculate_metrics(results_dir, is_coherent)

    if results:
        # Create heatmaps
        create_heatmaps(results, results_dir)

        # Plot CC vs realizations
        plot_cc_vs_realizations(results, results_dir)
    else:
        print("Error calculating metrics. Exiting.")


if __name__ == "__main__":
    main()