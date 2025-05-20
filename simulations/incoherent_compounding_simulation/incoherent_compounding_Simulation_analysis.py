import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import LinearSegmentedColormap


# Define custom colormap
def get_custom_colormap():
    colors = [
        (0, 0, 0),  # Black
        (0, 0.2, 0),  # Dark Green
        (0, 0.5, 0),  # Green
        (0, 0.8, 0),  # Bright Green
        (0.7, 1, 0),  # Light Green-Yellow
        (1, 1, 1)  # White
    ]
    positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
    return LinearSegmentedColormap.from_list('greenish_hot', list(zip(positions, colors)))


# Get the custom colormap
new_cmap = get_custom_colormap()


# Select the results directory using GUI
def select_results_dir():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    initial_dir = os.getcwd()
    results_dir = filedialog.askdirectory(
        title="Select Simulation Results Directory",
        initialdir=initial_dir
    )

    if not results_dir:
        print("No directory selected!")
        return None

    print(f"Selected directory: {results_dir}")
    return results_dir


# Main visualization function
def visualize_results(results_dir):
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    print(f"Visualizing results in: {results_dir}")

    # Load parameters
    K_values = np.load(os.path.join(results_dir, "K_values.npy"))

    # Create output directory for visualization figures
    viz_dir = os.path.join(results_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Plot ground truth
    gt = np.load(os.path.join(results_dir, "ground_truth.npy"))
    plt.figure(figsize=(8, 8))
    plt.title("Ground Truth")
    plt.imshow(gt, cmap=new_cmap)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "01_ground_truth.png"), dpi=300)
    plt.close()

    # 2. Plot widefield
    widefield = np.load(os.path.join(results_dir, "widefield.npy"))
    plt.figure(figsize=(8, 8))
    plt.title("Widefield Reference")
    plt.imshow(np.abs(widefield), cmap=new_cmap)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "02_widefield.png"), dpi=300)
    plt.close()

    # 3. Plot illumination diffuser
    diffuser = np.load(os.path.join(results_dir, "diffuser_example.npy"))
    plt.figure(figsize=(8, 8))
    plt.title("Example Illumination Diffuser (Phase)")
    plt.imshow(np.angle(diffuser), cmap='hsv')
    plt.colorbar(label='Phase (radians)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "03_illumination_diffuser.png"), dpi=300)
    plt.close()

    # 4. Plot 3 first illuminations - amplitude and phase in separate figures
    illuminations = np.load(os.path.join(results_dir, "illuminations.npy"))

    # 4a. Amplitude figure for illuminations
    n_show = min(3, len(illuminations))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("First 3 Illuminations", fontsize=16)

    for i in range(n_show):
        # --- amplitude on the top row ---
        ax_amp = axes[0, i]
        im_amp = ax_amp.imshow(np.abs(illuminations[i]),
                               cmap=new_cmap, vmin=0, vmax=1)
        ax_amp.set_title(f"Amplitude {i + 1}")
        ax_amp.axis('off')
        fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)

        # --- phase on the bottom row ---
        ax_phase = axes[1, i]
        im_phase = ax_phase.imshow(np.angle(illuminations[i]), cmap='hsv')
        ax_phase.set_title(f"Phase {i + 1}")
        ax_phase.axis('off')
        fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

    # hide any unused panels if len(illuminations) < 3
    for j in range(i + 1, 3):
        axes[0, j].axis('off')
        axes[1, j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(viz_dir, "04a_illuminations.png"), dpi=300)
    plt.close()

    # 5. Plot 3 first speckle patterns - amplitude and phase in separate figures
    speckle_patterns = np.load(os.path.join(results_dir, "speckle_patterns.npy"))

    n_show = min(3, len(speckle_patterns))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("First 3 Speckle Patterns at Object", fontsize=16)

    for i in range(n_show):
        # amplitude (top row)
        amp_ax = axes[0, i]
        amp_im = amp_ax.imshow(np.abs(speckle_patterns[i]), cmap=new_cmap)
        amp_ax.set_title(f"Speckle {i + 1}")
        amp_ax.axis('off')
        fig.colorbar(amp_im, ax=amp_ax, fraction=0.046, pad=0.04)

        # phase (bottom row)
        ph_ax = axes[1, i]
        ph_im = ph_ax.imshow(np.angle(speckle_patterns[i]), cmap='hsv')
        ph_ax.set_title(f"Phase {i + 1}")
        ph_ax.axis('off')
        fig.colorbar(ph_im, ax=ph_ax, fraction=0.046, pad=0.04)

    # turn off any unused panels
    for j in range(n_show, 3):
        axes[0, j].axis('off');
        axes[1, j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(viz_dir, "05a_speckle_patterns111.png"), dpi=300)
    plt.close()

    # 6. Plot macro intensities for each K value
    # Find all macro intensity files with the new naming pattern
    macro_intensity_files = glob.glob(os.path.join(results_dir, "macro_intensities_with_*_micro_frames.npy"))
    if macro_intensity_files:
        k_values_found = sorted([int(f.split('_with_')[1].split('_micro')[0]) for f in macro_intensity_files])

        # Limit to 9 for visibility
        n_plots = min(9, len(k_values_found))
        plt.figure(figsize=(15, 10))
        plt.suptitle("Macro Intensities for Different K Values", fontsize=16)

        for i, k in enumerate(k_values_found[:n_plots]):
            plt.subplot(3, 3, i + 1)
            macro_intensities = np.load(os.path.join(results_dir, f"macro_intensities_with_{k}_micro_frames.npy"))
            # Use the first element if it's a 3D array
            if len(macro_intensities.shape) > 2:
                macro_intensities = macro_intensities[0]

            plt.title(f"K = {k}")
            plt.imshow(macro_intensities, cmap=new_cmap)
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        plt.savefig(os.path.join(viz_dir, "06_macro_intensities.png"), dpi=300)
        plt.close()
    else:
        print("Macro intensity files not found with expected pattern")

    # 7. Plot macro objects for each K value
    macro_objects_files = glob.glob(os.path.join(results_dir, "macro_objects_intesities_with_*_micro_frames.npy"))
    if macro_objects_files:
        k_values_found = sorted([int(f.split('_with_')[1].split('_micro')[0]) for f in macro_objects_files])

        # Limit to 9 for visibility
        n_plots = min(9, len(k_values_found))
        plt.figure(figsize=(15, 10))
        plt.suptitle("Macro Objects for Different K Values", fontsize=16)

        for i, k in enumerate(k_values_found[:n_plots]):
            plt.subplot(3, 3, i + 1)
            macro_objects = np.load(os.path.join(results_dir, f"macro_objects_intesities_with_{k}_micro_frames.npy"))
            # Use the first element if it's a 3D array
            if len(macro_objects.shape) > 2:
                macro_objects = macro_objects[0]

            plt.title(f"K = {k}")
            plt.imshow(np.abs(macro_objects), cmap=new_cmap)
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(viz_dir, "07_macro_objects.png"), dpi=300)
        plt.close()
    else:
        print("Macro objects files not found with expected pattern")

    # 8. Plot macro frames for each K value
    macro_frames_files = glob.glob(os.path.join(results_dir, "macro_frames_with_*_micro_frames.npy"))
    if macro_frames_files:
        k_values_found = sorted([int(f.split('_with_')[1].split('_micro')[0]) for f in macro_frames_files])

        # Limit to 9 for visibility
        n_plots = min(9, len(k_values_found))
        plt.figure(figsize=(15, 10))
        plt.suptitle("Macro Frames for Different K Values", fontsize=16)

        for i, k in enumerate(k_values_found[:n_plots]):
            plt.subplot(3, 3, i + 1)
            macro_frames = np.load(os.path.join(results_dir, f"macro_frames_with_{k}_micro_frames.npy"))
            # Use the first element if it's a 3D array
            if len(macro_frames.shape) > 2:
                macro_frames = macro_frames[0]

            plt.title(f"K = {k}")
            plt.imshow(np.abs(macro_frames), cmap=new_cmap)
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(viz_dir, "08_macro_frames.png"), dpi=300)
        plt.close()
    else:
        print("Macro frames files not found with expected pattern")

    # 9. Plot all reconstructions - uniform and one per K value
    # Find all reconstruction files
    recon_files = [f for f in sorted(glob.glob(os.path.join(results_dir, "O_est_with_*_micro_frames.npy"))) if
                   "modulation" not in f]
    k_values_recon = sorted([int(f.split('_with_')[1].split('_micro')[0]) for f in recon_files])

    # Check if uniform reconstruction exists
    uniform_recon_path = os.path.join(results_dir, "uniform_reconstruction.npy")
    has_uniform = os.path.exists(uniform_recon_path)

    # Calculate grid size - we need one extra plot for uniform reconstruction
    n_k_plots = min(8, len(recon_files)) if has_uniform else min(9, len(recon_files))
    total_plots = n_k_plots + (1 if has_uniform else 0)
    rows = 3
    cols = (total_plots + rows - 1) // rows  # Ceiling division

    # Create figure
    plt.figure(figsize=(15, 10))
    plt.suptitle("All Reconstructions", fontsize=16)

    plot_idx = 0
    # First plot the uniform reconstruction if it exists
    if has_uniform:
        uniform_recon = np.load(uniform_recon_path)
        plt.subplot(rows, cols, plot_idx + 1)
        plt.title("Uniform (No Averaging)")
        plt.imshow(np.abs(uniform_recon), cmap=new_cmap)
        plt.colorbar()
        plt.axis('off')
        plot_idx += 1

    # Then plot the K-averaged reconstructions
    for i, k in enumerate(k_values_recon[:n_k_plots]):
        plt.subplot(rows, cols, plot_idx + 1)
        recon = np.load(os.path.join(results_dir, f"O_est_with_{k}_micro_frames.npy"))
        plt.title(f"K = {k}")
        plt.imshow(np.abs(recon), cmap=new_cmap)
        plt.colorbar()
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(os.path.join(viz_dir, "09_reconstructions.png"), dpi=300)
    plt.close()

    # 10. Plot all reconstructions - with modulatio
    # Find all reconstruction files

    recon_files = [f for f in sorted(glob.glob(os.path.join(results_dir, "O_est_with_*_micro_frames.npy"))) if
                   "modulation" in f]
    k_values_recon = sorted([int(f.split('modulation_with_')[1].split('_micro')[0]) for f in recon_files])

    # Check if uniform reconstruction exists
    uniform_recon_path = os.path.join(results_dir, "uniform_reconstruction.npy")
    has_uniform = os.path.exists(uniform_recon_path)

    # Calculate grid size - we need one extra plot for uniform reconstruction
    n_k_plots = min(8, len(recon_files)) if has_uniform else min(9, len(recon_files))
    total_plots = n_k_plots + (1 if has_uniform else 0)
    rows = 3
    cols = (total_plots + rows - 1) // rows  # Ceiling division

    # Create figure
    plt.figure(figsize=(15, 10))
    plt.suptitle("All Reconstructions with modulation", fontsize=16)

    plot_idx = 0
    # First plot the uniform reconstruction if it exists
    if has_uniform:
        uniform_recon = np.load(uniform_recon_path)
        plt.subplot(rows, cols, plot_idx + 1)
        plt.title("Uniform (No Averaging)")
        plt.imshow(np.abs(uniform_recon), cmap=new_cmap)
        plt.colorbar()
        plt.axis('off')
        plot_idx += 1

    # Then plot the K-averaged reconstructions
    for i, k in enumerate(k_values_recon[:n_k_plots]):
        plt.subplot(rows, cols, plot_idx + 1)
        recon = np.load(os.path.join(results_dir, f"O_est_with_modulation_with_{k}_micro_frames.npy"))
        plt.title(f"K = {k}")
        plt.imshow(np.abs(recon), cmap=new_cmap)
        plt.colorbar()
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(os.path.join(viz_dir, "09_reconstructions with modulation.png"), dpi=300)
    plt.close()

    print(f"Visualization complete! Results saved to {viz_dir}")
    return viz_dir


# Main execution
if __name__ == "__main__":
    results_dir = select_results_dir()
    if results_dir:
        visualize_results(results_dir)