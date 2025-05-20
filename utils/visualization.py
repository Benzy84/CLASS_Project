import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Button, Label


import numpy as np
import torch

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

def amp_and_hue(complex_array):
    # Compute amplitude and phase
    amplitude = np.abs(complex_array)
    phase = np.angle(complex_array)

    # Normalize amplitude and phase for visualization
    normalized_amplitude = amplitude / np.max(amplitude)
    normalized_phase = (phase + np.pi) / (2 * np.pi)

    # Create RGB values from normalized amplitude and phase
    combined_color = np.zeros((complex_array.shape[0], complex_array.shape[1], 3))
    combined_color[..., 0] = normalized_phase  # Hue
    combined_color[..., 1] = 1.0  # Full saturation
    combined_color[..., 2] = np.log10(normalized_amplitude)  # Brightness

    # Convert HSV to RGB
    return mcolors.hsv_to_rgb(combined_color)

def phplot(field, Amp=1, scale=0):
    """
    Plot a complex field using phase-as-hue and amplitude-as-brightness.

    Args:
        field: Complex field to visualize (numpy array)
        Amp: If 0, use uniform amplitude, if 1 (default) use actual amplitude
        scale: If > 0, add a phase color scale reference

    Returns:
        RGB image array
    """
    # Make sure input is numpy array
    field = np.asarray(field)

    # Extract phase and amplitude
    phase = np.angle(field)
    amplitude = np.abs(field)

    # Apply square root to amplitude for better visual contrast
    amplitude = np.sqrt(amplitude)

    # Avoid division by zero by adding a small epsilon
    max_amp = amplitude.max()
    if max_amp > 0:
        amplitude = amplitude / max_amp
    else:
        amplitude = np.ones_like(amplitude)

    # If Amp is 0, use uniform amplitude
    if Amp == 0:
        amplitude = np.ones_like(amplitude)

    # Create RGB image
    A = np.zeros((field.shape[0], field.shape[1], 3))

    # Map phase to RGB using sinusoidal transformations
    # This creates a smooth transition of colors around the phase circle
    A[:, :, 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
    A[:, :, 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amplitude  # Green
    A[:, :, 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

    # Normalize to prevent clipping
    max_val = A.max()
    if max_val > 0:
        A = A / max_val

    return A


def display_field(field=None, method=None):
    """
    Display a 2D field with appropriate visualization, supporting PyTorch tensors.
    If no field is provided, opens a file dialog to select a file.

    If the field is complex:
    - option 1: phplot (phase as color, amplitude as brightness)
    - option 2: separate phase and magnitude plots
    - option 3: amplitude plot only

    If the field is real:
    - Simple imshow

    Args:
        field: 2D array to display (PyTorch tensor or NumPy array)
        method: Visualization method (if None, user will be prompted with buttons):
                1 for phplot, 2 for separate phase/magnitude plots, 3 for amplitude only
    """

    # Function to create a button dialog for method selection
    def create_button_dialog():
        result = {"value": None}

        # Create dialog window
        dialog = Toplevel()
        dialog.title("Select Visualization Method")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Add a label with instructions
        Label(dialog, text="Choose visualization method for complex field:",
              font=("Arial", 12)).pack(pady=10)

        # Function to set result and close dialog
        def set_choice(choice):
            result["value"] = choice
            dialog.destroy()

        # Create buttons
        Button(dialog, text="Phase-Amplitude Color Plot",
               command=lambda: set_choice(1),
               width=30, height=2).pack(pady=5)

        Button(dialog, text="Separate Phase and Magnitude Plots",
               command=lambda: set_choice(2),
               width=30, height=2).pack(pady=5)

        Button(dialog, text="Magnitude Plot Only",
               command=lambda: set_choice(3),
               width=30, height=2).pack(pady=5)

        # Center the dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        # Make dialog modal
        dialog.transient()
        dialog.grab_set()
        dialog.wait_window()

        return result["value"]

    # Function to load a .npy file through a file dialog
    def load_npy_file():
        root = tk.Tk()
        root.withdraw()

        try:
            file_path = filedialog.askopenfilename(
                title="Select NPY File",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )

            if not file_path:  # User cancelled
                return None

            # Load the file
            data = np.load(file_path, allow_pickle=False)
            print(f"Loaded file: {file_path}")
            print(f"Shape: {data.shape}, dtype: {data.dtype}")

            # Try to ensure complex data is properly loaded
            if data.dtype.kind == 'f' and data.shape[-1] == 2:
                # This might be a complex array that got split into real/imag parts
                try:
                    data = data.view(np.complex128)
                    print("Converted to complex data type")
                except:
                    pass

            return data

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    # Helper function for phase-hue amplitude-brightness visualization
    def phplot(field, Amp=1):
        """Phase as hue, amplitude as brightness visualization."""
        phase = np.angle(field)
        amplitude = np.abs(field)

        # Apply square root to amplitude for better visual contrast
        amplitude = np.sqrt(amplitude)

        # Avoid division by zero by adding a small epsilon
        max_amp = amplitude.max()
        if max_amp > 0:
            amplitude = amplitude / max_amp
        else:
            amplitude = np.ones_like(amplitude)

        # If Amp is 0, use uniform amplitude
        if Amp == 0:
            amplitude = np.ones_like(amplitude)

        # Create RGB image
        A = np.zeros((field.shape[0], field.shape[1], 3))

        # Map phase to RGB using sinusoidal transformations
        # This creates a smooth transition of colors around the phase circle
        A[:, :, 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
        A[:, :, 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amplitude  # Green
        A[:, :, 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

        # Normalize to prevent clipping
        max_val = A.max()
        if max_val > 0:
            A = A / max_val

        return A

    # Load field if not provided
    if field is None:
        field = load_npy_file()
        if field is None:
            print("No field provided")
            return

    # Convert PyTorch tensor to NumPy if necessary
    try:
        if isinstance(field, torch.Tensor):
            # Move to CPU and convert to NumPy
            field = field.detach().cpu().numpy()
    except ImportError:
        # If torch is not available, assume field is already numpy
        pass

    # Ensure field is 2D
    original_shape = field.shape
    if field.ndim != 2:
        print(f"Expected 2D field, but got shape {field.shape}")
        try:
            # Create a hidden root window for the message box
            root = tk.Tk()
            root.withdraw()
            choice = messagebox.askyesno("Non-2D Field",
                                         f"Field has {field.ndim} dimensions. Display first 2D slice?")
            if choice:
                # Extract first 2D slice for higher dimensional arrays
                if field.ndim > 2:
                    field = field[(0,) * (field.ndim - 2)]
            else:
                return
        except:
            # If tkinter fails, just use the first slice
            print("Using first 2D slice of multi-dimensional array")
            if field.ndim > 2:
                field = field[(0,) * (field.ndim - 2)]

    # Check if field is complex
    is_complex = np.iscomplexobj(field)

    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    plt.suptitle(f"Field shape: {original_shape}, dtype: {field.dtype}", fontsize=10)

    if is_complex:
        # For complex fields, ask for visualization method if not specified
        if method is None:
            try:
                # Create root for tkinter dialogs
                root = tk.Tk()
                root.withdraw()

                # Show button dialog for method selection
                method = create_button_dialog()
            except Exception as e:
                print(f"Error creating dialog: {e}")
                # Default to method 1 if dialog fails
                method = 1

            if method is None:  # User cancelled
                plt.close(fig)
                return

        if method == 1:
            # Option 1: phplot (phase as color)
            ax = fig.add_subplot(111)
            ax.set_title('Complex Field (Phase as Color, Amplitude as Brightness)')

            # Call the phplot function
            rgb_image = phplot(field)
            im = ax.imshow(rgb_image)

            # Adjust layout to make room for the color reference
            fig.subplots_adjust(right=0.85)

            # Create a better phase colorbar
            # Generate HSV color wheel for the phase reference
            phase_values = np.linspace(-np.pi, np.pi, 256)
            hsv_colors = np.zeros((256, 3))
            hsv_colors[:, 0] = (phase_values + np.pi) / (2 * np.pi)  # Hue from 0 to 1
            hsv_colors[:, 1] = 1.0  # Full saturation
            hsv_colors[:, 2] = 1.0  # Full value (brightness)
            rgb_colors = hsv_to_rgb(hsv_colors.reshape(256, 1, 3)).reshape(256, 3)

            # Create a custom colormap from these RGB values
            phase_cmap = mpl.colors.ListedColormap(rgb_colors)

            # Add a better colorbar for phase reference
            cax = fig.add_axes([0.88, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
            norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=phase_cmap, norm=norm, orientation='vertical')
            cb.set_label('Phase (radians)')
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

            # Add text for field stats
            amp_max = np.abs(field).max()
            amp_min = np.abs(field).min()
            ax.text(0.02, 0.98, f"Max amplitude: {amp_max:.4g}\nMin amplitude: {amp_min:.4g}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        elif method == 2:
            # Option 2: Separate phase and magnitude plots
            ax1 = fig.add_subplot(121)
            ax1.set_title('Magnitude')
            magnitude = np.abs(field)
            magnitude_plot = ax1.imshow(magnitude, cmap='viridis')
            plt.colorbar(magnitude_plot, ax=ax1)

            ax2 = fig.add_subplot(122)
            ax2.set_title('Phase')

            # Create a better phase colormap
            phase_values = np.linspace(-np.pi, np.pi, 256)
            hsv_colors = np.zeros((256, 3))
            hsv_colors[:, 0] = (phase_values + np.pi) / (2 * np.pi)  # Hue from 0 to 1
            hsv_colors[:, 1] = 1.0  # Full saturation
            hsv_colors[:, 2] = 1.0  # Full value (brightness)
            rgb_colors = hsv_to_rgb(hsv_colors.reshape(256, 1, 3)).reshape(256, 3)
            phase_cmap = mpl.colors.ListedColormap(rgb_colors)

            phase_plot = ax2.imshow(np.angle(field), cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
            cbar = plt.colorbar(phase_plot, ax=ax2)
            cbar.set_label('Phase (radians)')
            cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

            # Use tight_layout safely here
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

        elif method == 3:
            # Option 3: Just magnitude
            ax = fig.add_subplot(111)
            ax.set_title('Magnitude')
            magnitude = np.abs(field)
            magnitude_plot = ax.imshow(magnitude, cmap='viridis')
            plt.colorbar(magnitude_plot, ax=ax)

            # Add text for field stats
            amp_max = magnitude.max()
            amp_min = magnitude.min()
            amp_mean = magnitude.mean()
            ax.text(0.02, 0.98, f"Max: {amp_max:.4g}\nMin: {amp_min:.4g}\nMean: {amp_mean:.4g}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Use tight_layout safely here
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    else:
        # For real fields, just use imshow
        ax = fig.add_subplot(111)
        ax.set_title('Real Field')
        im = ax.imshow(field, cmap='viridis')
        plt.colorbar(im, ax=ax)

        # Add text for field stats
        val_max = field.max()
        val_min = field.min()
        val_mean = field.mean()
        ax.text(0.02, 0.98, f"Max: {val_max:.4g}\nMin: {val_min:.4g}\nMean: {val_mean:.4g}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Use tight_layout safely here
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    plt.show()
