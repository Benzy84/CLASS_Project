import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Button, Label
from PIL import Image
import mat73
import glob
import datetime
import os



def load_file_to_tensor():
    """
    Opens a file dialog for the user to select an image or numpy file,
    and converts it to a 2D PyTorch tensor.
    If GPU is available, the tensor is moved to GPU.

    Returns:
        torch.Tensor: 2D tensor of the image (on GPU if available)
    """
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()

    # File dialog to select the image
    file_path = filedialog.askopenfilename(
        title="Select image file",
        filetypes=[
            ("All files", "*.*"),
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("NumPy files", "*.npy")
        ]
    )

    if not file_path:
        print("No file selected.")
        return None

    # Process based on file type
    if file_path.endswith('.npy'):
        # Load numpy file
        arr = np.load(file_path)
    else:
        # Load image file
        image = Image.open(file_path)
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        arr = np.array(image, dtype=np.float32)

    # Ensure it's 2D
    if arr.ndim > 2:
        arr = np.mean(arr, axis=2)

    # Normalize to 0-1 range
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Convert to torch tensor
    tensor = torch.from_numpy(arr).float()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)

    print(
        f"Loaded image: shape={tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}], device={tensor.device}")
    return tensor


def load_mat_file(file_path=None, purpose=None):
    # If file_path is not provided, open a file dialog
    if file_path is None:
        root = tk.Tk()
        root.withdraw()

        # Create title with purpose if provided
        title = "Select MAT File"
        if purpose:
            title = f"Select MAT File for {purpose}"

        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )

        if not file_path:  # User cancelled selection
            print("No file selected.")
            return None, None

    # Load the .mat file
    data = mat73.loadmat(file_path)

    # Process data as in the original function
    if len(data['I_cam'].shape) == 3:
        A = torch.permute(torch.tensor(data['I_cam'], dtype=torch.float32), [2, 0, 1])
    else:
        A = torch.tensor(data['I_cam'], dtype=torch.float32).T

    return data, A


def create_timestamped_dir(base_name, snr=None):
    """
    Create a uniquely named directory for saving results with a timestamp.
    Uses ISO 8601 format for timestamps to ensure chronological sorting.
    """

    # Create timestamp string in ISO 8601 format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory name with timestamp
    if snr is not None:
        if snr == float('inf') or snr == torch.inf:
            dir_name = f"{base_name}_{timestamp}_SNR_inf"
        else:
            dir_name = f"{base_name}_{timestamp}_SNR_{snr}"
    else:
        dir_name = f"{base_name}_{timestamp}"

    # Create the directory if it doesn't exist
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    return dir_name


def find_latest_result_dirs(base_pattern, n=10):
    """
    Find the most recent result directories matching a pattern.

    Parameters:
    -----------
    base_pattern : str
        Pattern to match at the beginning of directory names
    n : int
        Number of most recent directories to return

    Returns:
    --------
    list
        List of most recent matching directory paths
    """

    # Get absolute path of current directory
    current_dir = os.getcwd()

    # Get all directories matching the pattern (both relative and with full path)
    matching_dirs = []
    for pattern in [f"{base_pattern}*", f"*/{base_pattern}*"]:
        matching_dirs.extend([d for d in glob.glob(pattern) if os.path.isdir(d)])

    # Ensure unique directories (in case of duplicates from different patterns)
    matching_dirs = list(set(matching_dirs))

    # Sort by modification time (most recent first)
    matching_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)

    # Print debug info
    print(f"Found {len(matching_dirs)} directories matching '{base_pattern}*'")
    for d in matching_dirs[:min(5, len(matching_dirs))]:
        print(f"  {d} (modified: {os.path.getmtime(d)})")

    # Return the n most recent
    return matching_dirs[:n]


def mkdir(pth):
    if not os.path.isdir(pth):
        os.mkdir(pth)
