import os
import torch
from utils import *
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2 as fft, ifft2 as ifft
from tqdm import tqdm
import matplotlib
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
import tkinter as tk
import colorsys
from tkinter import filedialog, messagebox
from PIL import Image
import time
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import wiener
from skimage import restoration
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from diffractsim import MonochromaticField, mm, cm, um, set_backend



matplotlib.use('TkAgg')
plt.ion()

from torchvision.transforms import CenterCrop

import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import imageio

import torch
import numpy as np




def create_comparison_propagation_video(fields, field_names, z_initial, z_final, num_of_frames, fps, dx_values,
                                        dy_values,
                                        wavelength, output_filename='comparison_video.mp4', cmap='hot',
                                        padding_size=500, show_progress=True, vmin=None, vmax=None,
                                        layout='horizontal'):
    """
    Creates a side-by-side comparison video of multiple fields propagating from z_initial to z_final.

    Parameters:
    -----------
    fields : list of torch.Tensor or numpy.ndarray
        List of complex fields to propagate.
    field_names : list of str
        Names of each field for labeling.
    z_initial : float
        Initial propagation distance in meters.
    z_final : float
        Final propagation distance in meters.
    num_of_frames : int
        Number of frames to generate between z_initial and z_final.
    fps : int
        Frames per second for the output video.
    dx_values : list of float
        Pixel sizes in x direction (m) for each field.
    dy_values : list of float
        Pixel sizes in y direction (m) for each field.
    wavelength : float
        Wavelength of light (m).
    output_filename : str, optional
        Name of the output video file. Default is 'comparison_video.mp4'.
    cmap : str, optional
        Colormap to use for the video. Default is 'hot'.
    padding_size : int, optional
        Size of padding to add around the field for propagation. Default is 500.
    show_progress : bool, optional
        Whether to show a progress bar. Default is True.
    vmin : float, optional
        Minimum value for intensity normalization. Default is None.
    vmax : float, optional
        Maximum value for intensity normalization. Default is None.
    layout : str, optional
        Layout for the comparison video. Either 'horizontal' or 'vertical'. Default is 'horizontal'.

    Returns:
    --------
    None
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import cv2
    from tqdm import tqdm

    num_fields = len(fields)
    if num_fields != len(field_names) or num_fields != len(dx_values) or num_fields != len(dy_values):
        raise ValueError("The number of fields, names, dx values and dy values must be the same")

    # Convert fields to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded_fields = []
    original_shapes = []
    start_rows = []
    start_cols = []
    end_rows = []
    end_cols = []

    for i, field in enumerate(fields):
        # Check if field is a numpy array and convert to torch tensor
        if isinstance(field, np.ndarray):
            field_tensor = torch.from_numpy(field).to(device).to(torch.complex64)
        else:
            field_tensor = field.to(device).to(torch.complex64)

        # Pad the field
        padded_field = F.pad(field_tensor, (padding_size, padding_size, padding_size, padding_size),
                             mode='constant', value=0)
        padded_fields.append(padded_field)

        # Store original dimensions for later cropping
        original_height, original_width = field_tensor.shape
        original_shapes.append((original_height, original_width))

        # Calculate cropping indices
        start_row = padding_size
        end_row = start_row + original_height
        start_col = padding_size
        end_col = start_col + original_width

        start_rows.append(start_row)
        end_rows.append(end_row)
        start_cols.append(start_col)
        end_cols.append(end_col)

    # Generate z values
    z_values = torch.linspace(z_initial, z_final, num_of_frames)

    # Create a dummy figure to get the colormap exactly as matplotlib would render it
    plt.figure(figsize=(8, 8))
    dummy_img = plt.imshow(np.zeros((100, 100)), cmap=cmap)
    colormap = dummy_img.get_cmap()
    plt.close()

    # Find global min/max for normalization if not provided
    if vmin is None or vmax is None:
        # Propagate fields at a few sample z positions to find global min/max
        sample_indices = [0, num_of_frames // 2, num_of_frames - 1]
        sample_intensities = []

        for field_idx in range(num_fields):
            for z_idx in sample_indices:
                z = z_values[z_idx]
                # Propagate field
                field_propagated = fresnel_propagation(
                    padded_fields[field_idx],
                    dx_values[field_idx],
                    dy_values[field_idx],
                    z,
                    wavelength
                )
                # Crop to original size
                cropped_field = field_propagated[
                                start_rows[field_idx]:end_rows[field_idx],
                                start_cols[field_idx]:end_cols[field_idx]
                                ]
                # Get intensity
                intensity = torch.abs(cropped_field) ** 2
                sample_intensities.append(intensity)

        # Compute global min and max if not provided
        if vmin is None:
            vmin = min(torch.min(intensity).item() for intensity in sample_intensities)
        if vmax is None:
            vmax = max(torch.max(intensity).item() for intensity in sample_intensities)

    # For 'hot' colormap, adjust the scaling to match matplotlib's default behavior
    # which tends to show brighter output than direct normalization
    if cmap == 'hot':
        # Apply a gamma correction to make it brighter (more like matplotlib)
        gamma = 0.5  # Lower value makes the image brighter
        gamma_corrected_vmin = vmin ** gamma
        gamma_corrected_vmax = vmax ** gamma
    else:
        gamma_corrected_vmin = vmin
        gamma_corrected_vmax = vmax

    # Create a normalizer
    norm = Normalize(vmin=gamma_corrected_vmin, vmax=gamma_corrected_vmax)

    # Determine output video dimensions based on layout
    if layout == 'horizontal':
        # Sum of widths by max height
        output_width = sum(original_shapes[i][1] for i in range(num_fields))
        output_height = max(original_shapes[i][0] for i in range(num_fields))
    else:  # vertical layout
        # Max width by sum of heights
        output_width = max(original_shapes[i][1] for i in range(num_fields))
        output_height = sum(original_shapes[i][0] for i in range(num_fields))

    # Create video writer
    if output_filename.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_filename.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        # Default to mp4 if extension is not recognized
        output_filename += '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))

    # Loop through z values and propagate all fields for each one
    iterator = tqdm(z_values) if show_progress else z_values
    for z in iterator:
        if show_progress:
            iterator.set_description(f'z = {z.item():.4f} m')

        # Create a blank canvas for the combined frame
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # Process each field
        for field_idx in range(num_fields):
            # Propagate the current field
            field_propagated = fresnel_propagation(
                padded_fields[field_idx],
                dx_values[field_idx],
                dy_values[field_idx],
                z,
                wavelength
            )

            # Crop to original size
            h, w = original_shapes[field_idx]
            cropped_field = field_propagated[
                            start_rows[field_idx]:end_rows[field_idx],
                            start_cols[field_idx]:end_cols[field_idx]
                            ]

            # Calculate the intensity
            intensity = torch.abs(cropped_field) ** 2

            # Convert to numpy for visualization
            intensity_np = intensity.cpu().numpy()

            # Apply gamma correction for better brightness (similar to matplotlib)
            if cmap == 'hot':
                gamma = 0.5  # Adjust this value to control brightness
                intensity_np = intensity_np ** gamma

            # Normalize the intensity
            normalized_intensity = norm(intensity_np)

            # Apply colormap to get RGB
            colored_frame = (colormap(normalized_intensity) * 255).astype(np.uint8)

            # For 'hot' colormap, boost the brightness to match matplotlib's appearance
            if cmap == 'hot':
                # Convert to HSV to adjust brightness
                hsv_frame = cv2.cvtColor(colored_frame[:, :, :3], cv2.COLOR_RGB2HSV)
                # Increase the brightness (V channel)
                hsv_frame[:, :, 2] = np.clip(hsv_frame[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
                # Convert back to RGB
                colored_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)

            # Convert RGBA to BGR (for OpenCV)
            bgr_frame = cv2.cvtColor(colored_frame, cv2.COLOR_RGBA2BGR)

            # Add field name as text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            font_color = (255, 255, 255)  # White text
            cv2.putText(bgr_frame, field_names[field_idx], (20, 30), font, font_scale, font_color, font_thickness)

            # Add z-distance information
            z_text = f"z = {z.item() * 1000:.2f} mm"
            cv2.putText(bgr_frame, z_text, (20, h - 20), font, font_scale, font_color, font_thickness)

            # Add scale bar (1 mm)
            scale_length_px = int(1e-3 / dx_values[field_idx])  # 1 mm in pixels
            if scale_length_px > w // 4:
                # Cap scale bar to 1/4 of the image width if it's too large
                scale_length_px = w // 4
                scale_mm = scale_length_px * dx_values[field_idx] * 1000  # Convert to mm
                scale_text = f"{scale_mm:.1f} mm"
            else:
                scale_text = "1 mm"

            scale_start_x = 20
            scale_y = h - 50
            scale_end_x = scale_start_x + scale_length_px
            cv2.line(bgr_frame, (scale_start_x, scale_y), (scale_end_x, scale_y), font_color, 2)

            # Add scale bar text
            cv2.putText(bgr_frame, scale_text, (scale_start_x, scale_y - 10), font, font_scale, font_color,
                        font_thickness)

            # Place the frame in the combined image based on layout
            if layout == 'horizontal':
                # Calculate the x offset for this field
                x_offset = sum(original_shapes[j][1] for j in range(field_idx))
                combined_frame[0:h, x_offset:x_offset + w] = bgr_frame
            else:  # vertical layout
                # Calculate the y offset for this field
                y_offset = sum(original_shapes[j][0] for j in range(field_idx))
                combined_frame[y_offset:y_offset + h, 0:w] = bgr_frame

        # Write the combined frame to video
        video_writer.write(combined_frame)

    # Release video writer
    video_writer.release()

    print(f"Comparison video saved as {output_filename}")
    return None


def create_propagation_video(field, z_initial, z_final, num_of_frames, fps, dx, dy, wavelength,
                             output_filename='propagation_video.mp4', cmap='hot', padding_size=500,
                             show_progress=True, vmin=None, vmax=None):
    """
    Creates a video of a field propagating from z_initial to z_final.

    Parameters:
    -----------
    field : torch.Tensor or numpy.ndarray
        The complex field to propagate.
    z_initial : float
        Initial propagation distance in meters.
    z_final : float
        Final propagation distance in meters.
    num_of_frames : int
        Number of frames to generate between z_initial and z_final.
    fps : int
        Frames per second for the output video.
    dx : float
        Pixel size in x direction (m).
    dy : float
        Pixel size in y direction (m).
    wavelength : float
        Wavelength of light (m).
    output_filename : str, optional
        Name of the output video file. Default is 'propagation_video.mp4'.
    cmap : str, optional
        Colormap to use for the video. Default is 'hot'.
    padding_size : int, optional
        Size of padding to add around the field for propagation. Default is 500.
    show_progress : bool, optional
        Whether to show a progress bar. Default is True.
    vmin : float, optional
        Minimum value for intensity normalization. Default is None.
    vmax : float, optional
        Maximum value for intensity normalization. Default is None.

    Returns:
    --------
    None
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import cv2
    from tqdm import tqdm

    # Check if field is a numpy array and convert to torch tensor
    if isinstance(field, np.ndarray):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        field = torch.from_numpy(field).to(device).to(torch.complex64)

    # Pad the field
    padded_field = F.pad(field, (padding_size, padding_size, padding_size, padding_size),
                         mode='constant', value=0)

    # Calculate original shape for cropping later
    original_height, original_width = field.shape
    start_row = padding_size
    end_row = start_row + original_height
    start_col = padding_size
    end_col = start_col + original_width

    # Generate z values
    z_values = torch.linspace(z_initial, z_final, num_of_frames)

    # Create video writer
    if output_filename.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_filename.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        # Default to mp4 if extension is not recognized
        output_filename += '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a dummy figure to get the colormap exactly as matplotlib would render it
    plt.figure(figsize=(8, 8))
    dummy_img = plt.imshow(np.zeros((100, 100)), cmap=cmap)
    colormap = dummy_img.get_cmap()
    plt.close()

    # Find propagated field values for min/max normalization if not provided
    if vmin is None or vmax is None:
        # Propagate field at a few sample z positions to find global min/max
        sample_indices = [0, num_of_frames // 2, num_of_frames - 1]
        sample_intensities = []

        for idx in sample_indices:
            z = z_values[idx]
            # Propagate field
            field_propagated = fresnel_propagation(padded_field, dx, dy, z, wavelength)
            # Crop to original size
            cropped_field = field_propagated[start_row:end_row, start_col:end_col]
            # Get intensity
            intensity = torch.abs(cropped_field) ** 2
            sample_intensities.append(intensity)

        # Compute global min and max if not provided
        if vmin is None:
            vmin = min(torch.min(intensity).item() for intensity in sample_intensities)
        if vmax is None:
            vmax = max(torch.max(intensity).item() for intensity in sample_intensities)

    # For 'hot' colormap, adjust the scaling to match matplotlib's default behavior
    # which tends to show brighter output than direct normalization
    if cmap == 'hot':
        # Apply a gamma correction to make it brighter (more like matplotlib)
        gamma = 0.5  # Lower value makes the image brighter
        gamma_corrected_vmin = vmin ** gamma
        gamma_corrected_vmax = vmax ** gamma
    else:
        gamma_corrected_vmin = vmin
        gamma_corrected_vmax = vmax

    # Create a normalizer
    norm = Normalize(vmin=gamma_corrected_vmin, vmax=gamma_corrected_vmax)

    # Initialize video writer with the first frame's dimensions
    first_frame = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (original_width, original_height))

    # Loop through z values and propagate the field for each one
    iterator = tqdm(z_values) if show_progress else z_values
    for z in iterator:
        if show_progress:
            iterator.set_description(f'z = {z.item():.4f} m')

        # Propagate field
        field_propagated = fresnel_propagation(padded_field, dx, dy, z, wavelength)

        # Crop to original size
        cropped_field = field_propagated[start_row:end_row, start_col:end_col]

        # Calculate the intensity
        intensity = torch.abs(cropped_field) ** 2

        # Convert to numpy for visualization
        intensity_np = intensity.cpu().numpy()

        # Apply gamma correction for better brightness (similar to matplotlib)
        if cmap == 'hot':
            gamma = 0.5  # Adjust this value to control brightness
            intensity_np = intensity_np ** gamma

        # Normalize the intensity
        normalized_intensity = norm(intensity_np)

        # Apply colormap to get RGB
        colored_frame = (colormap(normalized_intensity) * 255).astype(np.uint8)

        # For 'hot' colormap, boost the brightness to match matplotlib's appearance
        if cmap == 'hot':
            # Convert to HSV to adjust brightness
            hsv_frame = cv2.cvtColor(colored_frame[:, :, :3], cv2.COLOR_RGB2HSV)
            # Increase the brightness (V channel)
            hsv_frame[:, :, 2] = np.clip(hsv_frame[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
            # Convert back to RGB
            colored_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)

        # Convert RGBA to BGR (for OpenCV)
        bgr_frame = cv2.cvtColor(colored_frame, cv2.COLOR_RGBA2BGR)

        # Add scale bar and z-distance information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 255, 255)  # White text

        # Add z-distance information
        z_text = f"z = {z.item() * 1000:.2f} mm"
        text_size = cv2.getTextSize(z_text, font, font_scale, font_thickness)[0]
        text_x = 20
        text_y = original_height - 20
        cv2.putText(bgr_frame, z_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Add scale bar (1 mm)
        scale_length_px = int(1e-3 / dx)  # 1 mm in pixels
        if scale_length_px > original_width // 4:
            # Cap scale bar to 1/4 of the image width if it's too large
            scale_length_px = original_width // 4
            scale_mm = scale_length_px * dx * 1000  # Convert to mm
            scale_text = f"{scale_mm:.1f} mm"
        else:
            scale_text = "1 mm"

        scale_start_x = 20
        scale_y = original_height - 50
        scale_end_x = scale_start_x + scale_length_px
        cv2.line(bgr_frame, (scale_start_x, scale_y), (scale_end_x, scale_y), font_color, 2)

        # Add scale bar text
        scale_text_size = cv2.getTextSize(scale_text, font, font_scale, font_thickness)[0]
        scale_text_x = scale_start_x + (scale_length_px - scale_text_size[0]) // 2
        scale_text_y = scale_y - 10
        cv2.putText(bgr_frame, scale_text, (scale_text_x, scale_text_y), font, font_scale, font_color, font_thickness)

        # Write frame to video
        video_writer.write(bgr_frame)

    # Release video writer
    video_writer.release()

    print(f"Video saved as {output_filename}")
    return None


def center_crop(image, crop_size):
    """
    Center crop an image, working with both NumPy arrays and PyTorch tensors

    Parameters:
    image: NumPy ndarray or PyTorch tensor
    crop_size (int): Size of the square crop

    Returns:
    Same type as input: Center-cropped image
    """
    # Check if input is NumPy array
    is_numpy = isinstance(image, np.ndarray)

    # If NumPy array, convert to torch tensor
    if is_numpy:
        # Remember original dtype for later conversion back
        original_dtype = image.dtype
        # Convert to torch tensor
        torch_image = torch.from_numpy(image)
    else:
        torch_image = image

    # Apply center crop using torchvision
    cropper = CenterCrop(crop_size)

    # Handle different dimensions
    if torch_image.dim() == 2:
        # Add batch and channel dims for 2D image
        torch_image = torch_image.unsqueeze(0).unsqueeze(0)
        cropped = cropper(torch_image).squeeze(0).squeeze(0)
    elif torch_image.dim() == 3:
        # Add batch dim for 3D tensor
        torch_image = torch_image.unsqueeze(0)
        cropped = cropper(torch_image).squeeze(0)
    else:
        # Use as is for 4D or higher
        cropped = cropper(torch_image)

    # Convert back to NumPy if the input was NumPy
    if is_numpy:
        return cropped.numpy().astype(original_dtype)

    return cropped


# Global variables for optimization and drag control
# Set the propagation parameters
num_fields = 3
system_mag = 200/125
original_shape = 2472
wavelength = 632.8e-9  # Wavelength (m)
last_update_time = 0
update_interval = 0.1  # Update every 100ms
preview_scale = 0.25  # Scale factor for preview during drag
is_dragging = False
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os





# Note: This function requires the fresnel_propagation function from the original script
# Make sure it's imported or defined in the same scope


# Note: This function requires the fresnel_propagation function from the original script
# Make sure it's imported or defined in the same scope

# Function to save images for GIF creation
def load_array_or_image2(file_path, device):
    if file_path.endswith('.npy'):
        return torch.from_numpy(np.load(file_path)).to(device)
    else:
        image = Image.open(file_path)
        if image.mode == 'I;16':
            image_array = np.array(image, dtype=np.uint16)
            image_array = image_array.astype(np.float32)  # Convert to float32 for PyTorch
        else:
            image_array = np.array(image, dtype=np.float32)
        return torch.from_numpy(image_array).to(device)

def load_images_as_tensor(files, device):
    images = []
    with tqdm(total=len(files), desc="Loading images") as pbar:
        for file in files:
            image = load_array_or_image2(file, device)  # Now this will work correctly
            images.append(image)
            pbar.update(1)
    return torch.stack(images)

def select_files_or_folder():
    image_types = ['*.jpg', '*.tif', '*.png', '*.npy']
    root = tk.Tk()
    root.withdraw()
    messagebox_result = messagebox.askquestion('Choose', 'Select a Folder?', icon='question')
    if messagebox_result == 'yes':
        folder = filedialog.askdirectory()
        if folder:
            files = []
            for image_type in image_types:
                files.extend(glob.glob(os.path.join(folder, image_type)))
            return [files, folder]
        return [], None
    else:
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.npy;*.tif;*.png")])
        folder = os.path.dirname(files[0]) if files else None
        return [files, folder]

def norm(field):
    if isinstance(field, torch.Tensor):
        field /= torch.max(torch.abs(field))
    elif isinstance(field, np.ndarray):
        field /= np.max(np.abs(field))
    else:
        raise TypeError("Unsupported type. The input should be a NumPy array or a PyTorch tensor.")
    return field

def gauss2D(a, size = 1025,mid = None):
    if mid is None:
        mid = [0,0]
    x = torch.linspace(-size/2, size/2, size,dtype=torch.float32)
    x, y = torch.meshgrid(x, x)
    r2 = (x-mid[0])**2 + (y-mid[1])**2
    g = torch.exp(-0.5*r2/a**2)
    return g/g.max()

def apply_circular_mask(image: torch.Tensor, radius_factor: float) -> torch.Tensor:
    """
    Apply a circular mask to a square image tensor.

    Args:
    image (torch.Tensor): A 2D square image tensor (H x W)
    radius_factor (float): A number between 0 and 1 determining the size of the circular mask

    Returns:
    torch.Tensor: The input image with the circular mask applied
    """
    if not (0 <= radius_factor <= 1):
        raise ValueError("radius_factor must be between 0 and 1")

    if image.shape[0] != image.shape[1]:
        raise ValueError("Input image must be square")

    size = image.shape[0]
    center = size // 2

    y_indices = torch.arange(size).unsqueeze(1).float().to(image.device)
    x_indices = torch.arange(size).unsqueeze(0).float().to(image.device)

    dist_from_center = torch.sqrt((x_indices - center) ** 2 + (y_indices - center) ** 2)

    mask = dist_from_center <= (radius_factor * size / 2)

    return image * mask.float()

def roll_fft(event):
    ax = event.inaxes  # Get the axis where the event occurred

    # Find which axis is being interacted with
    for i, current_ax in enumerate(axs):
        if ax == current_ax:
            y, x = int(event.ydata), int(event.xdata)
            rows_to_roll = rolled_fields_fft[i].shape[0] // 2 - y
            cols_to_roll = rolled_fields_fft[i].shape[1] // 2 - x

            # Roll the FFT for the current field and update the plot
            rolled_fields_fft[i] = torch.roll(rolled_fields_fft[i], shifts=(rows_to_roll, cols_to_roll), dims=(0, 1))
            update_plot(current_ax, rolled_fields_fft[i])
            break  # Stop searching once the correct axis is found

    plt.draw()  # Redraw the figure

def update_plot(ax, field):
    rgb_image = create_rgb_image(field)
    ax.get_images()[0].set_data(rgb_image)

def create_rgb_image(field, vmin=None, vmax=None):
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    phase = np.angle(field)
    amplitude = np.abs(field) ** 0.5

    # Set vmin and vmax for dynamic range control
    if vmin is None:
        vmin = np.min(amplitude)
    if vmax is None:
        vmax = np.max(amplitude)

    # Clip and normalize amplitude to [0, 1] based on vmin and vmax
    amplitude = np.clip((amplitude - vmin) / (vmax - vmin), 0, 1)

    # Normalize phase to range [0, 1]
    phase_normalized = (phase + np.pi) / (2 * np.pi)

    # Create HSV image
    hsv_image = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.float64)
    hsv_image[:, :, 0] = phase_normalized  # Hue (based on phase)
    hsv_image[:, :, 1] = 1.0  # Saturation
    hsv_image[:, :, 2] = amplitude  # Value (based on normalized amplitude)

    # Convert HSV to RGB
    rgb_image = np.zeros_like(hsv_image)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            rgb_image[i, j] = colorsys.hsv_to_rgb(hsv_image[i, j, 0], hsv_image[i, j, 1], hsv_image[i, j, 2])

    return rgb_image

def phplot(field, ax=None, vmin=None, vmax=None):
    # Create the RGB image with the new dynamic range (vmin, vmax)
    rgb_image = create_rgb_image(field, vmin=vmin, vmax=vmax)

    if ax is None:
        ax = plt.gca()

    # Plot the image
    im = ax.imshow(rgb_image)
    ax.axis('off')

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_ticks = np.linspace(-np.pi, np.pi, 9)
    cbar_labels = [f'{t:.2f}' for t in cbar_ticks]
    cbar.set_ticks(np.linspace(0, 1, 9))  # Adjust for normalized values
    cbar.set_ticklabels(cbar_labels)
    cbar.set_label('Phase (radians)')

    return im

def drag_roll(event):
    global rolled_fields_fft, start_x, start_y, last_update_time, is_dragging  # Declare globals

    if not is_dragging:
        return

    current_time = time.time()
    if current_time - last_update_time < update_interval:
        return  # Skip this update if it's too soon

    # Iterate over all axes to find which one is being interacted with
    for i, current_ax in enumerate(axs):
        if event.inaxes == current_ax:
            dx = int(start_x - event.xdata)
            dy = int(start_y - event.ydata)

            # Roll the FFT for the selected field
            rolled_fields_fft[i] = torch.roll(rolled_fields_fft[i], shifts=(-dy, -dx), dims=(0, 1))
            update_plot_preview(current_ax, rolled_fields_fft[i])

            # Update starting coordinates for drag
            start_x, start_y = event.xdata, event.ydata
            last_update_time = current_time
            break  # Stop searching once the correct axis is found

    plt.draw()  # Redraw the figure

def update_plot_preview(ax, field):
    preview = create_preview(field)
    ax.get_images()[0].set_data(preview)

def create_preview(field):
    # Create a lower resolution preview
    preview = field[::4, ::4]  # Take every 4th pixel
    return create_rgb_image(preview)

def on_press(event):
    global start_x, start_y, is_dragging  # Declare globals

    # Check if the event happened within any of the axes in the axs list
    for current_ax in axs:
        if event.inaxes == current_ax:
            start_x, start_y = event.xdata, event.ydata
            is_dragging = True
            break  # Stop once the correct axis is found

def on_release(event):
    global start_x, start_y, is_dragging  # Declare globals

    if is_dragging:
        # Iterate over the axes and corresponding rolled FFT fields
        for i, current_ax in enumerate(axs):
            if event.inaxes == current_ax:
                # Update the plot for the axis where the release event occurred
                update_plot(current_ax, rolled_fields_fft[i])
                break  # Stop once the correct axis is found

    # Reset dragging state
    start_x, start_y = 0, 0
    is_dragging = False
    plt.draw()

def confirm(event):
    plt.close()

def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    if file_path.endswith('.trc'):
        return torch.load(file_path).numpy()
    else:
        image = Image.open(file_path)
        return np.array(image)

def propagate_field(field_tensor, pixel_size, wavelength, distance, method='AS'):
    """
    Propagate a PyTorch tensor field using diffractsim.
    Handles both single fields (2D tensor) and batches (3D tensor).

    Parameters:
    -----------
    field_tensor : torch.Tensor
        Complex field tensor on GPU. Can be 2D (single field) or 3D (batch of fields)
    pixel_size : float
        Pixel size in meters
    wavelength : float
        Wavelength in meters
    distance : float
        Propagation distance in meters
    method : str, optional
        Propagation method ('AS' for Angular Spectrum, 'fraunhofer' for Fraunhofer)

    Returns:
    --------
    torch.Tensor
        Propagated field(s) on the same device as the input tensor
    """
    # Save original device
    device = field_tensor.device

    # Check if this is a batch (3D tensor) or single field (2D tensor)
    is_batch = len(field_tensor.shape) == 3

    if is_batch:
        # Process batch by looping through each field
        batch_size = field_tensor.shape[0]
        results = []

        for i in range(batch_size):
            # Extract single field
            single_field = field_tensor[i]

            # Convert to NumPy array
            field_numpy = single_field.cpu().detach().numpy()

            # Create MonochromaticField
            F = MonochromaticField(
                wavelength=wavelength,
                extent_x=field_numpy.shape[1] * pixel_size,
                extent_y=field_numpy.shape[0] * pixel_size,
                Nx=field_numpy.shape[1],
                Ny=field_numpy.shape[0]
            )

            # Set field data
            F.E = field_numpy

            # Propagate
            if method.lower() == 'fraunhofer':
                F.propagate(distance, method='fraunhofer')
            else:
                F.propagate(distance)  # Default is Angular Spectrum

            # Convert back to PyTorch tensor and append
            result_tensor = torch.tensor(F.E, dtype=torch.complex64).to(device)
            results.append(result_tensor)

        # Stack results to return a batch
        return torch.stack(results)

    else:
        # Original single-field behavior
        # Convert PyTorch tensor to NumPy array
        # check if the field is not numpy array:
        if not isinstance(field_tensor, np.ndarray):
            field_numpy = field_tensor.cpu().detach().numpy()
        else:
            field_numpy = field_tensor

        # Create MonochromaticField
        F = MonochromaticField(
            wavelength=wavelength,
            extent_x=field_numpy.shape[1] * pixel_size,
            extent_y=field_numpy.shape[0] * pixel_size,
            Nx=field_numpy.shape[1],
            Ny=field_numpy.shape[0]
        )

        # Set field data
        F.E = field_numpy

        # Propagate
        if method.lower() == 'fraunhofer':
            F.propagate(distance, method='fraunhofer')
        else:
            F.propagate(distance)  # Default is Angular Spectrum

        # Convert back to PyTorch tensor
        result_tensor = torch.tensor(F.E, dtype=torch.complex64).to(device)
        return result_tensor


def fresnel_propagation(initial_field, dx, dy, z, wavelength):
    # Convert initial field to PyTorch tensor and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(initial_field, torch.Tensor):
        field = torch.from_numpy(initial_field).to(device).to(dtype=torch.complex64)
    else:
        field = initial_field.to(device).to(dtype=torch.complex64)
    [Nx, Ny] = field.shape
    k = 2 * torch.pi / wavelength
    # Create coordinate grids
    fx = torch.fft.fftfreq(Nx, dx).to(device)
    fy = torch.fft.fftfreq(Ny, dx).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    # Fresnel kernel
    # Ensure z is a tensor for compatibility
    z_tensor = torch.tensor(z, device=device, dtype=torch.float32)
    H = torch.exp(torch.tensor(-1j * k, device=device) * z_tensor) * torch.exp(
        torch.tensor(-1j * torch.pi * wavelength, device=device) * z_tensor * (FX ** 2 + FY ** 2)
    )
    H_temp = H * 0
    t = (FX ** 2 + FY ** 2)**0.5
    condition = t < (1 / wavelength)
    H_temp[condition] = torch.exp(
        1j * k *1000* z_tensor * (1 - (wavelength * FX[condition]) ** 2 - (wavelength * FY[condition]) ** 2) ** 0.5)
    # Propagated field
    field_propagated = ifft(fft(field) * H)
    return field_propagated

def update_sliders(val):
    for i, slider in enumerate(sliders):
        extent = extents[i]
        z_index = int((slider.val - z_min) / ((z_max - z_min) / (num_z - 1)))

        # Update the image data for each field
        axs[i].images[0].set_data(np.abs(fields_all_numpy[i][z_index]))
        axs[i].set_title(f'Propagated Field {i + 1} (z = {z_values[z_index]:.2f} m)')

        # PSF size calculation for each field
        psf_size = theta * z_values[z_index] * 1e3  # PSF size in mm
        normalized_psf_size = np.abs(psf_size / (extent[1] - extent[0]))  # Normalize relative to the plot's width

        # Update the PSF line for each field
        psf_lines[i].set_data([0.05, 0.05 + normalized_psf_size], [0.9, 0.9])  # Horizontal line indicating PSF size

        # Restore the zoom levels for each axis only if no manual zoom occurred during slider change
        if xlims and ylims:
            axs[i].set_xlim(xlims[i])
            axs[i].set_ylim(ylims[i])

    fig.canvas.draw_idle()  # Update the canvas

def store_zoom_levels(event):
    global xlims, ylims
    xlims = [ax.get_xlim() for ax in axs]
    ylims = [ax.get_ylim() for ax in axs]


# Load the initial field from a .npy file
init_dir = 'D:\Lab Images and data local'

# Store the file paths in a list and load fields accordingly
field_paths = []
root = tk.Tk()
root.withdraw()

# Loop through to ask for field input based on num_fields
for i in range(num_fields):
    field_path = filedialog.askopenfilename(
        initialdir=init_dir,
        filetypes=[("NumPy files", "*.npy;*.trc"), ("Image files", "*.jpeg;*.jpg;*.tiff;*.tif;*.png")],
        title=f'Select Field {i+1} to propagate.'
    )
    field_paths.append(field_path)
del field_path, i
root.destroy()

# Load arrays or images and convert to 2D NumPy arrays for each field
original_fields = [load_array_or_image(path) for path in field_paths]

# Convert to PyTorch tensors and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert to PyTorch tensors and move to GPU if available
original_fields_tensors = [torch.from_numpy(field).to(device).to(torch.complex64) for field in original_fields]

# Normalize fields
original_fields_tensors = [norm(field) for field in original_fields_tensors]
#
#
# # creating propagated point field
# z = 0.07148 # Propagation distance in meters
# grid_size = original_fields_tensors[0].shape[0]  # Size of the grid (850x850)
# mag = 200/125 * grid_size / 2472  # Magnification for the current field
# dx = dy = 5.5e-6 / mag
# pixel_size = dx  # Pixel size in meters (adjust as needed)
# wavelength = 632.8e-9
#
# # Create a coordinate grid centered at (0, 0)
# x = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
# y = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
# X, Y = np.meshgrid(x, y)
#
# # Calculate radial distance from the point source (assumed at the origin)
# r = np.sqrt(X**2 + Y**2 + z**2)
#
# # Calculate field amplitude and phase at distance z from the source
# amplitude = torch.from_numpy(1 / r)  # Amplitude falls off as 1/r
# phase = torch.from_numpy((2 * np.pi / (wavelength)) * r)  # Phase term
#
# original_fields_tensors[0] *= torch.exp(1j * phase.to(device))
# Compute FFTs for all fields
fields_fft = [fftshift(fft(field)) for field in original_fields_tensors]

# Create the figure and subplots for dynamic number of fields
fig, axs = plt.subplots(1, num_fields, figsize=(12, 6))

# Store rolled versions of the FFT fields
rolled_fields_fft = [field_fft.clone() for field_fft in fields_fft]

# Plot each field dynamically
for i, ax in enumerate(axs):
    phplot(rolled_fields_fft[i], ax)
    ax.set_title(f'Field {i+1} FFT')

# Add zoom and drag functionality (adjust as needed for each field)
start_x, start_y = 0, 0
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', drag_roll)

# Remove the old roll_fft connection
# Confirm button
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Confirm')
button.on_clicked(confirm)

plt.show(block=True)


# After closing the plot, use the rolled FFTs for each field
rolled_fields = [ifft(ifftshift(field_fft)) for field_fft in rolled_fields_fft]
del original_fields, fields_fft, i, is_dragging, rolled_fields_fft, start_x, start_y, update_interval, last_update_time


# pad fields:
padding_size = 500
padded_fields = [F.pad(field, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
                 for field in rolled_fields]


# Initialize lists to store dx, dy, and other parameters for each field
dx_all = []
dy_all = []
blob_shapes = [field.shape[0] for field in rolled_fields]  # Get blob shapes for all fields

# Loop through the fields and calculate individual dx, dy, etc. based on blob shape
for blob_shape in blob_shapes:
    mag = system_mag * blob_shape / original_shape  # Magnification for the current field
    dx = dy = 5.5e-6 / mag  # Pixel size in x and y directions (m)
    dx_all.append(dx)
    dy_all.append(dy)
del blob_shape, dx, dy

# Compute the field at different z-positions
z_min_mm = 0
z_max_mm = 90
z_min, z_max = z_min_mm * 1e-3, z_max_mm * 1e-3
step_in_mm = 2
num_z = int((z_max_mm - z_min_mm) // step_in_mm + 1)
z_values = torch.linspace(z_min, z_max, num_z)

# Check if 0 is in the range and not already in z_values
if 0 not in z_values:
    z_values = torch.cat([z_values, torch.tensor([0.0])])
    z_values = torch.sort(z_values)[0]  # Ensure sorted order

# Round the values
z_values = torch.round(z_values, decimals=5)

# Find the index of 0
zero_index = torch.where(z_values == 0)[0].item()

del step_in_mm, z_min_mm, z_max_mm
# Compute the field at different z-positions for all fields
fields_all = [[] for _ in range(num_fields)]  # A list to hold computed fields for each z-value and each field

# Start and end indices for slicing will be calculated for each field
start_rows = []
end_rows = []
start_cols = []
end_cols = []

for field in rolled_fields:
    original_height, original_width = field.shape  # Get the original dimensions of each field
    start_row = padding_size
    end_row = start_row + original_height
    start_col = padding_size
    end_col = start_col + original_width

    start_rows.append(start_row)
    end_rows.append(end_row)
    start_cols.append(start_col)
    end_cols.append(end_col)

del start_row, end_row, start_col, end_col, field

# Loop through z-values and propagate the field for each one
for z in tqdm(z_values, desc='Computing fields'):
    for i, padded_field in enumerate(padded_fields):
        dx = dx_all[i]
        dy = dy_all[i]

        # Perform Fresnel propagation for each field
        # field_propagated = fresnel_propagation(padded_field, dx, dy, z, wavelength)
        field_propagated = angular_spectrum_gpu(padded_field,dx,wavelength,z)
        # field_propagated = propagate_field(padded_field, dx, z, wavelength)
        field_propagated /= torch.max(torch.abs(field_propagated))  # Normalize

        # Slice the padded field to retrieve the original field size
        original_field = field_propagated[start_rows[i]:end_rows[i], start_cols[i]:end_cols[i]]
        fields_all[i].append(norm(original_field))  # Normalize the field before storing

del start_rows, end_rows, start_cols, end_cols, z, i, padded_field
# Convert fields to NumPy for visualization
fields_all_numpy = [[tensor.cpu().numpy() for tensor in fields] for fields in fields_all]

# Plot the propagated fields for each field at z=0
fig, axs = plt.subplots(1, num_fields, figsize=(12, 6))

plt.subplots_adjust(bottom=0.3)
extents = []
for i, ax in enumerate(axs):
    extent = [0, rolled_fields[i].shape[1] * dx_all[i] * 1e3, 0, rolled_fields[i].shape[0] * dy_all[i] * 1e3]
    im = ax.imshow(np.abs(fields_all_numpy[i][zero_index]), cmap='hot', extent=extent)
    ax.set_title(f'Propagated Field {i+1}')
    plt.colorbar(im, ax=ax)
    extents.append(extent)


# Now we can add sliders or additional elements to handle dynamic updates for z-values
# Initialize the PSF lines and text labels in axes coordinates for all fields
psf_lines = []
text_labels = []

for i, ax in enumerate(axs):
    psf_line = Line2D([0, 0], [0, 0], color='white', linewidth=2, transform=ax.transAxes)
    ax.add_line(psf_line)
    psf_lines.append(psf_line)
    text_label = ax.text(0.05, 0.95, 'PSF size', color='white', fontsize=12, ha='left', va='top', transform=ax.transAxes)
    text_labels.append(text_label)

# Define sliders for z-values and connect them to an update function
sliders = []
for i in range(num_fields):
    ax_slider = plt.axes([0.25, 0.2 - 0.05 * i, 0.5, 0.03])
    slider = Slider(
        ax_slider, f'z (m) - Field {i+1}', z_min, z_max, valinit=0, valstep=(z_max - z_min) / (num_z - 1)
    )
    sliders.append(slider)

# Store the initial zoom levels for all fields
xlims = [ax.get_xlim() for ax in axs]  # Store x-axis limits for each field
ylims = [ax.get_ylim() for ax in axs]  # Store y-axis limits for each field
theta = np.deg2rad(0.5)


# Add the sliders and connect the update function
for slider in sliders:
    slider.on_changed(update_sliders)

fig.canvas.mpl_connect('button_release_event', store_zoom_levels)


# Trigger an initial plot update for all sliders
plt.ioff()
update_sliders(sliders[0].val)
plt.show(block=False)



pos = 0.07148
pos = np.round(pos, 5)


idx_at_obj = (np.abs(z_values - pos)).argmin()
uncorrected_at_diffuser_plane = fields_all[0][zero_index].cpu().numpy()
uncorrected_at_obj_plane = fields_all[0][idx_at_obj].cpu().numpy()
corrected_at_diffuser_plane = fields_all[1][zero_index].cpu().numpy()
corrected_at_obj_plane = fields_all[1][idx_at_obj].cpu().numpy()
gt_at_diffuser_plane = fields_all[2][zero_index].cpu().numpy()
gt_at_obj_plane = fields_all[2][idx_at_obj].cpu().numpy()
diffuser_phase_from_GT = np.exp(1j * np.angle(uncorrected_at_diffuser_plane / gt_at_diffuser_plane))
diffuser_phase = np.exp(1j * np.angle(uncorrected_at_diffuser_plane / corrected_at_diffuser_plane))
AS = propagate_field(gt_at_obj_plane, dx ,wavelength, -pos)

plt.figure()
# plotting the phase:
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(np.abs(gt_at_diffuser_plane), cmap='hot')
plt.colorbar(label="from GT")
plt.title("In diffuser plane")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(gt_at_obj_plane), cmap='hot')
plt.colorbar(label="from corrected image")
plt.title("Fresnel")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.imshow(np.abs(propagate_field(gt_at_diffuser_plane, dx ,wavelength, pos)), cmap='hot')
plt.colorbar(label="from GT")
plt.title("AS")
plt.xlabel("x (m)")
plt.ylabel("y (m)")





create_propagation_video(
    field=center_crop(corrected_at_diffuser_plane, 650),  # Start with the field at the diffuser plane
    z_initial=0,                          # Start at z=0
    z_final=1.5 * pos,                         # End at z=80mm (adjust as needed)
    num_of_frames=150,                    # Generate 100 frames
    fps=25,                               # 30 frames per second
    dx=dx_all[0],                         # Use the pixel size from your first field
    dy=dy_all[0],
    wavelength=wavelength,
    output_filename='corrected_propagation.mp4',
    cmap='hot',                           # Use 'hot' colormap for intensity
    padding_size=500,                     # Same padding size as used in your code
    show_progress=True
)

# Create a side-by-side comparison of all three fields
create_comparison_propagation_video(
    fields=[uncorrected_at_diffuser_plane, corrected_at_diffuser_plane, gt_at_diffuser_plane],
    field_names=["Uncorrected", "Corrected", "Ground Truth"],
    z_initial=0,
    z_final=1.5 * pos,  # 80mm propagation distance
    num_of_frames=100,
    fps=20,
    dx_values=dx_all,  # List of dx values for each field
    dy_values=dy_all,  # List of dy values for each field
    wavelength=wavelength,
    output_filename='three_field_comparison.mp4',
    cmap='hot',
    padding_size=500,
    show_progress=True,
    layout='horizontal'  # Side-by-side layout
)

files, folder = select_files_or_folder()
uncorrected_tensor = load_images_as_tensor(files, device)
# Create padded versions of the fields
padding_size = 500
padded_fields = [F.pad(field, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
                 for field in uncorrected_tensor]

# Propagate each field to the object plane
uncorrected_at_obj_tensor = []
for i, padded_field in enumerate(padded_fields):
    padded_field = padded_fields[i]
    # Propagate to object plane using existing dx_all and dy_all
    # field_propagated = fresnel_propagation(padded_field, dx, dy, pos, wavelength)
    field_propagated = angular_spectrum_gpu(padded_field, dx, wavelength, pos)

    # Remove padding (get back to original size)
    original_height, original_width = uncorrected_tensor[0].shape
    start_row = padding_size
    end_row = start_row + original_height
    start_col = padding_size
    end_col = start_col + original_width

    # Slice and normalize
    field_at_obj = field_propagated[start_row:end_row, start_col:end_col]
    field_at_obj = norm(field_at_obj)

    uncorrected_at_obj_tensor.append(field_at_obj)

# Stack all fields together
uncorrected_at_obj_tensor = torch.stack(uncorrected_at_obj_tensor)

# 1. Convert tensors to numpy arrays
# For phases
phase_frames = []
for i in range(uncorrected_tensor.size(0)):
    current_field = uncorrected_tensor[i].cpu().numpy()
    current_diffuser_phase = np.exp(1j * np.angle(current_field / corrected_at_diffuser_plane))
    phase_frames.append(np.angle(current_diffuser_phase))
phase_frames = np.array(phase_frames)

# For fields at object plane
amplitude_frames = np.array([field.cpu().numpy() for field in uncorrected_at_obj_tensor])

# Set desired fps and calculate duration in milliseconds
fps = 4  # Change this value to control speed
duration_ms = int(1000.0 / fps)  # Convert to milliseconds


def create_gif(frames, output_filename, cmap, duration_ms):
    frame_filenames = []

    # Save each frame as an image
    for i, frame in enumerate(frames):
        plt.figure(figsize=(6, 6))
        plt.imshow(frame, cmap=cmap)
        plt.colorbar(label="Phase (rad)" if "phase" in output_filename else "Amplitude")
        plt.title(f"Frame {i + 1}")
        plt.axis('off')

        filename = os.path.join(temp_dir, f'frame_{i}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        frame_filenames.append(filename)

    # Create GIF with duration instead of fps
    with imageio.get_writer(output_filename, mode='I', duration=duration_ms) as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up temporary files
    for filename in frame_filenames:
        os.remove(filename)


# Create temporary directory
temp_dir = 'temp_frames'
os.makedirs(temp_dir, exist_ok=True)

# Create GIFs
create_gif(phase_frames, 'phases.gif', cmap='hsv', duration_ms=duration_ms)
create_gif(np.abs(amplitude_frames)**2, 'fields_at_object.gif', cmap='hot', duration_ms=duration_ms)

# Remove temporary directory
os.rmdir(temp_dir)

print("GIFs created successfully!")



a=uncorrected_tensor[0]
# plotting the phase:
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(uncorrected_at_obj_plane), cmap='hot')
plt.colorbar(label="from GT")
plt.title("diffuser_phase_from_GT")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.subplot(1, 2, 2)
plt.imshow(np.abs(gt_at_obj_plane), cmap='hot')
plt.colorbar(label="from corrected image")
plt.title("Phase of the Point Source Field")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.show()


### mask
mask = np.abs(np.zeros_like(uncorrected_at_obj_plane))
left_col = 260
right_col = 600
top_row = 160
bottom_row = 650
mask[top_row:bottom_row, left_col:right_col] = 1
smooth_mask = gaussian_filter(mask, sigma=20)

# creating propagated point field
z = pos # Propagation distance in meters
grid_size = uncorrected_at_obj_plane.shape[0]  # Size of the grid (850x850)
pixel_size = dx_all[0]  # Pixel size in meters (adjust as needed)

# Create a coordinate grid centered at (0, 0)
x = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
y = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
X, Y = np.meshgrid(x, y)

# Calculate radial distance from the point source (assumed at the origin)
eps = 1e-12  # Small epsilon to prevent division by zero
r = np.sqrt(X ** 2 + Y ** 2 + z ** 2) + eps

# Calculate field amplitude and phase at distance z from the source
amplitude = 1 / r * smooth_mask # Amplitude falls off as 1/r
phase = (2 * np.pi / (wavelength)) * r * mask # Phase term
propagated_point_field = amplitude * np.exp(1j * phase)  # Complex field
propagated_point_field /= np.max(np.abs(propagated_point_field))


# Plotting the intensity and phase of the field
plt.figure(figsize=(12, 5))
# Plot intensity
plt.subplot(1, 2, 1)
plt.imshow(np.abs(propagated_point_field), cmap='inferno', extent=extents[0], vmin=0, vmax=1)
plt.colorbar(label="Intensity")
plt.title("Intensity of the Point Source Field")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
# Plot phase
plt.subplot(1, 2, 2)
plt.imshow(np.angle(propagated_point_field), cmap='hsv', extent=extents[0])
plt.colorbar(label="Phase (radians)")
plt.title("Phase of the Point Source Field")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()

field_to_propagate = propagated_point_field  * smooth_mask * diffuser_phase
psf = nel_propagation(field_to_propagate, dx_all[0], dy_all[0], -pos, wavelength).cpu().numpy()
psf_fft = np.fft.fftshift(np.fft.fft2(psf))
padded_psf_fft = F.pad(torch.from_numpy(psf_fft), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0).numpy()
oversampled_psf = np.fft.ifft2(np.fft.ifftshift(padded_psf_fft))

gt_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gt_at_obj_plane)))
distorted_test = np.fft.ifft2(np.fft.ifftshift(psf_fft * gt_fft))
psf /= np.max(np.abs(psf)) # Normalize

fig, ax = plt.subplots(1,4)
im1 = ax[0].imshow(np.abs(distorted_test), cmap='hot')
ax[0].set_title('mag of estimated uncorrected at obj plane')

im2 = ax[1].imshow(np.abs(uncorrected_at_obj_plane), cmap='hot')
ax[1].set_title('mag of uncorrected at obj plane')

im3 = ax[2].imshow(np.abs(psf))
ax[2].set_title('mag of psf at obj plane')

im4 = ax[3].imshow(np.abs(oversampled_psf))
ax[3].set_title('mag of corrected obj plane')
plt.show()



plt.figure()
phplot(psf)
plt.show()
plt.figure()
phplot(oversampled_psf)
plt.show()
plt.figure()
phplot(oversampled_psf[750:1100, 750:1100])
plt.show()


grid_size = psf.shape[0]
edge_width = 50  # Width of the edge region for tapering (adjust as needed)

# Create a 1D Hanning window for the edges
edge_taper_x = np.ones(grid_size)
edge_taper_y = np.ones(grid_size)

# Apply Hanning window to the edges
edge_taper_x[:edge_width] = np.hanning(edge_width * 2)[:edge_width]  # Left edge
edge_taper_x[-edge_width:] = np.hanning(edge_width * 2)[edge_width:]  # Right edge
edge_taper_y[:edge_width] = np.hanning(edge_width * 2)[:edge_width]  # Top edge
edge_taper_y[-edge_width:] = np.hanning(edge_width * 2)[edge_width:]  # Bottom edge

# Create a 2D tapering mask by outer product
smooth_edge_mask = np.outer(edge_taper_y, edge_taper_x)

padding_size = 850


central_region_size = 350

# Calculate the start and end indices for the central 450 x 450 region
start = (grid_size - central_region_size) // 2
end = start + central_region_size

# Create a mask with ones in the central 450 x 450 region and zeros elsewhere
mask = np.zeros((grid_size, grid_size), dtype=np.complex64)
mask[start:end, start:end] = 1
masked_amplitude = 1 / r * mask # Amplitude falls off as 1/r
masked_phase = (2 * np.pi / (wavelength)) * r * mask # Phase term
masked_field = amplitude * np.exp(1j * phase) * mask # Complex field
masked_field = norm(masked_field)

field_to_propagate = propagated_point_field *  smooth_mask * diffuser_phase
field_to_propagate = masked_field
field_to_propagate_fft = np.fft.fftshift(np.fft.fft2(field_to_propagate))
# Assume field_to_propagate_fft is the frequency-domain field after fftshift

# Apply the edge-only smooth tapering mask to the FFT field
smooth_field_to_propagate_fft = field_to_propagate_fft# * smooth_edge_mask
# smooth_field_to_propagate_fft = norm(smooth_field_to_propagate_fft)
padded_field_to_propagate_fft = F.pad(torch.from_numpy(smooth_field_to_propagate_fft), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0).numpy()
oversampled_field_to_propagate = np.fft.ifft2(np.fft.ifftshift(padded_field_to_propagate_fft))
oversampled_field_to_propagate = norm(oversampled_field_to_propagate)


fig, ax = plt.subplots(1,4)
im1 = ax[0].imshow(np.abs(field_to_propagate))
ax[0].set_title('field_to_propagate')

im2 = ax[1].imshow(np.abs(field_to_propagate_fft))
ax[1].set_title('field_to_propagate_fft')

im3 = ax[2].imshow(np.round(np.abs(oversampled_field_to_propagate), 5))
ax[2].set_title('oversampled_field_to_propagate')

im4 = ax[3].imshow(np.abs(field_to_propagate_fft))
ax[3].set_title('mag of corrected obj plane')
plt.show()



idx1 = idx_at_obj - (180-idx_at_obj)
corrected_before_obj_plane = fields_all[1][idx1].cpu().numpy()
idx2 = idx_at_obj
corrected_at_obj_plane = fields_all[1][idx2].cpu().numpy()
idx3 = 180
corrected_after_obj_plane = fields_all[1][idx3].cpu().numpy()




# Plotting uncorrected, corrected, and GT fields with a consistent scale bar
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the figsize to fit the images

# Uncorrected on camera (first field)
im0 = axes[0].imshow(np.abs(corrected_before_obj_plane), cmap='hot', extent=extents[0])
axes[0].set_title(f'corrected_before_obj_plane (z = {np.round(1e2 * z_values[idx1], 2)} cm)')
plt.colorbar(im0, ax=axes[0])  # Add colorbar

# Uncorrected at object plane (first field propagated to object plane)
im1 = axes[1].imshow(np.abs(corrected_at_obj_plane), cmap='hot', extent=extents[0])  # Reuse extent[0]
axes[1].set_title(f'corrected_at_obj_plane (z = {np.round(1e2 * z_values[idx2], 4)} cm)')
plt.colorbar(im1, ax=axes[1])  # Add colorbar

# Corrected at object plane (second field)
im2 = axes[2].imshow(np.abs(corrected_after_obj_plane), cmap='hot', extent=extents[1])  # Use the correct extent
axes[2].set_title(f'corrected_after_obj_plane (z = {np.round(1e2 * z_values[idx3], 4)} cm)')
plt.colorbar(im2, ax=axes[2])  # Add colorbar


# Add a 1 mm scale bar to each plot
scale_length_mm = 1  # 1 mm
for i, ax in enumerate(axes):
    # For the first and second images (uncorrected on camera and at object plane), use extent[0]
    extent = extents[0] if i < 2 else extents[i-1]  # Reuse extent[0] for the first two images

    scale_x_start = 0.05 * (extent[1] - extent[0]) + extent[0]  # Start 5% from the left
    scale_x_end = scale_x_start + scale_length_mm  # 1 mm long scale bar
    scale_y = 0.05 * (extent[3] - extent[2]) + extent[2]  # 5% up from the bottom

    # Add scale lines in data coordinates
    scale_line = Line2D([scale_x_start, scale_x_end], [scale_y, scale_y], color='white', linewidth=2)
    ax.add_line(scale_line)

    # Add text label for the scale line
    ax.text(scale_x_start, scale_y + 0.05 * (extent[3] - extent[2]), '1 mm', color='white', fontsize=12)

# Display the plot
plt.tight_layout()  # Adjust the layout to prevent overlap
plt.show(block=True)
a=6