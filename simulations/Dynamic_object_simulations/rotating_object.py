import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.transforms.functional import rotate
from utils.io import load_file_to_tensor
from utils.field_utils import generate_diffusers_and_PSFs
from propagation.propagation import angular_spectrum_gpu
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from utils.image_processing import resize, center_crop, shift_cross_correlation, nrm
from core.CTRCLASS import CTR_CLASS
from utils.visualization import display_field, get_custom_colormap
from torch.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_laplace
import torch
from torchvision.transforms.functional import rotate





def rotate_field(arr, angle_degrees, expand=True):
    """Rotate array/tensor by angle_degrees without cutting corners"""
    # 1. Convert to PyTorch tensor if it's numpy (works for both real and complex)
    is_numpy = isinstance(arr, np.ndarray)
    tensor = torch.from_numpy(arr) if is_numpy else arr

    original_device = tensor.device
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(compute_device)

    # 2. Remember if it was real and convert to complex if needed
    was_real = not torch.is_complex(tensor)
    if was_real:
        tensor = tensor.to(torch.complex64)

    # 3. Handle rotation based on dimensions
    if tensor.dim() == 2:
        # Split into real and imaginary parts for rotation
        real_part = rotate(tensor.real.unsqueeze(0).unsqueeze(0), angle_degrees, expand=expand).squeeze(0).squeeze(0)
        imag_part = rotate(tensor.imag.unsqueeze(0).unsqueeze(0), angle_degrees, expand=expand).squeeze(0).squeeze(0)
        rotated = real_part + 1j * imag_part
    else:  # 3D tensor
        real_part = rotate(tensor.real.unsqueeze(0), angle_degrees, expand=expand).squeeze(0)
        imag_part = rotate(tensor.imag.unsqueeze(0), angle_degrees, expand=expand).squeeze(0)
        rotated = real_part + 1j * imag_part

    # 4. Convert back to real if original was real
    if was_real:
        rotated = rotated.real

    # 5. Move back to original device (if not numpy)
    if not is_numpy:
        rotated = rotated.to(original_device)
    else:
        # Move to CPU before converting to numpy
        rotated = rotated.cpu()

    # 6. Return as numpy if input was numpy
    return rotated.numpy() if is_numpy else rotated


def show_fields_gif(fields, fps=2):
    """
    Show a sequence of 2D fields (real or complex) as an animation in the console.

    Parameters:
    -----------
    fields : np.ndarray or torch.Tensor
        A 3D array of shape (M, N, N) with complex or real values.
    fps : int
        Frames per second for animation speed.
    """
    # Convert torch tensor to numpy
    if isinstance(fields, torch.Tensor):
        fields = fields.detach().cpu().numpy()

    # Ensure it's 3D
    assert fields.ndim == 3, "Input must be of shape (M, N, N)"

    M = fields.shape[0]

    # Convert to complex if real
    if np.isrealobj(fields):
        fields = fields.astype(np.complex64)

    abs_fields = np.abs(fields)
    phase_fields = np.angle(fields)

    # Prepare the plot
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(abs_fields[0], cmap='gray')
    axes[0].set_title("Amplitude")
    im2 = axes[1].imshow(phase_fields[0], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase")

    for i in range(M):
        im1.set_data(abs_fields[i])
        im2.set_data(phase_fields[i])
        fig.suptitle(f'Frame {i + 1}/{M}')
        plt.pause(1.0 / fps)

    plt.ioff()
    plt.show()


def estimate_rotation1(img1, img2):

    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Convert complex inputs to magnitude
    if np.iscomplexobj(img1):
        img1 = np.abs(img1)
    if np.iscomplexobj(img2):
        img2 = np.abs(img2)

    # Convert to log-polar coordinates
    radius = min(img1.shape) // 2
    precision = 0.1
    angle_bins = int(1 / precision * 360)
    logpolar1 = warp_polar(img1, radius=radius, output_shape=(angle_bins, 360))
    logpolar2 = warp_polar(img2, radius=radius, output_shape=(angle_bins, 360))

    # Use phase correlation to detect shift
    shift, _, _ = phase_cross_correlation(logpolar1, logpolar2)

    # Convert vertical shift to rotation angle
    angle = (shift[0] / angle_bins) * 360
    angle = angle % 360  # normalize to [0, 360)
    return float(angle)


def estimate_rotation(img1, img2, precision=0.05, use_phase=False, upsample_factor=100, normalize=True):
    """
    Estimate rotation angle between img1 and img2 using log-polar phase correlation.

    Parameters:
    -----------
    img1, img2 : np.ndarray or torch.Tensor
        Input 2D fields (real or complex). img2 is assumed to be a rotated version of img1.
    precision : float
        Angular resolution in degrees (e.g., 0.1 for 0.1° steps).
    use_phase : bool
        Whether to use phase (angle of complex field) instead of magnitude.
    upsample_factor : int
        For subpixel estimation in phase correlation.
    normalize : bool
        Whether to return angle in [0, 360) (modulo 360).

    Returns:
    --------
    float
        Estimated rotation angle in degrees.
    """
    # Convert from torch to numpy if needed
    if 'torch' in str(type(img1)):
        img1 = img1.detach().cpu().numpy()
    if 'torch' in str(type(img2)):
        img2 = img2.detach().cpu().numpy()

    # Ensure complex format
    if not np.iscomplexobj(img1):
        img1 = img1.astype(np.complex64)
    if not np.iscomplexobj(img2):
        img2 = img2.astype(np.complex64)

    # Choose magnitude or phase
    if use_phase:
        img1_proc = np.angle(img1)
        img2_proc = np.angle(img2)
    else:
        img1_proc = np.abs(img1)
        img2_proc = np.abs(img2)

    # Compute log-polar transform
    radius = min(img1.shape) // 2
    angle_bins = int(360 / precision)

    logpolar1 = warp_polar(img1_proc, radius=radius, output_shape=(angle_bins, 360))
    logpolar2 = warp_polar(img2_proc, radius=radius, output_shape=(angle_bins, 360))

    # Estimate vertical shift via phase correlation
    shift, _, _ = phase_cross_correlation(logpolar1, logpolar2, upsample_factor=upsample_factor)

    # Convert shift to angle
    angle = (shift[0] / angle_bins) * 360.0

    if normalize:
        angle = angle % 360

    return float(angle)


def preprocess_for_rotation(image, sigma=2):
    """Apply LoG filtering to enhance angular features for rotation estimation."""
    return gaussian_laplace(image, sigma=sigma)


def estimate_rotation_gpu(ref, target, precision=0.5, similarity='correlation', expand=False):
    """
    Estimate rotation using brute-force similarity search on GPU.

    Parameters:
    -----------
    ref : torch.Tensor (2D)
        Reference magnitude image (real-valued).
    target : torch.Tensor (2D)
        Target image to rotate and compare (real-valued).
    angles : torch.Tensor
        Rotation angles to test (in degrees).
    similarity : str
        Metric: 'correlation' (default) or 'mse'
    expand : bool
        If True, expands image during rotation (usually False for fixed-size)

    Returns:
    --------
    float : estimated rotation angle
    """
    assert ref.shape == target.shape, "ref and target must have the same shape"
    assert similarity in ['correlation', 'mse']
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    angles = torch.arange(0, 360, precision, device=compute_device)


    ref = ref.to(compute_device)
    target = target.to(compute_device)

    ref = ref.float()
    target = target.float()

    # Normalize reference
    ref_mean = ref.mean()
    ref_zero = ref - ref_mean
    ref_norm = torch.norm(ref_zero)

    # Prepare inputs for batch rotation
    N = len(angles)

    # Expand target to batch shape (N, 1, H, W)
    target_batch = target.unsqueeze(0).expand(N, -1, -1).clone()  # (N, H, W)
    target_batch = target_batch.unsqueeze(1)  # (N, 1, H, W)

    # Perform batched rotation (looped since torchvision doesn't support angle batching)
    rotated_batch = []
    for i in range(N):
        rotated = rotate(target_batch[i], -angles[i].item(), expand=expand)
        rotated_batch.append(rotated)
    rotated_batch = torch.stack(rotated_batch, dim=0)  # (N, 1, H, W)

    # Crop back to original size if needed
    if rotated_batch.shape[-2:] != ref.shape:
        center_crop = torch.nn.functional.center_crop
        rotated_batch = center_crop(rotated_batch, ref.shape)

    rotated_batch = rotated_batch.squeeze(1)  # (N, H, W)

    if similarity == 'correlation':
        # Normalize each rotated version
        rotated_zero = rotated_batch - rotated_batch.mean(dim=(1, 2), keepdim=True)
        rotated_norm = torch.norm(rotated_zero.view(N, -1), dim=1)

        dot = torch.sum((rotated_zero * ref_zero).view(N, -1), dim=1)
        scores = dot / (rotated_norm * ref_norm + 1e-8)
    elif similarity == 'mse':
        diff = (rotated_batch - ref.unsqueeze(0)) ** 2
        scores = -diff.view(N, -1).mean(dim=1)

    best_idx = torch.argmax(scores)
    return float(angles[best_idx])



# parameters
obj_size = 100
pixel_size_m = 5e-6
speckle_size_m = 20e-6  # Speckle size in meters
wavelength_m = 0.6328e-6 # (632.8 nm)
theta_deg = 0.5
M = 180  # Number of realizations
z_m = 3e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = load_file_to_tensor()
image = resize(image, obj_size)
field = image#*torch.exp(1j*image)


h, w = field.shape[:2]
diagonal = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
# Make diagonal even for better centering
if diagonal % 2 != 0:
    diagonal += 1

# Pre-pad the image to diagonal × diagonal
pad_h = 3*(diagonal - h) // 2 + 4
pad_w = 3*(diagonal - w) // 2 + 4

padded_field = torch.nn.functional.pad(field, (pad_w, pad_w, pad_h, pad_h))
padded_size = padded_field.shape[0]

clean_diffuser, clean_PSF = generate_diffusers_and_PSFs(padded_size, 0 ,speckle_size_m ,pixel_size_m, wavelength_m, 1)
image_at_diffuser_plane = angular_spectrum_gpu(padded_field, pixel_size_m, wavelength_m, z_m) * clean_diffuser.to(device)
widefield = angular_spectrum_gpu(image_at_diffuser_plane, pixel_size_m, wavelength_m, -z_m)


angles = np.linspace(0, 360, M)
rotated_frames = torch.zeros((M, padded_size, padded_size), dtype=torch.complex64, device=device)
for idx, angle in enumerate(angles):
    rotated_frame = rotate_field(padded_field,angle,expand=False)
    rotated_frames[idx] = rotated_frame

show_fields_gif(rotated_frames, fps=30)


rotated_at_diffuser_plane = angular_spectrum_gpu(rotated_frames, pixel_size_m, wavelength_m, z_m)
diffusers, PSFs = generate_diffusers_and_PSFs(padded_size,theta_deg,speckle_size_m,pixel_size_m, wavelength_m, M)
distorted_at_diffuser_plane = diffusers.to(device) * rotated_at_diffuser_plane

show_fields_gif(distorted_at_diffuser_plane, fps=30)

aligned_distorted_at_diffuser_plane = torch.zeros((M, padded_size, padded_size), dtype=torch.complex64, device=device)
reference_frame = torch.abs(distorted_at_diffuser_plane[0])
# reference_frame = fftshift(fft2(torch.abs(reference_frame)))
for idx in range(M):
    # rotation_angle = estimate_rotation(reference_frame, fftshift(fft2(torch.abs(distorted_at_diffuser_plane[idx]))), use_phase=False)

    rotation_angle = estimate_rotation_gpu(reference_frame, torch.abs(distorted_at_diffuser_plane[idx]), precision=0.5)

    print(rotation_angle)
    aligned_frame = rotate_field(distorted_at_diffuser_plane[idx], -rotation_angle, expand=False)
    aligned_distorted_at_diffuser_plane[idx] = aligned_frame
show_fields_gif(aligned_distorted_at_diffuser_plane, fps=30)


cropping_size = diagonal
cropped_aligned_distorted_at_diffuser_plane = center_crop(aligned_distorted_at_diffuser_plane, cropping_size)
show_fields_gif(cropped_aligned_distorted_at_diffuser_plane, fps=30)
# Prepare for CLASS reconstruction
T = torch.permute(cropped_aligned_distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T = T.to(device)

# Run CLASS algorithm
_, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)

# _, PSF_std, obj_fourier_angle, obj_fourier_abs = CTR_CLASS(T, num_iters, imsize=shp)
O_est_at_diff = torch.conj(phi_tot) * MTF
O_est_0 = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z_m)

z_values = np.linspace(0.5 * z_m, 2 * z_m, 100)
propagated_Os = torch.zeros((100, cropping_size, cropping_size), dtype=torch.complex64, device=device)

for idx, z_val in enumerate(z_values):
    propagated_Os[idx] = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z_val)

show_fields_gif(propagated_Os, 10)



O_est = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0).cpu().numpy()
O_est = O_est / np.abs(O_est).max()

new_camp = get_custom_colormap()

plt.figure()
idx = M//3
plt.subplot(251)
img = widefield.to(torch.complex64) if not torch.is_complex(widefield) else widefield
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'widefield - magnitude')

plt.subplot(256)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'widfeield - Phase')

plt.subplot(252)
img = diffusers[idx]
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap='gray', vmin=0, vmax=1)
plt.title(f'diffuser - magnitude')

plt.subplot(257)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'diffuser - Phase')

plt.subplot(253)
img = distorted_at_diffuser_plane[idx]
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at diff) - mag')

plt.subplot(258)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'distorted (at diff) - Phase')

plt.subplot(254)
img = angular_spectrum_gpu(distorted_at_diffuser_plane[idx], pixel_size_m,wavelength_m, -z_m)
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at obj) - mag')

plt.subplot(259)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'distorted (at obj) - Phase')

plt.subplot(255)
img = O_est
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'reconstructed - mag')

plt.subplot(2,5,10)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'reconstructed - Phase')

plt.show()

