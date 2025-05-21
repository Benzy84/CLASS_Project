import torch
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from torch.fft import *
from torchvision.transforms import CenterCrop, Resize
from matplotlib.colors import LinearSegmentedColormap


from core.CTRCLASS import CTR_CLASS
from utils.visualization import display_field
from utils.field_utils import generate_diffusers_and_PSFs, gauss2D
from utils.io import load_file_to_tensor
from utils.image_processing import fourier_convolution, shift_cross_correlation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'GPU activated: {torch.cuda.is_available()}')

# Define the custom colormap for visualizations
colors = [
    (0, 0, 0),  # Black
    (0, 0.2, 0),  # Dark Green
    (0, 0.5, 0),  # Green
    (0, 0.8, 0),  # Bright Green
    (0.7, 1, 0),  # Light Green-Yellow
    (1, 1, 1)  # White
]
positions = [0, 0.3, 0.5, 0.7, 0.9, 1]
new_cmap = LinearSegmentedColormap.from_list('greenish_hot', list(zip(positions, colors)))

# Helper functions
nrm = lambda x: x / x.abs().max()


def fraunhofer_pytorch_auto_pad(E0: torch.Tensor, dx: float, wavelength: float, z: float) -> torch.Tensor:
    """
    Fraunhofer propagation using automatic padding and cropping.
    Works with a single field (M, M) or batch (N, M, M), even if E0 is real-valued.

    Parameters:
    -----------
    E0 : torch.Tensor
        Input field(s), shape (M, M) or (N, M, M), can be real or complex
    dx : float
        Input pixel size [meters]
    wavelength : float
        Wavelength [meters]
    z : float
        Propagation distance [meters]

    Returns:
    --------
    torch.Tensor
        Output propagated field(s), same shape as input (real or complex as needed)
    """
    is_single = False
    device = E0.device
    dtype = E0.dtype
    comuting_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Clean up memory
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    if E0.ndim == 2:
        E0 = E0.unsqueeze(0)  # (1, M, M)
        is_single = True

    if not torch.is_complex(E0):
        E0 = E0.to(torch.complex64)

    N, M, _ = E0.shape

    # Calculate padded size to match output dx to input dx
    P_float = wavelength * z / dx ** 2
    P = int(np.ceil(P_float))
    P += P % 2  # make even

    pad_total = P - M
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    # Clean up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()

    # Pad the input
    E_padded = torch.nn.functional.pad(E0, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)

    del E0
    # Clean up memory again after padding
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()

    # Frequency grid
    fx = torch.fft.fftshift(torch.fft.fftfreq(P, d=dx)).to(device)
    FX, FY = torch.meshgrid(fx, fx, indexing='ij')

    # Phase terms
    quad_phase = torch.exp(1j * torch.pi * wavelength * z * (FX ** 2 + FY ** 2))
    phase_factor = torch.tensor(
        np.exp(1j * 2 * np.pi * z / wavelength),
        dtype=dtype,
        device=device
    ) * (dx ** 2) / (1j * wavelength * z)

    # Clean GPU memory before cropping and transferring
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Free the original large frequency grid tensors
    del fx, FX, FY
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    start = (P - M) // 2
    quad_phase = quad_phase[start:start + M, start:start + M].to(comuting_device)
    phase_factor = phase_factor.to(comuting_device)
    E_padded = E_padded.to(comuting_device)

    # FFT propagation
    E_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(E_padded, dim=(-2, -1))), dim=(-2, -1))

    # Free padded input immediately
    del E_padded
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    E_fft_cropped = E_fft[:, start:start + M, start:start + M]

    # Free the full FFT result
    del E_fft
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    E_far_cropped = E_fft_cropped * quad_phase * phase_factor

    # Free intermediate tensors
    del E_fft_cropped, quad_phase, phase_factor
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    result = E_far_cropped.to(device)

    # Free the GPU result after transferring to destination device
    del E_far_cropped
    if comuting_device.type == 'cuda':
        torch.cuda.empty_cache()

    return result[0] if is_single else result

def angular_spectrum_propagation(fields, wavelength, pixel_size, distance):
    """
    Propagates complex field(s) using the Angular Spectrum method.

    Args:
        fields (torch.Tensor): shape (H, W) or (N, H, W), complex64/complex128
        wavelength (float): wavelength in meters
        pixel_size (float): pixel size in meters
        distance (float): propagation distance in meters

    Returns:
        torch.Tensor: propagated field(s) of same shape
    """
    device = fields.device
    is_batched = (fields.ndim == 3)

    if not is_batched:
        fields = fields.unsqueeze(0)  # Add batch dimension

    N, H, W = fields.shape
    k = 2 * np.pi / wavelength  # Wavenumber

    # Frequency coordinates
    fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
    fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    FX, FY = FX.to(device), FY.to(device)

    # Spatial frequency squared
    f_squared = FX ** 2 + FY ** 2

    # Transfer function (H) for angular spectrum
    # Only propagate propagating waves (real sqrt); evanescent waves are discarded
    argument = 1 - (wavelength ** 2) * f_squared
    argument[argument < 0] = 0  # Remove evanescent components
    H = torch.exp(1j * k * distance * torch.sqrt(argument))

    # Apply propagation
    field_fft = fft2(fields)
    out_fft = field_fft * H
    propagated = ifft2(out_fft)

    if not is_batched:
        propagated = propagated.squeeze(0)  # Remove batch dimension

    return propagated


def makedir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# Create timestamped output directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"spot_size_simulation_{timestamp}_incoherent"
makedir(results_dir)

# Simulation parameters
sz = 800  # Image size in pixels
N_obj = sz * 3 // 8  # Object size
C = CenterCrop(int(1.5 * N_obj))  # Cropping transform
C2 = CenterCrop(N_obj + 10)  # Smaller crop for visualization

# Physical parameters
pixel_size = 5  # Pixel size in microns
speckle_size = 20  # Speckle size in microns
theta = 0.5  # Angular spread in degrees
wavelength = 0.6328  # Wavelength in microns (632.8 nm)
M = 180  # Number of diffuser patterns

# Convert to SI units
pixel_size_m = pixel_size * 1e-6
speckle_size_m = speckle_size * 1e-6
wavelength_m = wavelength * 1e-6

# Calculate correlation distance from angular spread
theta_rad = torch.deg2rad(torch.tensor(theta))
d_corr = wavelength_m / theta_rad

# Ratio between spot size and correlation distance to test
ratios = [100]#[0.1, 0.5, 1, 2, 5, 10, 25, 50]
spot_sizes = [ratio * d_corr for ratio in ratios]

# Create README file
with open(os.path.join(results_dir, "README.txt"), "w") as f:
    f.write(f"Spot Size Simulation Results\n")
    f.write(f"=========================\n\n")
    f.write(f"Simulation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"This simulation investigates how varying the illumination spot size relative\n")
    f.write(f"to the diffuser correlation distance affects imaging through scattering media.\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  Image size: {sz}x{sz} pixels\n")
    f.write(f"  Object size: {N_obj}x{N_obj} pixels\n")
    f.write(f"  Pixel size: {pixel_size} microns\n")
    f.write(f"  Speckle size: {speckle_size} microns\n")
    f.write(f"  Angular spread (theta): {theta} degrees\n")
    f.write(f"  Wavelength: {wavelength} microns\n")
    f.write(f"  Correlation distance: {d_corr * 1e6:.2f} microns\n\n")
    f.write(f"Tested spot_size/correlation_distance ratios: {ratios}\n\n")

    f.write(f"File Structure:\n")
    f.write(f"  parameters.npy: Simulation parameters\n")
    f.write(f"  ground_truth.npy: Original object\n")
    f.write(f"  widefield.npy: Widefield reference image\n")
    f.write(f"  For each ratio (e.g., 0.1, 0.5, 1, etc.):\n")
    f.write(f"    illumination_[ratio].npy: Illumination pattern\n")
    f.write(f"    illumination_at_diffuser_[ratio].npy: Illumination after diffuser\n")
    f.write(f"    illumination_at_object_[ratio].npy: Illumination at object plane\n")
    f.write(f"    effective_object_[ratio].npy: Object with illumination pattern\n")
    f.write(f"    distorted_object_[ratio].npy: Distorted effective object\n")
    f.write(f"    reconstruction_[ratio].npy: Reconstructed object\n")

# Generate diffusers and PSFs
diffusers, PSFs = generate_diffusers_and_PSFs(sz, theta, speckle_size_m, pixel_size_m, wavelength_m,
                                              num_diffusers=M)
_, clean_PSF = generate_diffusers_and_PSFs(sz, 0, speckle_size_m, pixel_size_m, wavelength_m,
                                           num_diffusers=1)

# Load and prepare object
P = load_file_to_tensor().cpu()
resize = Resize((N_obj, N_obj), antialias=True)
P = resize(P.unsqueeze(0)).squeeze(0)
padding_size = (sz - N_obj) // 2
P = torch.nn.functional.pad(P, 4 * [padding_size])

# Generate reference widefield image
widefield = fourier_convolution(P / P.sum(), clean_PSF ** 2)
widefield = widefield.abs().to(device)

# Save ground truth and widefield
np.save(os.path.join(results_dir, "ground_truth.npy"), P.cpu().numpy())
np.save(os.path.join(results_dir, "widefield.npy"), widefield.cpu().numpy())
np.save(os.path.join(results_dir, "diffuser.npy"), diffusers[0].cpu().numpy())

# Save parameters
parameters = {
    'sz': sz,
    'N_obj': N_obj,
    'pixel_size': pixel_size,
    'speckle_size': speckle_size,
    'theta': theta,
    'wavelength': wavelength,
    'd_corr': d_corr.item(),
    'ratios': ratios,
    'spot_sizes': [s.item() for s in spot_sizes]
}
np.save(os.path.join(results_dir, "parameters.npy"), parameters)

# Initialize dictionaries to store results
all_illuminations = {}
all_ill_at_diff = {}
all_ill_at_obj = {}
all_eff_obj = {}
all_distorted_eff_obj = {}
all_recons = {}

# Process each ratio
for i, spot_size in enumerate(spot_sizes):
    ratio = ratios[i]
    print(f'Processing spot_size/d_corr = {ratio}: {i + 1}/{len(ratios)}')

    # Calculate spot size in pixels
    spot_size_in_pixels = int(spot_size / pixel_size_m)

    # Create illumination pattern
    illumination = gauss2D(spot_size_in_pixels, sz)
    illumination_through_diffuser = illumination * diffusers

    # Propagate to object plane
    illumination_at_object_plane = fraunhofer_pytorch_auto_pad(
        illumination_through_diffuser, pixel_size_m, wavelength_m, 10e-2)

    # Create effective object (object * illumination)
    eff_obj = illumination_at_object_plane * P

    # Distort through the diffuser
    distorted_eff_obj = fourier_convolution(eff_obj.abs() ** 2, PSFs.abs() ** 2)
    tensor_with_dims = distorted_eff_obj.unsqueeze(1)
    cropped_tensor = C(tensor_with_dims)
    cropped_distorted_obj = cropped_tensor.squeeze(1)


    # Prepare for CLASS reconstruction
    shp = cropped_distorted_obj.shape[-1] * cropped_distorted_obj.shape[-2]

    Icam_fft = torch.fft.fftshift(torch.fft.fft2(cropped_distorted_obj - cropped_distorted_obj.mean(0)))
    T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).to(device)
    T = T.to(device)

    # Run CLASS algorithm
    _, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)
    O_raw = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
    O_est = shift_cross_correlation(C(widefield), O_raw.real)
    O_est /= torch.abs(O_est).max()

    # Store results
    all_illuminations[f'spot_size/d_corr = {ratio}'] = illumination
    all_ill_at_diff[f'spot_size/d_corr = {ratio}'] = illumination_through_diffuser[0]
    all_ill_at_obj[f'spot_size/d_corr = {ratio}'] = illumination_at_object_plane[0]
    all_eff_obj[f'spot_size/d_corr = {ratio}'] = eff_obj[0]
    all_distorted_eff_obj[f'spot_size/d_corr = {ratio}'] = distorted_eff_obj[0]
    all_recons[f'spot_size/d_corr = {ratio}'] = O_est


# Save all compiled results
np.save(os.path.join(results_dir, "all_illuminations.npy"),
        {k: v.cpu().numpy() for k, v in all_illuminations.items()})
np.save(os.path.join(results_dir, "all_illuminations_at_diffuser.npy"),
        {k: v.cpu().numpy() for k, v in all_ill_at_diff.items()})
np.save(os.path.join(results_dir, "all_illuminations_at_object.npy"),
        {k: v.cpu().numpy() for k, v in all_ill_at_obj.items()})
np.save(os.path.join(results_dir, "all_effective_objects.npy"),
        {k: v.cpu().numpy() for k, v in all_eff_obj.items()})
np.save(os.path.join(results_dir, "all_distorted_objects.npy"),
        {k: v.cpu().numpy() for k, v in all_distorted_eff_obj.items()})
np.save(os.path.join(results_dir, "all_reconstructions.npy"),
        all_recons)



print(f"Simulation complete! Results saved to {results_dir}")