import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import datetime
from torch.fft import *
from core.CTRCLASS import CTR_CLASS
from utils.field_utils import gauss2D, circ, generate_diffusers_and_PSFs
from utils.image_processing import shift_cross_correlation, fourier_convolution
from utils.io import load_file_to_tensor
from utils.visualization import display_field

from torchvision.transforms import CenterCrop, Resize
from matplotlib.colors import LinearSegmentedColormap
from diffractsim import MonochromaticField, mm, cm, um, set_backend


from waveprop.rs import angular_spectrum

from tqdm import tqdm

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'GPU activated: {torch.cuda.is_available()}')


# Define custom colormap for visualization
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
        field_numpy = field_tensor.cpu().detach().numpy()

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


def propagate_field2(field_tensor, pixel_size, wavelength, distance, method='AS'):
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

            result_tensor, _, _ = angular_spectrum(u_in=single_field, wv=wavelength, d1=pixel_size, dz=distance, device=device)
            results.append(result_tensor)

        # Stack results to return a batch
        return torch.stack(results)

    else:
        # Original single-field behavior

        result_tensor = angular_spectrum(u_in=field_tensor, wv=wavelength, d1=pixel_size, dz=distance, device=device)
        return result_tensor

# Utility functions
def makedir(dr):
    if not os.path.isdir(dr):
        os.makedirs(dr)


nrm = lambda x: x / x.abs().max()


# Use a loaded image instead of random points?
use_loaded_image = True  # Set to False to use random points

# Create results directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"coherent_incoherent_compounding_{timestamp}"
makedir(results_dir)


# =====================================
# Simulation parameters
# =====================================
# Image size parameters
sz = 600  # Image size in pixels
N_obj = sz * 1 // 6  # Object size
C = CenterCrop(int(1.5 * N_obj))  # Cropping transform
C2 = CenterCrop(N_obj + 10)  # Smaller crop for visualization

# Physical parameters
pixel_size = 5  # Pixel size in microns
speckle_size = 20  # Speckle size in microns
theta = 0.5  # Angular spread in degrees
wavelength = 0.6328  # Wavelength in microns (632.8 nm)
M = 150  # Number of diffuser patterns

# Convert to SI units
pixel_size_m = pixel_size * 1e-6
speckle_size_m = speckle_size * 1e-6
wavelength_m = wavelength * 1e-6

# Calculate correlation distance from angular spread
theta_rad = torch.deg2rad(torch.tensor(theta))
d_corr = wavelength_m / theta_rad

# Spot size parameters
spot_size_to_d_corr_ratio = 100  # Ratio of spot size to correlation distance
spot_size = spot_size_to_d_corr_ratio * d_corr
spot_size_in_pixels = spot_size // pixel_size_m


# Simulation parameters
K_values = [400]#,2,3,4,8,16,32,100]  # Number of speckle illuminations per realization  # Number of speckle illuminations per realization


# Create README file
with open(os.path.join(results_dir, "README.txt"), "w") as f:
    f.write(f"Coherent Illumination with Incoherent Compounding Simulation\n")
    f.write(f"======================================================\n\n")
    f.write(f"Simulation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"This simulation demonstrates the use of incoherent compounding of multiple\n")
    f.write(f"coherent illuminations to achieve more uniform illumination while maintaining\n")
    f.write(f"coherence-gating capability.\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  Image size: {sz}x{sz} pixels\n")
    f.write(f"  Pixel size: {pixel_size} microns\n")
    f.write(f"  Speckle size: {speckle_size} microns\n")
    f.write(f"  Angular spread (theta): {theta} degrees\n")
    f.write(f"  Wavelength: {wavelength} microns\n")
    f.write(f"  Correlation distance: {d_corr} pixels\n")
    f.write(f"  Number of medium realizations (M): {M}\n")
    f.write(f"  Using loaded image: {use_loaded_image}\n")
    f.write(f"  K values (speckle patterns per realization): {K_values}\n")

    f.write(f"File Structure:\n")
    f.write(f"  ground_truth.npy: Original object\n")
    f.write(f"  widefield.npy: Widefield reference image\n")
    f.write(f"  For each trial and K value:\n")
    f.write(f"    diffuser_trial_T.npy: One example diffuser pattern\n")
    f.write(f"    speckle_patterns_trial_T_K_X.npy: One example set of speckle illuminations\n")
    f.write(f"    macro_frame_trial_T_K_X.npy: Example macro-frame\n")
    f.write(f"    reconstructions/reconstruction_trial_T_K_X.npy: Reconstructed object\n")

# Save parameter arrays
np.save(os.path.join(results_dir, "K_values.npy"), np.array(K_values))
np.save(os.path.join(results_dir, "parameters.npy"), {
    'sz': sz,
    'pixel_size': pixel_size,
    'speckle_size': speckle_size,
    'theta': theta,
    'wavelength': wavelength,
    'd_corr': d_corr,
    'M': M,
    'K_values': K_values,
    'use_loaded_image': use_loaded_image
})


# Generate diffusers for M different medium realizations
diffusers, PSFs = generate_diffusers_and_PSFs(sz, theta, 3 * speckle_size_m, pixel_size_m, wavelength_m,
                                              num_diffusers=M)

# Create clean PSF for widefield reference
_, clean_PSF = generate_diffusers_and_PSFs(sz, 0, speckle_size_m, pixel_size_m, wavelength_m, 1)
clean_PSF /= clean_PSF.abs().sum() ** 0.5
clean_PSF = clean_PSF.to(device)



# =====================================
# Main simulation
# =====================================


# Create ground truth object - either load image or create sparse points
# Load image from file
gt = load_file_to_tensor()
# Resize to desired dimensions with padding
resize = Resize((N_obj,N_obj), antialias=True)
gt_resized = resize(gt.unsqueeze(0)).squeeze(0)  # Add/remove batch dimension for torchvision
padding_size = (sz - N_obj) // 2
gt = torch.nn.functional.pad(gt_resized, 4 * [padding_size])

# Normalize and prepare for processing
gt_intensity = gt / gt.sum()
gt_amplitude = torch.sqrt(gt_intensity)
gt_amplitude = gt_amplitude.to(device)

# Generate widefield reference image
widefield = fourier_convolution(gt_intensity, clean_PSF ** 2)
widefield = widefield.abs().to(device)

# Save ground truth and widefield for this trial
np.save(os.path.join(results_dir, "ground_truth.npy"), nrm(gt_intensity).cpu().numpy())
np.save(os.path.join(results_dir, "widefield.npy"), nrm(widefield).cpu().numpy())
np.save(os.path.join(results_dir, "diffuser_example.npy"), nrm(diffusers[0]).cpu().numpy())
illumination = gauss2D(spot_size_in_pixels, sz).to(device)

distorted_objects = fourier_convolution(gt_intensity, PSFs ** 2)
distorted_objects_with_dims = distorted_objects.unsqueeze(1)
cropped_tensor = distorted_objects_with_dims
cropped_distorted_obj = cropped_tensor.squeeze(1)
cropped_distorted_obj = nrm(cropped_distorted_obj)

shp = cropped_distorted_obj.shape[-1] * cropped_distorted_obj.shape[-2]

# Add modulation
rM = 1 + (torch.arange(0, M, device='cuda') / M)
modulated_frames = cropped_distorted_obj # * rM[:, None, None]

# Prepare for CLASS reconstruction
Icam_fft = torch.fft.fftshift(torch.fft.fft2(cropped_distorted_obj - cropped_distorted_obj.mean(0)))
Icam_fft_modulated = torch.fft.fftshift(torch.fft.fft2(modulated_frames - modulated_frames.mean(0)))

# Prepare for CLASS reconstruction
T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).to(device)
T_modulated = torch.permute(Icam_fft_modulated, [2, 1, 0]).reshape(shp, -1).to(device)
T = T.to(device)
T_modulated = T_modulated.to(device)
#
# # Run CLASS algorithm
# _, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)
# _, _, phi_tot_modulated, MTF_modulated = CTR_CLASS(T_modulated, num_iters=1000)
# # Raw reconstruction
# O_raw = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
# O_raw_modulated = ifft2(ifftshift(torch.conj(phi_tot_modulated) * MTF_modulated))
# O_est_uniform = shift_cross_correlation(widefield, O_raw.real)
# O_est_uniform_modulated = shift_cross_correlation(widefield, O_raw_modulated.real)
# O_est_uniform /= torch.abs(O_est_uniform).max()
# O_est_uniform_modulated /= torch.abs(O_est_uniform_modulated).max()
#
# # Save the uniform reconstruction
# np.save(os.path.join(results_dir, "uniform_reconstruction.npy"), nrm(O_est_uniform).cpu().numpy())
# np.save(os.path.join(results_dir, "uniform_with_modulation_reconstruction.npy"), nrm(O_est_uniform_modulated).cpu().numpy())


total_operations = sum(K_values) * M
operation_count = 0
print(f"Total operations to perform: {total_operations}")
print(f"- {len(K_values)} CLASS reconstructions")

speckle_patterns = []
illuminations = []
ill_diffusers = []
F = MonochromaticField(
    wavelength=wavelength_m,  # Your wavelength in meters
    extent_x=sz * pixel_size_m,  # Physical width in meters
    extent_y=sz * pixel_size_m,  # Physical height in meters
    Nx=sz,  # Number of pixels in x
    Ny=sz  # Number of pixels in y
)
# Process each K value
for k_idx, K in enumerate(K_values):
    print(f"\nProcessing K={K} ({k_idx+1}/{len(K_values)})")

    # Storage for all macro-frames
    macro_fields = []
    macro_eff_objects = []
    macro_distorted_fields_at_diffuser = []
    macro_distorted_fields_at_obj_plan = []
    macro_frames = []
    macro_frames_2 = []

    # For each medium realization, create K different speckle illuminations
    for m in range(M):
        current_psf = PSFs[m]
        current_diffuser = diffusers[m]
        # Generate K speckle patterns for this realization
        Fields_at_obj = []
        Effective_obj_fields = []
        Distorted_captured_obj_fields = []
        Distorted_obj_fields_at_obj_plane = []
        Distorted_obj_intensities_at_obj_plane = []
        Distorted_obj_intensities_at_obj_plane_2 = []

        # Create K different speckle illuminations
        batch_illuminations = torch.ones((K, sz, sz), dtype=torch.complex64, device=device)

        # If you need different patterns for each k:
        x = torch.linspace(-1, 1, sz, dtype=torch.float32, device=device)
        y = torch.linspace(-1, 1, sz, dtype=torch.float32, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        Z = X + 1j * Y  # Complex plane

        # Create K different phase patterns (vectorized)
        k_values = torch.arange(1, K + 1, device=device).view(-1, 1, 1)
        batch_illuminations = torch.exp(1j * torch.angle(Z) * k_values)
        batch_illuminations = batch_illuminations / batch_illuminations.abs().max(dim=1, keepdim=True).values.max(dim=2,
                                                                                                                  keepdim=True).values

        # Generate K random diffusers at once, or reuse one diffuser
        batch_diffusers = torch.ones_like(batch_illuminations)
        for k in range(K):
            ill_diffuser, _ = generate_diffusers_and_PSFs(sz, 10, 0.1 * speckle_size_m, pixel_size_m, wavelength_m, 1)
            batch_diffusers[k] = ill_diffuser.to(device)

        # Apply diffusers to illuminations (all at once)
        batch_at_diffuser = batch_illuminations * batch_diffusers

        # Propagate all fields at once (using the batch version of propagation)
        st = time.time()
        batch_at_object = propagate_field2(batch_at_diffuser, pixel_size_m, wavelength_m, 0.02)
        print("wavepropagation time1:", time.time() - st)
        st = time.time()
        batch_at_object2 = propagate_field(batch_at_diffuser, pixel_size_m, wavelength_m, 0.02)
        print("wavepropagation time2:", time.time() - st)

        # Apply all fields to object at once
        batch_eff_obj = gt_amplitude.to(device) * batch_at_object

        # Propagate back to diffuser plane (all at once)
        batch_eff_at_diffuser = propagate_field(batch_eff_obj, pixel_size_m, wavelength_m, -0.02)

        # Apply diffuser to distort (all at once)
        batch_distorted = batch_eff_at_diffuser * current_diffuser.to(device)

        # Propagate to object plane (all at once)
        batch_distorted_at_obj = propagate_field(batch_distorted, pixel_size_m, wavelength_m, 0.02)

        # Calculate intensities (all at once)
        batch_intensities = batch_distorted_at_obj.abs() ** 2

        # Store results
        Fields_at_obj = batch_at_object
        Effective_obj_fields = batch_eff_obj
        Distorted_captured_obj_fields = batch_distorted
        Distorted_obj_fields_at_obj_plane = batch_distorted_at_obj
        Distorted_obj_intensities_at_obj_plane = batch_intensities
        # plt.figure(), plt.imshow(torch.abs(nrm(AS.cpu())), cmap=new_cmap, vmin=0, vmax=1) ,plt.colorbar(), plt.axis('off')


        # Incoherently sum the K intensity patterns to create a macro-frame
        macro_field = torch.sum(Fields_at_obj, dim=0)
        macro_eff_obj = torch.sum(Effective_obj_fields, dim=0)
        macro_distorted_field_at_diffuser = torch.sum(Distorted_captured_obj_fields, dim=0)
        macro_distorted_field_at_obj_plan = torch.sum(Distorted_obj_fields_at_obj_plane, dim=0)
        macro_frame = torch.sum(Distorted_obj_intensities_at_obj_plane, dim=0)

        # Store results for this medium realization
        macro_fields.append(nrm(macro_field))
        macro_eff_objects.append(nrm(macro_eff_obj))
        macro_distorted_fields_at_diffuser.append(nrm(macro_distorted_field_at_diffuser))
        macro_distorted_fields_at_obj_plan.append(nrm(macro_distorted_field_at_obj_plan))
        macro_frames.append(nrm(macro_frame))


    rM = 1 + (torch.arange(0, M, device='cuda') / M)
    # Stack all macro-frames for this K value
    all_frames = torch.stack(macro_distorted_fields_at_diffuser)
    all_intenites = torch.stack(macro_frames)
    all_intenites_2 = torch.stack(macro_frames)
    # Add modulation
    modulated_frames = all_intenites * rM[:, None, None]
    modulated_frames_2 = all_intenites_2# * rM[:, None, None]

    # Prepare for CLASS reconstruction
    Icam_fft = all_frames# torch.fft.fftshift(torch.fft.fft2(all_frames - all_frames.mean(0)))
    Icam_fft_modulated = torch.fft.fftshift(torch.fft.fft2(modulated_frames- modulated_frames.mean(0)))
    Icam_fft_modulated_2 = torch.fft.fftshift(torch.fft.fft2(modulated_frames_2- modulated_frames_2.mean(0)))
    T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).to(device)
    T_modulated = torch.permute(Icam_fft_modulated, [2, 1, 0]).reshape(shp, -1).to(device)
    T_modulated_2 = torch.permute(Icam_fft_modulated_2, [2, 1, 0]).reshape(shp, -1).to(device)

    # Run I-CLASS algorithm
    print("")
    print(f"Running I-CLASS for K={K}...")
    _, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)
    _, _, phi_tot_modulated, MTF_modulated = CTR_CLASS(T_modulated, num_iters=1000)
    _, _, phi_tot_modulated_2, MTF_modulated_2 = CTR_CLASS(T_modulated_2, num_iters=1000)

    # Raw reconstruction
    O_row = torch.conj(phi_tot) * MTF
    O_est = propagate_field(O_row, wavelength_m, pixel_size_m, 0.025)
    O_est = ifftshift(ifft2(O_row))
    display_field(O_est)


    O_raw_modulated = ifft2(ifftshift(torch.conj(phi_tot_modulated) * MTF_modulated))
    O_est_modulated = shift_cross_correlation(widefield, O_raw_modulated.real)
    O_est_modulated /= torch.abs(O_est_modulated).max()
    display_field(O_est_modulated)


    O_raw_modulated_2 = ifft2(ifftshift(torch.conj(phi_tot_modulated_2) * MTF_modulated_2))
    O_est_modulated_2 = shift_cross_correlation(widefield, O_raw_modulated_2.real)
    O_est_modulated_2 /= torch.abs(O_est_modulated_2).max()
    display_field(O_est_modulated_2)




    np.save(os.path.join(results_dir, f"O_est_with_{K}_micro_frames.npy"), nrm(O_est).cpu().numpy())
    np.save(os.path.join(results_dir, f"O_est_with_modulation_with_{K}_micro_frames.npy"), nrm(O_est_modulated).cpu().numpy())

    # convert macro_intensities to np.array
    np.save(os.path.join(results_dir, f"macro_intensities_with_{K}_micro_frames.npy"), torch.stack(macro_intensities).cpu().numpy())
    np.save(os.path.join(results_dir, f"macro_objects_fields_with_{K}_micro_frames.npy"), torch.stack(macro_obj_fields).cpu().numpy())
    np.save(os.path.join(results_dir, f"macro_objects_intesities_with_{K}_micro_frames.npy"), torch.stack(macro_obj_intensities).cpu().numpy())
    np.save(os.path.join(results_dir, f"macro_frames_with_{K}_micro_frames.npy"), torch.stack(macro_frames).cpu().numpy())

    # Save the reconstruction

    # Clear GPU memory
    torch.cuda.empty_cache()

np.save(os.path.join(results_dir, "illuminations.npy"), np.array(illuminations))
np.save(os.path.join(results_dir, "ill_diffusers.npy"), np.array(ill_diffusers))
np.save(os.path.join(results_dir, "speckle_patterns.npy"), np.array(speckle_patterns))