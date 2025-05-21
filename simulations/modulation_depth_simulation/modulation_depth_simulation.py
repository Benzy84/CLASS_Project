import matplotlib.pyplot as plt

is_real_data = True
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import torch
import numpy as np
import os
import datetime
from torch.fft import *
from torchvision.transforms import CenterCrop
from torchvision.transforms import Resize
from matplotlib.colors import LinearSegmentedColormap

from core.CTRCLASS import CTR_CLASS
from simulations.simulation_utils import get_modulation_strategies
from utils.io import create_timestamped_dir, load_mat_file, load_file_to_tensor
from utils.field_utils import gauss2D, generate_diffusers_and_PSFs
from utils.image_processing import shift_cross_correlation, fourier_convolution


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

# =====================================
# Utility functions
# =====================================
# Create directory for results
def makedir(dr):
    if not os.path.isdir(dr):
        os.mkdir(dr)


# Utility for normalization
nrm = lambda x: x / x.abs().max()


# =====================================
# Simulation parameters - MODIFY THESE
# =====================================
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulation size parameters
sz = 200  # Image size in pixels
center_crop = CenterCrop(sz)
pixel_size = 5  # Pixel size in microns
speckle_size = 15  # Speckle size in microns
theta = 1 # Angular spread in degrees
wavelength = 0.6328  # Wavelength in microns (632.8 nm)

# Simulation experiment parameters
M = 100  # Number of realizations
n = 500  # Number of points
num_trials = 1  # Number of independent trials
min_weight = 0 #0.01  # Minimum allowed weight to prevent zeros

# Alpha values from 0 to 1
depth_values = torch.linspace(0, 0.97, 3)
# depth_values[3] = 1/3
alpha_values = - (depth_values + 1) / (depth_values - 1)
alpha_values = alpha_values.tolist()
# print(alpha_values)
# alpha_values = torch.linspace(10, 60, 10).tolist()
# alpha_values = [1, 1.2, 1.5, 1.75, 2, 4, 8, 16, 24, 35, 40, 48, 56, 64]
alpha_values = [round(value, 3) for  value in alpha_values]
# alpha_values = alpha_values[::4]
# alpha_values = [0, 0.25, 0.5, 0.75, 1.0]
# alpha_values = [65.6667]


# Get all available modulation strategies
all_modulation_strategies = get_modulation_strategies(device, min_weight)

# Select which strategies to use in this simulation
# selected_strategy_names = ["none", "linear", "increasing_ramp", "sinusoidal", "logarithmic"]
selected_strategy_names = ["none", "linear", "increasing_ramp", "linear_2", "sinusoidal", "logarithmic", "quadratic", "bell_curve", "step", "linear_ramp", "oscillating", "exponential" ]
selected_strategy_names = ["ratio_ramp"]

# Create a filtered dictionary with only the selected strategies
modulation_strategies = {name: all_modulation_strategies[name] for name in selected_strategy_names}


# =====================================
# Main execution code - DO NOT MODIFY
# =====================================
# GPU check
print(f'Using device: {device}')
print(f'GPU activated: {torch.cuda.is_available()}')

# Create timestamped output directory
output_dir = create_timestamped_dir('modulation_depth_results')


# Create clean PSF for widefield reference (common for all trials)
# Convert speckle size from microns to pixels
speckle_size_px = sz // (2 * speckle_size / pixel_size)
clean_PSF = fftshift(fft2(gauss2D(speckle_size_px, sz)).abs() ** 2)
clean_PSF /= clean_PSF.sum()
clean_PSF = clean_PSF.to(device)

resize = Resize((64, 64), antialias=True)

if is_real_data:
    # For real data processing
    # For real data processing
    print("Please select widefield reference file:")
    _, widefield = load_mat_file(purpose="Widefield Reference")

    print("Please select raw data file:")
    _, I = load_mat_file(purpose="Raw Experimental Data")

    # Move data to device
    widefield = widefield.to(device)
    widefield = widefield.T  # Transpose if needed
    I = I.to(device)

    # Process widefield (2D tensor)
    widefield_fft = fftshift(fft2(widefield))
    tensor_with_dims = widefield_fft.unsqueeze(0)  # Add a channel dimension
    cropped_tensor = center_crop(tensor_with_dims)
    cropped_widefield_fft = cropped_tensor.squeeze(0)  # Remove the channel dimension
    widefield = ifft2(ifftshift(cropped_widefield_fft)).abs()

    # Process I (3D tensor with shape [num_fields, H, W])
    I_fft = fftshift(fft2(I))  # [num_fields, H, W]
    tensor_with_dims = I_fft.unsqueeze(1)  # [num_fields, 1, H, W]
    cropped_tensor = center_crop(tensor_with_dims)
    cropped_I_fft = cropped_tensor.squeeze(1)  # [num_fields, crop_size, crop_size]
    I = ifft2(ifftshift(cropped_I_fft)).abs()

    M = I.shape[0] if len(I.shape) > 2 else 1  # Number of realizations
    num_trials = 1  # Only one trial for real data

    # Save widefield (no ground truth for real data)
    np.save(f'{output_dir}/widefield.npy', widefield.cpu().numpy())
else:

    # Create ground truth object
    # Choose one of the following options:

    # Option 2: Load image from file (uncomment to use)
    gt = load_file_to_tensor()
    gt = resize(gt.unsqueeze(0)).squeeze(0)  # Add/remove batch dimension for torchvision
    #
    # gt = getObject(int(64), sz, 'Two Points')

    gt = torch.nn.functional.pad(gt, 4 * [int((200-64)/2)])
    gt_Intensity = gt / gt.sum()
    gt_Intensity = gt_Intensity.to(device)

    # Reference widefield image (also created ONCE)
    widefield = fourier_convolution(gt_Intensity, clean_PSF).abs()

    # Save ground truth and widefield ONCE before the trials
    np.save(f'{output_dir}/widefield.npy', widefield.cpu().numpy())
    np.save(f'{output_dir}/gt.npy', gt_Intensity.cpu().numpy())

# Create a human-readable README file
# Create a human-readable README file
with open(f'{output_dir}/README.txt', 'w') as f:
    f.write(f"CLASS Modulation Depth Simulation\n")
    f.write(f"==============================\n\n")

    if is_real_data:
        f.write(f"REAL DATA PROCESSING\n")
        f.write(f"-------------------\n")
        f.write(f"This analysis uses experimental measurements.\n")
        f.write(f"Ground truth object is not available.\n")
        f.write(f"PSFs are theoretical approximations.\n\n")

    f.write(f"Parameters:\n")
    f.write(f"---------------------\n")
    f.write(f"Image size: {sz} pixels\n")
    f.write(f"Pixel size: {pixel_size} microns\n")
    f.write(f"Speckle size: {speckle_size} microns\n")
    f.write(f"Angular spread (theta): {theta} degree\n")
    f.write(f"Number of realizations (M): {M}\n")

    if not is_real_data:
        f.write(f"Number of points (n): {n}\n")

    f.write(f"Number of trials: {num_trials}\n")
    f.write(f"Minimum weight: {min_weight}\n")
    f.write(f"Wavelength: {wavelength} microns\n\n")

    f.write(f"Modulation Strategies: {', '.join(modulation_strategies.keys())}\n")
    f.write(f"Alpha values: {alpha_values}\n\n")

    f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    f.write(f"\nFile Structure:\n")
    f.write(f"--------------\n")
    f.write(f"widefield_trial_N.npy: Reference widefield image for trial N\n")

    if not is_real_data:
        f.write(f"gt_trial_N.npy: Ground truth object for trial N\n")

    f.write(f"PSF_trial_N.npy: PSF for trial N\n")
    f.write(f"aberrated_image_trial_N_sample_X.npy: Raw aberrated images\n")
    f.write(f"modulation_strategy_alpha.npy: Modulation pattern for given strategy and alpha\n")
    f.write(f"O_raw_strategy_alpha_trial_N.npy: Raw reconstructed object from CLASS algorithm\n")
    f.write(f"phi_tot_strategy_alpha_trial_N.npy: Phase mask from CLASS algorithm\n")
    f.write(f"MTF_strategy_alpha_trial_N.npy: Modulation Transfer Function from CLASS\n")

    # Add parameter format section to make parsing easier
    f.write(f"\nParameter Format (for parsing):\n")
    f.write(f"----------------------------\n")
    f.write(f"PARAM:sz={sz}\n")
    f.write(f"PARAM:pixel_size={pixel_size}\n")
    f.write(f"PARAM:speckle_size={speckle_size}\n")
    f.write(f"PARAM:theta={theta}\n")
    f.write(f"PARAM:M={M}\n")

    if not is_real_data:
        f.write(f"PARAM:n={n}\n")

    f.write(f"PARAM:num_trials={num_trials}\n")
    f.write(f"PARAM:min_weight={min_weight}\n")
    f.write(f"PARAM:wavelength={wavelength}\n")
    f.write(f"PARAM:strategies={','.join(modulation_strategies.keys())}\n")
    f.write(f"PARAM:alpha_values={','.join(map(str, alpha_values))}\n")
    f.write(f"PARAM:is_real_data={'True' if is_real_data else 'False'}\n")

# Calculate total iterations for progress tracking
total_iterations = len(modulation_strategies) * len(alpha_values) * num_trials
completed_iterations = 0

# Main simulation loop with trials
for trial in range(num_trials):
    print(f"\nStarting Trial {trial + 1}/{num_trials}")

    # Generate new diffuser patterns for each trial
    # Convert microns to meters for the physics calculations
    pixel_size_m = pixel_size * 1e-6
    speckle_size_m = speckle_size * 1e-6
    wavelength_m = wavelength * 1e-6

    diffusers, PSFs = generate_diffusers_and_PSFs(sz, theta, speckle_size_m, pixel_size_m, wavelength_m,
                                                  num_diffusers=M)

    PSFs = PSFs.abs() ** 2
    PSFs = PSFs.to(device)

    psf_sums = PSFs.sum(dim=(-2, -1), keepdim=True)  # shape will be (N,1,1)
    # Avoid division by zero if any sum is 0 (unlikely, but just in case)
    psf_sums = torch.clamp(psf_sums, min=1e-20)
    # Normalize each PSF slice so it sums to 1
    PSFs_normalized = PSFs / psf_sums
    PSFs_normalized = PSFs_normalized.to(device)


    # Save PSF reference
    np.save(f'{output_dir}/PSF_trial_{trial + 1}.npy', PSFs[0].cpu().numpy())

    # Generate images through diffusers
    if not is_real_data:
        I = fourier_convolution(gt_Intensity, PSFs_normalized).abs()

    # Save aberrated image example for reference
    np.save(f'{output_dir}/aberrated_image_trial_{trial + 1}.npy', I[0].cpu().numpy())


    # Process different strategies and alpha values
    for strategy_name, strategy_func in modulation_strategies.items():
        # plt.figure()
        for a_idx, alpha in enumerate(alpha_values):
            # Update progress counter
            completed_iterations += 1
            print(f"Progress: {completed_iterations}/{total_iterations} iterations completed")
            print(f"Trial: {trial + 1}, Strategy: {strategy_name}, Alpha: {alpha:.2f}")

            # Apply modulation
            modulation = strategy_func(M, alpha)

            # Verify no negative or zero weights
            min_mod = torch.min(modulation).item()
            if min_mod <= 0:
                print(f"WARNING: {strategy_name} with alpha={alpha} has minimum weight {min_mod}")
                modulation = torch.clamp(modulation, min=min_weight)

            # Save modulation pattern (only once per strategy/alpha)
            if trial == 0:
                np.save(f'{output_dir}/modulation_{strategy_name}_alpha_{alpha:.2f}.npy', modulation.cpu().numpy())

            # Multiply each field by its corresponding weight
            I_modulated = I * modulation[:, None, None]
            I_modulated_mean_reduced = I_modulated - I_modulated.mean(0)

            # Compute Fourier transform of the images
            Icam_fft = torch.fft.fftshift(torch.fft.fft2(I_modulated_mean_reduced))
            shp = I.shape[-1] * I.shape[-2]
            T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).to(device)


            # Run CLASS algorithm
            _, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)

            # Raw reconstruction
            O_raw = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
            O_est = shift_cross_correlation(widefield, O_raw.real)
            O_est /= torch.abs(O_est).max()

            # img = O_est
            # img -= torch.min(img)
            # img = nrm(img).cpu().numpy()
            # plt.subplot(2, 5, completed_iterations)
            # plt.imshow(img, cmap=new_cmap)
            # if completed_iterations == 1:
            #     plt.colorbar()
            # Save raw results
            # This will preserve complex information
            np.save(f'{output_dir}/O_raw_{strategy_name}_alpha_{alpha:.2f}_trial_{trial + 1}.npy',O_raw.cpu().detach().numpy())
            np.save(f'{output_dir}/O_est_{strategy_name}_alpha_{alpha:.2f}_trial_{trial + 1}.npy', O_est.cpu().numpy())
            np.save(f'{output_dir}/phi_tot_{strategy_name}_alpha_{alpha:.2f}_trial_{trial + 1}.npy', phi_tot.cpu().numpy())
            np.save(f'{output_dir}/MTF_{strategy_name}_alpha_{alpha:.2f}_trial_{trial + 1}.npy', MTF.cpu().numpy())

            # Clear GPU memory
            torch.cuda.empty_cache()
        # plt.show()



print(f"Simulation data collection completed. Results saved to: {output_dir}")
print(f"Run the analysis script to analyze the collected data.")
