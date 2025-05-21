import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os
import datetime
from torchvision.transforms import CenterCrop, Resize
from matplotlib.colors import LinearSegmentedColormap
from core.CTRCLASS import CTR_CLASS
from utils.field_utils import gauss2D, generate_diffusers_and_PSFs
from utils.io import load_file_to_tensor
from propagation.propagation import angular_spectrum_gpu
from utils.image_processing import shift_cross_correlation, fourier_convolution

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

def makedir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# Create timestamped output directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"spot_size_simulation_{timestamp}"
makedir(results_dir)

# Simulation parameters
sz = 600  # Image size in pixels
N_obj = 80  # Object size
C = CenterCrop(int(3 * N_obj))  # Cropping transform
C2 = CenterCrop(N_obj + 10)  # Smaller crop for visualization

# Physical parameters
pixel_size = 5  # Pixel size in microns
speckle_size = 15  # Speckle size in microns
theta = 0.3  # Angular spread in degrees
wavelength = 0.6328  # Wavelength in microns (632.8 nm)
M = 150  # Number of diffuser patterns

# Convert to SI units
pixel_size_m = pixel_size * 1e-6
speckle_size_m = speckle_size * 1e-6
wavelength_m = wavelength * 1e-6

# Calculate correlation distance from angular spread
theta_rad = torch.deg2rad(torch.tensor(theta))
d_corr = wavelength_m / theta_rad

# Ratio between spot size and correlation distance to test
ratios = [0.0001, 0.1, 0.3, 0.5, 1]
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
widefield = fourier_convolution(P, clean_PSF).to(device)
cutP = C(nrm(widefield.cpu())).numpy()

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
all_distorted_eff_obj_at_obj_plane = {}

# Process each ratio
for i, spot_size in enumerate(spot_sizes):
    ratio = ratios[i]
    print(f'Processing spot_size/d_corr = {ratio}: {i + 1}/{len(ratios)}')

    # Calculate spot size in pixels
    spot_size_in_pixels = spot_size // pixel_size_m

    # Create illumination pattern
    illumination = gauss2D(spot_size_in_pixels, sz)
    illumination_through_diffuser = illumination * diffusers

    z = 3e-2

    # Propagate to object plane
    start_time = time.time()

    illumination_at_object_plane = angular_spectrum_gpu(illumination_through_diffuser, pixel_size_m, wavelength_m, z)
    end_time = time.time()
    print(f"Time taken for GPU AS: {end_time - start_time:.4f} seconds")


    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(np.abs(P.cpu().numpy()))
    # plt.title('Oroginal object')
    # plt.subplot(1,3,2)
    # plt.imshow(np.abs((illumination_at_object_plane1).cpu().numpy()))
    # plt.title('packaged AS')
    # plt.subplot(1,3,3)
    # plt.imshow(np.abs((illumination_at_object_plane2).cpu().numpy()))
    # plt.title('angular_spectrum')
    # plt.show()


    # Create effective object (object * illumination)
    eff_obj = illumination_at_object_plane * P

    # Distort through the diffuser
    eff_obj_at_diffuser = angular_spectrum_gpu(eff_obj, pixel_size_m, wavelength_m, z)
    distorted_eff_obj = eff_obj_at_diffuser * diffusers
    distorted_eff_obj_at_obj_plane =  angular_spectrum_gpu(distorted_eff_obj, pixel_size_m, wavelength_m, -z)

    # distorted_eff_obj = fourier_convolution(eff_obj, PSFs)
    tensor_with_dims = distorted_eff_obj.unsqueeze(1)
    cropped_tensor = C(tensor_with_dims)
    cropped_distorted_obj = cropped_tensor.squeeze(1)
    # distorted_obj_fft = fftshift(fft2(cropped_distorted_obj))

    # Prepare for CLASS reconstruction
    # T = torch.permute(distorted_obj_fft, [2, 1, 0]).reshape((int(5 * N_obj)) ** 2, -1)
    T = torch.permute(cropped_distorted_obj, [2, 1, 0]).reshape((int(3 * N_obj)) ** 2, -1)
    T = T.to(device)

    # Run CLASS algorithm
    _, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)

    # _, PSF_std, obj_fourier_angle, obj_fourier_abs = CTR_CLASS(T, num_iters, imsize=shp)
    O_est_at_diff = torch.conj(phi_tot) * MTF
    O_est_0 = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z)


    # O_est_0 = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
    O_est = shift_cross_correlation(C(widefield), O_est_0).cpu().numpy()
    O_est = O_est / np.abs(O_est).max()

    # Store results
    all_illuminations[f'spot_size/d_corr = {ratio}'] = illumination
    all_ill_at_diff[f'spot_size/d_corr = {ratio}'] = illumination_through_diffuser[0]
    all_ill_at_obj[f'spot_size/d_corr = {ratio}'] = illumination_at_object_plane[0]
    all_eff_obj[f'spot_size/d_corr = {ratio}'] = eff_obj[0]
    all_distorted_eff_obj[f'spot_size/d_corr = {ratio}'] = distorted_eff_obj[0]
    all_distorted_eff_obj_at_obj_plane[f'spot_size/d_corr = {ratio}'] = distorted_eff_obj_at_obj_plane[0]
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