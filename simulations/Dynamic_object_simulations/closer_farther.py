import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.transforms.functional import rotate, center_crop
from utils.io import load_file_to_tensor
from utils.field_utils import generate_diffusers_and_PSFs
from propagation.propagation import angular_spectrum_gpu
from utils.image_processing import resize, center_crop, shift_cross_correlation, nrm, compute_similarity_score
from core.CTRCLASS import CTR_CLASS
from utils.visualization import display_field, get_custom_colormap
from torch.fft import fft2, ifft2, fftshift, ifftshift
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation



def rescale_complex_field(field, scale):
    """Rescales a complex field by a given scale factor."""
    return field * scale

def find_best_scaling_by_intensity_alignment(field, ref_intensity, candidate_dz, size, pixel_size, wavelength):
    """Find the best scaling factor and propagation distance for field alignment."""
    field_intensity = torch.abs(field)**2
    field_intensity_np = field_intensity.cpu().numpy()
    
    best_score = -np.inf
    best_scale = 1.0
    best_dz = 0.0
    
    for dz in candidate_dz:
        # Estimate scale using intensity ratio
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



# parameters
obj_size = 80
pixel_size_m = 5e-6
speckle_size_m = 15e-6  # Speckle size in meters
wavelength_m = 0.6328e-6 # (632.8 nm)
theta_deg = 0.5
M = 180  # Number of realizations
z_m = 4e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = load_file_to_tensor()
image = resize(image, obj_size).to(device)
field = image#*torch.exp(1j*image)

h, w = field.shape[:2]
# Pre-pad the image to
pad_h = int(1.5 * h)
pad_w = int(1.5*w)

padded_field = torch.nn.functional.pad(field, (pad_w, pad_w, pad_h, pad_h))
padded_size = padded_field.shape[0]

clean_diffuser, clean_PSF = generate_diffusers_and_PSFs(padded_size, 0 ,speckle_size_m ,pixel_size_m, wavelength_m, 1)
image_at_diffuser_plane = angular_spectrum_gpu(padded_field, pixel_size_m, wavelength_m, z_m) * clean_diffuser.to(device)
widefield = angular_spectrum_gpu(image_at_diffuser_plane, pixel_size_m, wavelength_m, -z_m)

shifts_z = np.random.uniform(-0.5e-2, 0.5e-2, size=M)

shifts_z[0] = 0
print(f"X range:  {100 *shifts_z.min()} to {100 * shifts_z.max()} cm")
shifted_frames = torch.zeros((M, padded_size, padded_size), dtype=torch.complex64, device=device)
for idx in range(M):
    current_shift = shifts_z[idx]
    z_to_propagate = current_shift + z_m
    shifted_frame = angular_spectrum_gpu(padded_field, pixel_size_m, wavelength_m, z_to_propagate)
    shifted_frames[idx] = shifted_frame

# img1 = shifted_frames[13].abs().cpu()
# img2 = shifted_frames[5].abs().cpu()
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(img1)
# plt.subplot(122)
# plt.imshow(img2)
# plt.show()
# show_fields_gif(shifted_frames, fps=40)
# show_fields_gif(fftshift(fft2(shifted_frames)), fps=3)


shifted_at_diffuser_plane = shifted_frames
diffusers, PSFs = generate_diffusers_and_PSFs(padded_size,theta_deg,speckle_size_m,pixel_size_m, wavelength_m, M)
distorted_at_diffuser_plane = diffusers.to(device) * shifted_at_diffuser_plane
#
# show_fields_gif(distorted_at_diffuser_plane, fps=20)
#
aligned_distorted_at_diffuser_plane = torch.zeros((M, padded_size, padded_size), dtype=torch.complex64, device=device)

# Define candidate ∆z range to scan (in meters)
candidate_dz = np.linspace(-1e-2, 1e-2, 201)  # e.g., -1 cm to +1 cm, 0.1 mm steps

ref_intensity = torch.abs(distorted_at_diffuser_plane[0])**2
ref_intensity_np = ref_intensity.cpu().numpy()

for idx in range(M):
    field = distorted_at_diffuser_plane[idx]
    best_scale, best_dz = find_best_scaling_by_intensity_alignment(
        field, ref_intensity_np, candidate_dz, padded_size, pixel_size_m, wavelength_m
    )

    scaled_field = rescale_complex_field(field, best_scale)
    aligned_distorted_at_diffuser_plane[idx] = scaled_field

    print(f"Frame {idx:3d}: best ∆z = {best_dz*1e3:+.2f} mm, scale = {best_scale:.5f}")


reference_frame = torch.abs(distorted_at_diffuser_plane[0])
# Reference intensity for scale estimation
reference_intensity = torch.abs(distorted_at_diffuser_plane[0]).cpu().numpy()

for idx in range(M):
    current_field = distorted_at_diffuser_plane[idx]
    current_intensity = torch.abs(current_field).cpu().numpy()

    # Estimate relative scale w.r.t. frame 0
    

    # Apply scaling to the complex field
    # scaled_field = ...

    aligned_distorted_at_diffuser_plane[idx] = scaled_field

    print(f"Frame {idx}: estimated scale = {scale:.5f}")


show_fields_gif(aligned_distorted_at_diffuser_plane, fps=10)


cropping_size = padded_size
cropped_aligned_distorted_at_diffuser_plane = center_crop(aligned_distorted_at_diffuser_plane, cropping_size)
cropped_distorted_at_diffuser_plane = center_crop(distorted_at_diffuser_plane, cropping_size)
# show_fields_gif(cropped_aligned_distorted_at_diffuser_plane, fps=30)
# Prepare for CLASS reconstruction
T = torch.permute(distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T_orig = torch.permute(cropped_distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T = T.to(device)
T_orig = T_orig.to(device)

# Run CLASS algorithm
_, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)
_, _, phi_tot_orig, MTF_orig = CTR_CLASS(T_orig, num_iters=1000)

O_est_at_diff = torch.conj(phi_tot) * MTF
O_est_0 = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z_m)
O_est = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0).cpu().numpy()
O_est = O_est / np.abs(O_est).max()

O_est_at_diff_orig = torch.conj(phi_tot_orig) * MTF_orig
O_est_0_orig = angular_spectrum_gpu(O_est_at_diff_orig, pixel_size_m, wavelength_m, -z_m)
O_est_orig = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0_orig).cpu().numpy()
O_est_orig = O_est_orig / np.abs(O_est_orig).max()

z_values = np.linspace(0.5 * z_m, 2 * z_m, 100)
propagated_Os = torch.zeros((100, cropping_size, cropping_size), dtype=torch.complex64, device=device)
std = 0
for idx, z_val in enumerate(z_values):
    propagated_Os[idx] = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z_val)
    if torch.std((propagated_Os[idx])) > std:
        std = torch.std((propagated_Os[idx]))
        best_idx = idx

show_fields_gif(propagated_Os, 10)

# display_field(propagated_Os[best_idx])


new_camp = get_custom_colormap()

plt.figure()
idx = M//3
plt.subplot(261)
img = widefield.to(torch.complex64) if not torch.is_complex(widefield) else widefield
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'widefield - magnitude')

plt.subplot(267)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'widfeield - Phase')

plt.subplot(262)
img = diffusers[idx]
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap='gray', vmin=0, vmax=1)
plt.title(f'diffuser - magnitude')

plt.subplot(268)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'diffuser - Phase')

plt.subplot(263)
img = distorted_at_diffuser_plane[idx]
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at diff) - mag')

plt.subplot(269)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'distorted (at diff) - Phase')

plt.subplot(264)
img = angular_spectrum_gpu(distorted_at_diffuser_plane[idx], pixel_size_m,wavelength_m, -z_m)
img = img.cpu()
plt.imshow(nrm(torch.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at obj) - mag')

plt.subplot(2,6,10)
plt.imshow(torch.angle(img), cmap='hsv')
plt.title(f'distorted (at obj) - Phase')

plt.subplot(2,6,5)
img = O_est_orig
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'reconstructed - mag \n no alignment')

plt.subplot(2,6,11)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'reconstructed - Phase\n no alignment')

plt.subplot(2,6,6)
img = O_est
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'reconstructed - mag')

plt.subplot(2,6,12)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'reconstructed - Phase')

plt.show()

