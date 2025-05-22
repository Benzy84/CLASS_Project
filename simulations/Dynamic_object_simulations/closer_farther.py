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


'''
Hi Claude,
You already have access to my simulation project, including cases where the object moves laterally (in the x-y plane) or rotates. In those cases, I handled the misalignment by detecting the motion (via cross-correlation) and then applying inverse geometric transforms directly on the complex fields at the diffuser plane. This worked well and preserved the phase structure needed for the CLASS algorithm.
Now I'm simulating a new case (closer_farther) where the object shifts slightly along the z-axis between frames — that is, each realization has the object located at a different axial distance before the diffuser. As before, the object field is propagated using the Angular Spectrum Method (ASM) to the diffuser plane. Then, I multiply the propagated field by a different random phase diffuser for each frame.
The problem is this: in the CLASS algorithm, the assumption is that the object is fixed, and the per-frame variation comes only from the phase distortions (like different diffuser patterns). But in this axial case, the object changes between frames due to the z-shift, and this breaks the CLASS model. I can't fix this by propagating the field back or forward, because the diffuser has already applied an unknown, random phase. So the axial deformation is baked into the data.
What I'm trying to do is find a way to normalize or correct for the axial shift — ideally making the object look effectively the same in all frames, so I can still use CLASS as-is.
I know all the physical parameters (wavelength, pixel size, field size, etc.), and I can define a range of possible axial shifts to scan over. I tried estimating radial magnification from intensity images, and even using Fresnel scaling models to rescale the complex fields accordingly — but this didn’t help. The speckle patterns are too dominant, and rescaling the complex field just introduces new distortions.
So I’m stuck: I can’t undo the propagation after the diffuser, and I can’t accurately rescale or align the fields just from intensity similarity. But the phase distortions caused by the diffuser are acceptable to CLASS — the only problem is the variation in the object itself.
Can you help me come up with a method to preprocess the complex fields in a way that approximates a fixed objects intesites on the diffuser?  Ideally, the solution should only operate on the complex fields at the diffuser plane (that’s all I have), and not assume knowledge of the true z-position per frame.

I suggested maybe not estimating Deltaz at all since we can not propagate it after we have it multiplied by the diffuser phase, but just take some reference intesity frame. frame [0], and scale all frames to this frame (zoom in \out or cut and resize), without knowing its z posizion. now after scaling the intensities will bee close to be identical, no? and the phase will be different, but it will be like this also because of the diffuser, so the algorithm will fix the phases already, as it is a part of the different diffusers, yes? in the rescaling I just do not know if we will just rescale it to be bigger\smaller or we should resacale it using the knowledge that we have (ASM and etc)
Thanks!
'''
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


def find_best_scale_from_intensities(field_intensity, ref_intensity, scale_range=(0.8, 1.2), num_scales=100):
    """
    Find the best zoom scale by comparing intensities only.

    Parameters:
    -----------
    field_intensity : torch.Tensor (real)
        Intensity of field to scale (|field|²)
    ref_intensity : torch.Tensor (real)
        Reference intensity to match

    Returns:
    --------
    best_scale : float
        Optimal zoom factor
    """
    h, w = field_intensity.shape
    scales = torch.linspace(scale_range[0], scale_range[1], num_scales)
    best_score = -float('inf')
    best_scale = 1.0

    for scale in scales:
        scale_float = scale.item()  # Convert tensor to float

        if scale_float == 1.0:
            scaled_intensity = field_intensity
        else:
            # Zoom the intensity image
            scaled_intensity = torch.nn.functional.interpolate(
                field_intensity.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_float,  # Use the float value
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Crop or pad to original size
            if scale_float > 1.0:
                # Crop center
                sh, sw = scaled_intensity.shape
                start_h = (sh - h) // 2
                start_w = (sw - w) // 2
                scaled_intensity = scaled_intensity[start_h:start_h + h, start_w:start_w + w]
            else:
                # Pad
                sh, sw = scaled_intensity.shape
                pad_h = (h - sh) // 2
                pad_w = (w - sw) // 2
                scaled_intensity = torch.nn.functional.pad(
                    scaled_intensity,
                    (pad_w, w - sw - pad_w, pad_h, h - sh - pad_h)
                )

        # Use your similarity function
        score = compute_similarity_score(scaled_intensity, ref_intensity)

        if score > best_score:
            best_score = score
            best_scale = scale_float  # Store as float

    return best_scale


import torch
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch.nn.functional as F


def fourier_domain_scaling(field, scale_factor):
    """
    Apply scaling in Fourier domain - more accurate than spatial interpolation.
    Scaling in real space = inverse scaling in Fourier space.
    """
    if abs(scale_factor - 1.0) < 1e-6:
        return field

    # Get FFT (properly shifted)
    F_field = fftshift(fft2(field))
    h, w = F_field.shape

    # Create coordinate grids for interpolation
    u = torch.linspace(-1, 1, w, device=field.device)
    v = torch.linspace(-1, 1, h, device=field.device)
    U, V = torch.meshgrid(u, v, indexing='xy')

    # Scale coordinates (inverse scaling in Fourier domain)
    U_scaled = U / scale_factor
    V_scaled = V / scale_factor

    # Clamp to valid range for grid_sample
    coords = torch.stack([U_scaled, V_scaled], dim=-1).unsqueeze(0)
    coords = torch.clamp(coords, -1, 1)

    # Interpolate real and imaginary parts separately
    F_real_scaled = F.grid_sample(
        F_field.real.unsqueeze(0).unsqueeze(0),
        coords,
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze()

    F_imag_scaled = F.grid_sample(
        F_field.imag.unsqueeze(0).unsqueeze(0),
        coords,
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze()

    F_scaled = F_real_scaled + 1j * F_imag_scaled

    # IFFT back (with proper shifting)
    return ifft2(ifftshift(F_scaled))


def find_scale_via_radial_profile(field_intensity, ref_intensity, scale_range=(0.8, 1.2), num_scales=100):
    """
    Find scaling by matching radial profiles in Fourier domain.
    This is more robust for speckle patterns.
    """

    def get_radial_profile(intensity):
        # FFT of intensity
        F = fftshift(fft2(intensity))
        magnitude = torch.log(torch.abs(F) + 1)

        h, w = magnitude.shape
        center = (h // 2, w // 2)

        # Create radial coordinate
        y, x = torch.meshgrid(torch.arange(h, device=intensity.device),
                              torch.arange(w, device=intensity.device), indexing='ij')
        r = torch.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # Bin into radial profile
        max_r = int(r.max().item())
        profile = torch.zeros(max_r, device=intensity.device)

        for i in range(max_r):
            mask = (r >= i) & (r < i + 1)
            if mask.any():
                profile[i] = magnitude[mask].mean()

        return profile

    ref_profile = get_radial_profile(ref_intensity)

    scales = torch.linspace(scale_range[0], scale_range[1], num_scales, device=field_intensity.device)
    best_score = -float('inf')
    best_scale = 1.0

    for scale in scales:
        # Scale the intensity and get its profile
        if scale != 1.0:
            scaled_intensity = F.interpolate(
                field_intensity.unsqueeze(0).unsqueeze(0),
                scale_factor=scale.item(),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Crop/pad to match size
            h, w = ref_intensity.shape
            if scale > 1.0:
                sh, sw = scaled_intensity.shape
                start_h, start_w = (sh - h) // 2, (sw - w) // 2
                scaled_intensity = scaled_intensity[start_h:start_h + h, start_w:start_w + w]
            else:
                sh, sw = scaled_intensity.shape
                pad_h, pad_w = (h - sh) // 2, (w - sw) // 2
                scaled_intensity = F.pad(scaled_intensity, (pad_w, w - sw - pad_w, pad_h, h - sh - pad_h))
        else:
            scaled_intensity = field_intensity

        field_profile = get_radial_profile(scaled_intensity)

        # Compare profiles (use shorter length)
        min_len = min(len(ref_profile), len(field_profile))
        if min_len > 10:  # Need enough points
            ref_norm = ref_profile[:min_len]
            field_norm = field_profile[:min_len]

            # Normalize
            ref_norm = ref_norm / (torch.norm(ref_norm) + 1e-8)
            field_norm = field_norm / (torch.norm(field_norm) + 1e-8)

            # Correlation score
            score = torch.dot(ref_norm, field_norm).item()

            if score > best_score:
                best_score = score
                best_scale = scale.item()

    return best_scale


def physics_constrained_scaling(z_nominal, z_range, field_intensity, ref_intensity):
    """
    Use physics knowledge to constrain the search.
    Magnification ≈ z_actual / z_nominal for small changes.
    """
    z_min = z_nominal - z_range / 2
    z_max = z_nominal + z_range / 2

    scale_min = z_min / z_nominal
    scale_max = z_max / z_nominal

    return find_scale_via_radial_profile(
        field_intensity, ref_intensity,
        scale_range=(scale_min, scale_max),
        num_scales=50
    )


def speckle_correlation_scaling(field_intensity, ref_intensity, scale_range=(0.8, 1.2)):
    """
    Use speckle correlation length to find scaling.
    Speckle size scales with magnification.
    """

    def get_correlation_length(intensity):
        # Autocorrelation
        autocorr = ifft2(torch.abs(fft2(intensity)) ** 2).real
        autocorr = fftshift(autocorr)

        # Find width at half maximum
        center = tuple(s // 2 for s in autocorr.shape)
        max_val = autocorr[center]

        # Look along horizontal line
        h_line = autocorr[center[0], :]
        half_max = max_val / 2

        # Find where it drops below half max
        center_idx = len(h_line) // 2
        width = 0
        for i in range(1, center_idx):
            if h_line[center_idx + i] < half_max:
                width = i
                break

        return width if width > 0 else 1

    ref_corr_length = get_correlation_length(ref_intensity)

    scales = torch.linspace(scale_range[0], scale_range[1], 50, device=field_intensity.device)
    best_scale = 1.0
    best_diff = float('inf')

    for scale in scales:
        # Scale and measure correlation length
        if scale != 1.0:
            scaled = F.interpolate(
                field_intensity.unsqueeze(0).unsqueeze(0),
                scale_factor=scale.item(),
                mode='bilinear'
            ).squeeze()

            # Crop/pad to match
            h, w = ref_intensity.shape
            if scale > 1.0:
                sh, sw = scaled.shape
                start_h, start_w = (sh - h) // 2, (sw - w) // 2
                scaled = scaled[start_h:start_h + h, start_w:start_w + w]
            else:
                sh, sw = scaled.shape
                pad_h, pad_w = (h - sh) // 2, (w - sw) // 2
                scaled = F.pad(scaled, (pad_w, w - sw - pad_w, pad_h, h - sh - pad_h))
        else:
            scaled = field_intensity

        corr_length = get_correlation_length(scaled)
        diff = abs(corr_length - ref_corr_length)

        if diff < best_diff:
            best_diff = diff
            best_scale = scale.item()

    return best_scale


def multi_metric_scaling(field, ref_intensity, z_nominal=None, z_range=None):
    """
    Combine multiple metrics for more robust scaling estimation.
    """
    field_intensity = torch.abs(field) ** 2

    # Method 1: Radial profile matching
    scale1 = find_scale_via_radial_profile(field_intensity, ref_intensity)

    # Method 2: Speckle correlation (if reliable)
    try:
        scale2 = speckle_correlation_scaling(field_intensity, ref_intensity)
        weight2 = 0.3
    except:
        scale2 = scale1
        weight2 = 0.0

    # Method 3: Physics constraint (if available)
    if z_nominal is not None and z_range is not None:
        scale3 = physics_constrained_scaling(z_nominal, z_range, field_intensity, ref_intensity)
        weight3 = 0.4
    else:
        scale3 = scale1
        weight3 = 0.0

    # Weighted average
    weight1 = 1.0 - weight2 - weight3
    final_scale = weight1 * scale1 + weight2 * scale2 + weight3 * scale3

    return final_scale


def enhanced_field_alignment(distorted_fields, z_nominal, z_range):
    """
    Enhanced alignment using multiple methods.

    Parameters:
    -----------
    distorted_fields : torch.Tensor (M, H, W)
        Complex fields at diffuser plane after random phase multiplication
    z_nominal : float
        Nominal z distance
    z_range : float
        Expected range of z variations

    Returns:
    --------
    aligned_fields : torch.Tensor (M, H, W)
        Aligned complex fields
    """
    M = distorted_fields.shape[0]
    aligned_fields = torch.zeros_like(distorted_fields)

    # Use first frame as reference
    aligned_fields[0] = distorted_fields[0]
    ref_intensity = torch.abs(distorted_fields[0]) ** 2

    print("Aligning fields using enhanced multi-metric approach...")

    for idx in range(1, M):
        current_field = distorted_fields[idx]

        # Find best scale using multiple metrics
        best_scale = multi_metric_scaling(
            current_field, ref_intensity, z_nominal, z_range
        )

        # Apply scaling in spatial domain (more reliable)
        aligned_field = apply_zoom_to_complex_field(current_field, best_scale)

        aligned_fields[idx] = aligned_field
        print(f"Frame {idx}/{M}: scale = {best_scale:.4f}")

    return aligned_fields


# Alternative approach: Use known physics directly
def physics_based_correction(fields, z_shifts, z_nominal):
    """
    If you know the actual z_shifts, use physics directly.
    This is the most accurate method.
    """
    corrected_fields = torch.zeros_like(fields)

    for idx, z_shift in enumerate(z_shifts):
        z_actual = z_nominal + z_shift
        scale_factor = z_nominal / z_actual  # Inverse magnification to correct

        corrected_fields[idx] = fourier_domain_scaling(fields[idx], scale_factor)

    return corrected_fields


# Helper function for spatial scaling (your current method, improved)
def apply_zoom_to_complex_field(field, scale):
    """Improved version of your current spatial scaling method."""
    if abs(scale - 1.0) < 1e-6:
        return field

    h, w = field.shape

    # Use more sophisticated interpolation
    real_zoomed = F.interpolate(
        field.real.unsqueeze(0).unsqueeze(0),
        scale_factor=float(scale),
        mode='bicubic',  # Better interpolation
        align_corners=False
    ).squeeze()

    imag_zoomed = F.interpolate(
        field.imag.unsqueeze(0).unsqueeze(0),
        scale_factor=float(scale),
        mode='bicubic',  # Better interpolation
        align_corners=False
    ).squeeze()

    zoomed = real_zoomed + 1j * imag_zoomed

    # Improved cropping/padding
    if scale > 1.0:
        sh, sw = zoomed.shape
        start_h = (sh - h) // 2
        start_w = (sw - w) // 2
        end_h = start_h + h
        end_w = start_w + w

        # Ensure we don't go out of bounds
        if end_h > sh or end_w > sw:
            # Pad if needed
            pad_h = max(0, end_h - sh)
            pad_w = max(0, end_w - sw)
            zoomed = F.pad(zoomed, (0, pad_w, 0, pad_h))

        return zoomed[start_h:start_h + h, start_w:start_w + w]
    else:
        sh, sw = zoomed.shape
        pad_h = (h - sh) // 2
        pad_w = (w - sw) // 2
        # Handle odd differences
        pad_h_after = h - sh - pad_h
        pad_w_after = w - sw - pad_w

        return F.pad(zoomed, (pad_w, pad_w_after, pad_h, pad_h_after))

# parameters
obj_size = 130
pixel_size_m = 5e-6
speckle_size_m = 15e-6  # Speckle size in meters
wavelength_m = 0.6328e-6 # (632.8 nm)
theta_deg = 0.5
M = 200  # Number of realizations
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
# show_fields_gif(shifted_frames, fps=4)
# show_fields_gif(fftshift(fft2(shifted_frames)), fps=3)


shifted_at_diffuser_plane = shifted_frames
diffusers, PSFs = generate_diffusers_and_PSFs(padded_size,theta_deg,speckle_size_m,pixel_size_m, wavelength_m, M)
distorted_at_diffuser_plane = diffusers.to(device) * shifted_at_diffuser_plane
#
# show_fields_gif(distorted_at_diffuser_plane, fps=20)
#

# Replace your existing loop (lines around 220-230) with:
# After getting distorted_at_diffuser_plane...

# Replace this section in your closer_farther.py:
# (Around where you have the alignment loop)

# After getting distorted_at_diffuser_plane...

# Method 1: Enhanced multi-metric alignment (RECOMMENDED)
aligned_distorted_at_diffuser_plane = enhanced_field_alignment(
    distorted_at_diffuser_plane,
    z_nominal=z_m,  # Your nominal z distance
    z_range=1e-2  # Expected range of z variations (1 cm)
)

# Method 2: If you know the actual z_shifts (MOST ACCURATE)
# aligned_distorted_at_diffuser_plane = physics_based_correction(
#     distorted_at_diffuser_plane,
#     shifts_z,  # Your actual z shifts
#     z_m        # Nominal z distance
# )

# Method 3: Manual selection of method
aligned_distorted_at_diffuser_plane_a = torch.zeros_like(distorted_at_diffuser_plane)
aligned_distorted_at_diffuser_plane_b = torch.zeros_like(distorted_at_diffuser_plane)
ref_intensity = torch.abs(distorted_at_diffuser_plane[0])**2

for idx in range(M):
    current_field = distorted_at_diffuser_plane[idx]
    current_intensity = torch.abs(current_field)**2

    # Choose your preferred method:
    # Option A: Radial profile matching (best for speckle)
    best_scale_a = find_scale_via_radial_profile(current_intensity, ref_intensity)

    # Option B: Physics-constrained search
    best_scale_b = physics_constrained_scaling(z_m, 1e-2, current_intensity, ref_intensity)


    aligned_field_a = apply_zoom_to_complex_field(current_field, best_scale_a)
    aligned_field_b = apply_zoom_to_complex_field(current_field, best_scale_b)

    # # Apply scaling (Fourier domain is more accurate)
    # try:
    #     aligned_field = fourier_domain_scaling(current_field, best_scale)
    # except:
    #     # Fallback to improved spatial scaling
    #     aligned_field = apply_zoom_to_complex_field(current_field, best_scale)

    aligned_distorted_at_diffuser_plane_a[idx] = aligned_field_a
    aligned_distorted_at_diffuser_plane_b[idx] = aligned_field_b
    print(f"Frame {idx}/{M}: scale = {best_scale_a:.4f}")

print("Enhanced alignment complete!")

# Continue with your existing code...


# show_fields_gif(aligned_distorted_at_diffuser_plane, fps=2)


cropping_size = padded_size
cropped_aligned_distorted_at_diffuser_plane = center_crop(aligned_distorted_at_diffuser_plane, cropping_size)
cropped_aligned_distorted_at_diffuser_plane_a = center_crop(aligned_distorted_at_diffuser_plane_a, cropping_size)
cropped_aligned_distorted_at_diffuser_plane_b = center_crop(aligned_distorted_at_diffuser_plane_b, cropping_size)
cropped_distorted_at_diffuser_plane = center_crop(distorted_at_diffuser_plane, cropping_size)
# show_fields_gif(cropped_aligned_distorted_at_diffuser_plane, fps=30)
# Prepare for CLASS reconstruction
T = torch.permute(cropped_aligned_distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T_a = torch.permute(cropped_aligned_distorted_at_diffuser_plane_a, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T_b = torch.permute(cropped_aligned_distorted_at_diffuser_plane_b, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T_orig = torch.permute(cropped_distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
T = T.to(device)
T_a = T_a.to(device)
T_b = T_b.to(device)
T_orig = T_orig.to(device)

# Run CLASS algorithm
_, _, phi_tot, MTF = CTR_CLASS(T, num_iters=1000)
_, _, phi_tot_a, MTF_a = CTR_CLASS(T_a, num_iters=1000)
_, _, phi_tot_b, MTF_b = CTR_CLASS(T_b, num_iters=1000)
_, _, phi_tot_orig, MTF_orig = CTR_CLASS(T_orig, num_iters=1000)

O_est_at_diff = torch.conj(phi_tot) * MTF
O_est_0 = angular_spectrum_gpu(O_est_at_diff, pixel_size_m, wavelength_m, -z_m)
O_est = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0).cpu().numpy()
O_est = O_est / np.abs(O_est).max()

O_est_at_diff_a = torch.conj(phi_tot_a) * MTF_a
O_est_0_a = angular_spectrum_gpu(O_est_at_diff_a, pixel_size_m, wavelength_m, -z_m)
O_est_a = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0_a).cpu().numpy()
O_est_a = O_est_a / np.abs(O_est_a).max()

O_est_at_diff_b = torch.conj(phi_tot_b) * MTF_b
O_est_0_b = angular_spectrum_gpu(O_est_at_diff_b, pixel_size_m, wavelength_m, -z_m)
O_est_b = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0_b).cpu().numpy()
O_est_b = O_est_b / np.abs(O_est_b).max()

O_est_at_diff_orig = torch.conj(phi_tot_orig) * MTF_orig
O_est_0_orig = angular_spectrum_gpu(O_est_at_diff_orig, pixel_size_m, wavelength_m, -z_m)
O_est_orig = shift_cross_correlation(center_crop(widefield,cropping_size), O_est_0_orig).cpu().numpy()
O_est_orig = O_est_orig / np.abs(O_est_orig).max()

z_values = np.linspace(0.5 * z_m, 2 * z_m, 100)
propagated_Os = torch.zeros((100, cropping_size, cropping_size), dtype=torch.complex64, device=device)
std = 0
for idx, z_val in enumerate(z_values):
    propagated_Os[idx] = angular_spectrum_gpu(O_est_at_diff_orig, pixel_size_m, wavelength_m, -z_val)
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
img = O_est_orig
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at diff) - mag')

plt.subplot(269)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'distorted (at diff) - Phase')

plt.subplot(264)
img = angular_spectrum_gpu(distorted_at_diffuser_plane[idx], pixel_size_m,wavelength_m, -z_m)
img = O_est
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'distorted (at obj) - mag')

plt.subplot(2,6,10)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'distorted (at obj) - Phase')

plt.subplot(2,6,5)
img = O_est_a
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'reconstructed - mag \n no alignment')

plt.subplot(2,6,11)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'reconstructed - Phase\n no alignment')

plt.subplot(2,6,6)
img = O_est_b
plt.imshow(nrm(np.abs(img)), cmap=new_camp, vmin=0, vmax=1)
plt.title(f'reconstructed - mag')

plt.subplot(2,6,12)
plt.imshow(np.angle(img), cmap='hsv')
plt.title(f'reconstructed - Phase')

plt.show()

