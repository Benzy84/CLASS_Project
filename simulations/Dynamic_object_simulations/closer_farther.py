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


def apply_zoom_to_complex_field(field, scale):
    """
    Apply zoom to a complex field by processing real and imaginary parts separately.
    """
    if scale == 1.0:
        return field

    h, w = field.shape

    # Split and zoom
    real_zoomed = torch.nn.functional.interpolate(
        field.real.unsqueeze(0).unsqueeze(0),
        scale_factor=float(scale),  # Ensure it's a float
        mode='bilinear',
        align_corners=False
    ).squeeze()

    imag_zoomed = torch.nn.functional.interpolate(
        field.imag.unsqueeze(0).unsqueeze(0),
        scale_factor=float(scale),  # Ensure it's a float
        mode='bilinear',
        align_corners=False
    ).squeeze()

    zoomed = real_zoomed + 1j * imag_zoomed

    # Crop or pad
    if scale > 1.0:
        sh, sw = zoomed.shape
        start_h = (sh - h) // 2
        start_w = (sw - w) // 2
        return zoomed[start_h:start_h + h, start_w:start_w + w]
    else:
        sh, sw = zoomed.shape
        pad_h = (h - sh) // 2
        pad_w = (w - sw) // 2
        return torch.nn.functional.pad(
            zoomed,
            (pad_w, w - sw - pad_w, pad_h, h - sh - pad_h)
        )


def align_via_fourier_magnitude(field, ref_field):
    """
    Align fields by matching their Fourier magnitude profiles.
    Magnification in real space appears as radial scaling in Fourier space.
    """
    # Get Fourier transforms
    F_field = fftshift(fft2(field))
    F_ref = fftshift(fft2(ref_field))

    # Work with log-magnitude to enhance features
    mag_field = torch.log(torch.abs(F_field) + 1)
    mag_ref = torch.log(torch.abs(F_ref) + 1)

    # Radial average to get 1D profile
    def radial_profile(image):
        h, w = image.shape
        center = (h // 2, w // 2)
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w))
        R = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

        # Bin the radii
        max_r = int(torch.max(R))
        profile = torch.zeros(max_r)
        for r in range(max_r):
            mask = (R >= r) & (R < r + 1)
            if mask.any():
                profile[r] = image[mask].mean()
        return profile

    profile_field = radial_profile(mag_field)
    profile_ref = radial_profile(mag_ref)

    # Find scale by matching profiles...
    # Then apply inverse scaling in Fourier domain


def estimate_scale_from_speckle_statistics(field_intensity, ref_intensity):
    """
    Estimate scale from speckle grain size rather than direct intensity matching.
    """

    # Compute autocorrelation of both patterns
    def autocorrelation(img):
        F = fft2(img)
        return ifft2(F * torch.conj(F)).real

    acf_field = autocorrelation(field_intensity)
    acf_ref = autocorrelation(ref_intensity)

    # The width of the autocorrelation peak tells us the speckle size
    # which scales with magnification
    # Measure the width at half maximum...


def phase_gradient_alignment(field, ref_field):
    """
    Use phase gradients which are less sensitive to random phase but
    still contain scale information.
    """

    # Compute phase gradients
    def phase_gradient(f):
        phase = torch.angle(f)
        dy, dx = torch.gradient(phase)
        return torch.sqrt(dx ** 2 + dy ** 2)

    grad_field = phase_gradient(field)
    grad_ref = phase_gradient(ref_field)

    # These gradients should have similar patterns but different scales


def optimize_scale(field, ref_intensity, num_iterations=100):
    """
    Use gradient descent to find optimal scale.
    """
    scale = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([scale], lr=0.01)

    for _ in range(num_iterations):
        # Apply scale
        scaled_field = apply_scale_in_fourier(field, scale)

        # Loss
        loss = -compute_similarity_score(torch.abs(scaled_field) ** 2, ref_intensity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return scale.item()

def warp_field_registration(field, ref_field):
    """
    Use optical flow or similar to find warping between patterns.
    """
    # This could use cv2.calcOpticalFlowFarneback or similar
    # on the intensity patterns

# parameters
obj_size = 80
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

aligned_distorted_at_diffuser_plane = torch.zeros((M, padded_size, padded_size), dtype=torch.complex64, device=device)
aligned_distorted_at_diffuser_plane[0] = distorted_at_diffuser_plane[0]

# Reference intensity
ref_intensity = torch.abs(distorted_at_diffuser_plane[0]) ** 2

for idx in range(1, M):
    # Get current intensity
    current_intensity = torch.abs(distorted_at_diffuser_plane[idx]) ** 2

    # Find best scale using ONLY intensities
    best_scale = find_best_scale_from_intensities(
        current_intensity,
        ref_intensity,
        scale_range=(0.8, 1.2),  # Adjust based on your expected z-range
        num_scales=100
    )
    # Apply the zoom to the complex field
    aligned_field = apply_zoom_to_complex_field(
        distorted_at_diffuser_plane[idx],
        best_scale
    )

    aligned_distorted_at_diffuser_plane[idx] = aligned_field
    print(f"Frame {idx}/{M}: best scale = {best_scale:.4f}")

# show_fields_gif(aligned_distorted_at_diffuser_plane, fps=4)


cropping_size = padded_size
cropped_aligned_distorted_at_diffuser_plane = center_crop(aligned_distorted_at_diffuser_plane, cropping_size)
cropped_distorted_at_diffuser_plane = center_crop(distorted_at_diffuser_plane, cropping_size)
# show_fields_gif(cropped_aligned_distorted_at_diffuser_plane, fps=30)
# Prepare for CLASS reconstruction
T = torch.permute(cropped_aligned_distorted_at_diffuser_plane, [2, 1, 0]).reshape(cropping_size ** 2, -1)
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

