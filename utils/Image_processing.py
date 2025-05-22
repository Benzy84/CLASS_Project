import numpy as np
import torch
import scipy
from scipy.signal import fftconvolve
from torch.fft import fft2, fftshift, ifftshift, ifft2
from torchvision.transforms import CenterCrop
from utils.field_utils import gauss2D, circ
from torchvision.transforms import CenterCrop, Resize

nrm = lambda x: x/np.abs(x).max()


def resize(img, desired_shp):
    resize = Resize((desired_shp, desired_shp), antialias=True)
    return resize(img.unsqueeze(0)).squeeze(0)


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
    global original_dtype
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

def compute_similarity_score(img1, img2):
    # Ensure the images are the same size
    if img1.shape != img2.shape:
        raise ValueError("Both images should have the same dimensions.")

    # Convert PyTorch tensors to NumPy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Convert to float32 for better numerical accuracy
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Subtract mean and normalize the images
    img1 -= np.mean(img1)
    img1 /= np.linalg.norm(img1)
    img2 -= np.mean(img2)
    img2 /= np.linalg.norm(img2)

    # Compute FFTs
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    # Multiply by complex conjugate in frequency domain
    corr = np.fft.ifft2(f1 * np.conj(f2))

    # Obtain the similarity score as the maximum of the NCC
    similarity_score = np.abs(np.max(np.real(corr)))
    return similarity_score

def maxCorr(im1,im2):
    rot = torch.rot90
    sqrt = torch.sqrt
    if type(im1) == np.ndarray:
        rot = np.rot90
        sqrt = np.sqrt
    im1 -= im1.mean()
    im2 -= im2.mean()
    corr = fftconvolve(im2,rot(im1,2,[0,1]))
    norm_corr = (corr / sqrt((im1**2).sum()*(im2**2).sum()))
    return norm_corr.max(),np.unravel_index(norm_corr.argmax(), norm_corr.shape)


def shift_max_to_center(tensor):
    # Assuming tensor is of shape NxN
    N = tensor.size(0)
    # Get the index of the maximum element
    max_idx = torch.argmax(tensor)
    max_row, max_col = max_idx // N, max_idx % N
    # Compute the shifts needed
    shift_row = N // 2 - max_row
    shift_col = N // 2 - max_col
    # Apply the shift using torch.roll
    shifted_tensor = torch.roll(tensor, shifts=(shift_row, shift_col), dims=(0, 1))
    return shifted_tensor,shift_row,shift_col


def shift(im,s,mode='wrap'):
    return torch.DoubleTensor(scipy.ndimage.shift(im.real.numpy(),s,mode=mode)) + 1j*torch.DoubleTensor(scipy.ndimage.shift(im.imag.numpy(),s,mode=mode))


def shiftToCM(img):
    com = np.array(np.unravel_index(img.argmax(), img.shape))
    midpoint = np.array([img.shape[1] / 2, img.shape[0] / 2])
    s = midpoint - com
    return np.abs(shift(img,s,mode='wrap'))


def imageMosaic(imlist):
    im = shiftToCM(nrm(imlist[0]))
    for i in imlist:
        i = nrm(i)
        im += shift(i,maxCorr(im,i)[1],mode='wrap')
    return im


def shift_cross_correlation(dest, src):

    device = src.device
    dest = dest.to(device)

    if torch.is_complex(dest):
        dest_mag = dest.abs()
    else:
        dest_mag = dest

    # Handle reconstruction (source) - could be complex
    if torch.is_complex(src):
        # For complex src, first get magnitude
        src_mag = src.abs()
    else:
        # For real src, ensure positive values
        offset = src.min().abs() if src.min() < 0 else 0
        src_mag = src + offset
    # Get device


    # Create a stronger high-pass filter in frequency domain
    h, w = src_mag.shape[-2:]
    y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device),
                                    indexing='ij')
    center_y, center_x = h // 2, w // 2
    # Distance from center (shifted to put origin at center)
    dist_from_center = torch.sqrt(((y_grid - center_y) / center_y) ** 2 +
                                  ((x_grid - center_x) / center_x) ** 2)

    # High-pass filter: boost high frequencies, attenuate low frequencies
    high_freq_emphasis = 0
    high_pass = torch.pow(dist_from_center, high_freq_emphasis)
    high_pass /= high_pass.max().abs()


    # Apply filtering in frequency domain
    src_fft = fftshift(fft2(src_mag))
    dest_fft = fftshift(fft2(dest_mag))

    src_filtered = ifft2(ifftshift(src_fft * high_pass)).real
    dest_filtered = ifft2(ifftshift(dest_fft * high_pass)).real

    # Normalize to emphasize structure rather than intensity
    src_norm = (src_filtered - src_filtered.mean()) / (src_filtered.std() + 1e-8)
    dest_norm = (dest_filtered - dest_filtered.mean()) / (dest_filtered.std() + 1e-8)

    # Calculate cross-correlation
    q = torch.abs(fourier_cross_correlation(dest_norm, src_norm))

    # Find the maximum correlation position
    mxidx = torch.unravel_index(torch.argmax(q), dest.shape[-2:])
    shape_tensor = torch.tensor(dest.shape[-2:], device=dest.device)
    max_loc = shape_tensor // 2 - torch.tensor(mxidx, device=dest.device)

    # Apply shift to the original source
    src_shifted = torch.roll(src, shifts=(-max_loc[0].item(), -max_loc[1].item()), dims=(-2, -1))
    # display_field(src_shifted / src_shifted.abs().max() + dest / dest.abs().max())

    return src_shifted


def fourier_convolution(a, b, mode='same'):
    """
    Performs convolution between two 2D arrays using the Fourier transform method.
    Works with both PyTorch tensors and NumPy arrays.

    Parameters:
    -----------
    a : torch.Tensor or numpy.ndarray
        First input array
    b : torch.Tensor or numpy.ndarray
        Second input array
    mode : str, default='same'
        'same': Output size matches input size
        'full': Output size is the full convolution size

    Returns:
    --------
    torch.Tensor or numpy.ndarray
        Result of the convolution a * b, where * denotes convolution
    """
    # Original device tracking
    orig_device = getattr(a, 'device', None)
    computin_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle NumPy arrays
    is_numpy_a = isinstance(a, np.ndarray)
    is_numpy_b = isinstance(b, np.ndarray)

    if is_numpy_a:
        a_torch = torch.from_numpy(a.astype(np.complex64))
    else:
        a_torch = a.to(computin_device)

    if is_numpy_b:
        b_torch = torch.from_numpy(b.astype(np.complex64))
    else:
        b_torch = b.to(computin_device)

    # Ensure complex datatype
    if not torch.is_complex(a_torch):
        a_torch = a_torch.to(torch.complex64)
    if not torch.is_complex(b_torch):
        b_torch = b_torch.to(torch.complex64)

    # Handle padding based on mode
    if mode in ['full', 'Full', 'pad', 'Pad']:
        out_size = (a_torch.size(-2) + b_torch.size(-2) - 1,
                    a_torch.size(-1) + b_torch.size(-1) - 1)
        a_fft = fft2(a_torch, s=out_size)
        b_fft = fft2(b_torch, s=out_size)
    else:
        a_fft = fftshift(fft2(a_torch))
        b_fft = fftshift(fft2(ifftshift(b_torch)))

    # t_psf = gauss2D(speckle_size_pixels / 2, size=sz)
    # display_field(t_psf)
    # t_diff = fftshift(fft2(ifftshift(t_psf)))
    # display_field(t_diff)
    # psf = fftshift(ifft2(ifftshift(t_diff)))
    # display_field(psf)
    # display_field(fftshift(fft2(ifftshift(psf))))

    # Multiply in frequency domain
    result_fft = a_fft * b_fft

    # Inverse Fourier transform
    result = ifft2(ifftshift(result_fft))

    # Normalize by the number of elements
    result = result / torch.numel(result_fft)

    # Return to original device
    if orig_device is not None:
        result = result.to(orig_device)

    # Convert back to NumPy if input was NumPy
    if is_numpy_a or is_numpy_b:
        return result.cpu().numpy()
    else:
        return result


def low_pass_filter(x, radius=30, filter_type='gaussian', preserve_phase=False):
    """
    Apply a low-pass filter in the frequency domain.

    Parameters:
    -----------
    x : torch.Tensor or numpy.ndarray
        Input image/field
    radius : float, default=30
        Filter radius (pixels for 'gaussian', divisor for 'circle')
    filter_type : str, default='gaussian'
        Type of filter: 'gaussian' or 'circle'
    preserve_phase : bool, default=False
        If True, normalizes output to preserve phase (unit magnitude)

    Returns:
    --------
    torch.Tensor or numpy.ndarray
        Filtered image/field
    """
    # Compute the 2D FFT of the image
    f = fft2(x)

    # Shift the zero frequency component to the center
    fshift = fftshift(f)

    # Create filter mask based on type
    if filter_type.lower() == 'gaussian':
        mask = gauss2D(radius, x.shape[-1])
    else:  # 'circle'
        mask = circ(x.shape[-1] // radius, x.shape[-1])

    # Apply mask to the shifted FFT
    filtered_freq = fshift * mask

    # Inverse shift and FFT
    result = ifft2(ifftshift(filtered_freq))

    # Preserve phase if requested (unit magnitude)
    if preserve_phase:
        magnitude = torch.abs(result)
        magnitude_mask = (magnitude > 1e-10)  # Avoid division by zero
        result = torch.where(magnitude_mask,
                             result / magnitude,
                             result)

    return result


def high_pass_filter(x, radius=30, filter_type='gaussian', preserve_phase=False):
    """
    Apply a high-pass filter in the frequency domain.

    Parameters:
    -----------
    x : torch.Tensor or numpy.ndarray
        Input image/field
    radius : float, default=30
        Filter radius (pixels for 'gaussian', divisor for 'circle')
    filter_type : str, default='gaussian'
        Type of filter: 'gaussian' or 'circle'
    preserve_phase : bool, default=False
        If True, normalizes output to preserve phase (unit magnitude)

    Returns:
    --------
    torch.Tensor or numpy.ndarray
        Filtered image/field
    """
    # Compute the 2D FFT of the image
    f = fft2(x)

    # Shift the zero frequency component to the center
    fshift = fftshift(f)

    # Create high-pass filter mask based on type
    if filter_type.lower() == 'gaussian':
        # Gaussian high-pass = 1 - Gaussian low-pass
        mask = torch.ones(x.shape[-2:], device=x.device) - gauss2D(radius, x.shape[-1])
    else:  # 'circle'
        # Circle high-pass = 1 - Circle low-pass
        mask = torch.ones(x.shape[-2:], device=x.device) - circ(x.shape[-1] // radius, x.shape[-1])

    # Apply mask to the shifted FFT
    filtered_freq = fshift * mask

    # Inverse shift and FFT
    result = ifft2(ifftshift(filtered_freq))

    # Preserve phase if requested (unit magnitude)
    if preserve_phase:
        magnitude = torch.abs(result)
        magnitude_mask = (magnitude > 1e-10)  # Avoid division by zero
        result = torch.where(magnitude_mask,
                             result / magnitude,
                             result)

    return result


def fourier_cross_correlation(A, B, mode='same'):
    """
    Performs cross-correlation between two arrays with the peak centered.

    Parameters:
    -----------
    A : torch.Tensor
        First input array
    B : torch.Tensor
        Second input array
    mode : str, default='same'
        'same': Output size matches input size
        'full': Output includes complete overlap

    Returns:
    --------
    torch.Tensor
        Centered cross-correlation result
    """
    if mode in ['Full', 'full', 'pad', 'Pad']:
        # Calculate the size of the padded output
        out_size = A.size(-2) + B.size(-2) - 1, A.size(-1) + B.size(-1) - 1
        # Perform 2D FFT on both input matrices
        A_fft = fft2(A, s=out_size)
        B_fft = fft2(B, s=out_size)
    else:
        A_fft = fft2(A)
        B_fft = fft2(B)

    # Multiply in frequency domain (with conjugate for correlation)
    result_freq = A_fft * torch.conj(B_fft)

    # Inverse FFT, then shift to center the peak, and normalize
    result = fftshift(ifft2(result_freq)) / torch.numel(result_freq)

    return result


def symmetric_pad_to_shape(array, target_shape):
    """
    Symmetrically pads a 2D NumPy array to a specified shape.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array to be padded.
    target_shape : tuple of int
        Desired shape (height, width) after padding.

    Returns
    -------
    padded_array : np.ndarray
        The symmetrically padded array with shape equal to target_shape.

    Notes
    -----
    If the difference between target and original shape is odd,
    the extra padding is added to the bottom or right sides.
    """
    current_shape = np.array(array.shape)
    required_padding = np.array(target_shape) - current_shape

    if np.any(required_padding < 0):
        raise ValueError("Target shape must be greater than or equal to array shape in each dimension.")

    pad_before = required_padding // 2
    pad_after = required_padding - pad_before

    pad_width = ((pad_before[0], pad_after[0]),  # top, bottom
                 (pad_before[1], pad_after[1]))  # left, right

    return np.pad(array, pad_width, mode='constant')
