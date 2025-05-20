import numpy as np
import torch
from torch.fft import *
from scipy.special import jn




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


def gauss2D(sigma, size = 1025,mid = None):
    if mid is None:
        mid = [0,0]

    # Create delta function for very small sigma values
    if sigma < 1.0:
        # Create a zero tensor
        g = torch.zeros(size, size, dtype=torch.float32)

        # Set the center pixel to 1.0
        center_x = size // 2 + int(mid[0])
        center_y = size // 2 + int(mid[1])

        # Ensure indices are within bounds
        center_x = max(0, min(size - 1, center_x))
        center_y = max(0, min(size - 1, center_y))

        g[center_y, center_x] = 1.0
        return g

    x = torch.linspace(-size/2, size/2, size,dtype=torch.float32)
    x, y = torch.meshgrid(x, x)
    r2 = (x-mid[0])**2 + (y-mid[1])**2
    g = torch.exp(-0.5*r2/sigma**2)
    return g/g.max()


def circ(a, size = 1025,half = False):
    x = torch.linspace(-size/2, size/2, size,dtype=torch.float64)
    x, y = torch.meshgrid(x, x)
    r = torch.sqrt(x**2 + y**2)
    b = torch.zeros(2*[size],dtype=torch.float64)
    b[r < a] = 1
    if half:
        b[:, size // 2:] = 0
    # b /= b.sum()
    return b


def box(a, size = 1025,half = False):
    b = torch.zeros((size,size), dtype=torch.complex128)
    a2 = a//2
    b[size//2-a2:size//2+1+a2,size//2-a2:size//2+1+a2] = 1
    if half:
        b[:, size // 2:] = 0
    # b /= b.sum()
    return b


def hann(width, grid_size, center):

    px, py = center
    H = grid_size
    W = H
    x = torch.arange(W).float()
    y = torch.arange(H).float()

    X, Y = torch.meshgrid(x, y, indexing='xy')

    r = torch.sqrt((X - px+W//2) ** 2 + (Y - py+H//2) ** 2)

    hann_window = 0.5 * (1 + torch.cos(2 * torch.pi * r / width))

    hann_window[r > width / 2] = 0
    return hann_window


def hann2D(a, size=1025, mid=None):
    """
    Create a 2D Hann window with a specified width 'a' and grid size.

    Args:
        a (float): Defines the radius where the Hann window tapers to zero.
        size (int): Size of the 2D grid (grid will be size x size).
        mid (list or tuple): Center of the window. Defaults to the center of the grid.

    Returns:
        torch.Tensor: 2D Hann window.
    """
    if mid is None:
        mid = [0, 0]

    # Create a 1D grid and meshgrid
    x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float32)
    x, y = torch.meshgrid(x, x)

    # Compute the squared radial distance from the center
    r = torch.sqrt((x - mid[0]) ** 2 + (y - mid[1]) ** 2)

    # Hann window formula, only applying within the radius 'a'
    hann_window = 0.5 * (1 + torch.cos(torch.pi * r / a))

    # Set values outside the radius 'a' to zero
    hann_window[r > a] = 0

    return hann_window / hann_window.max()  # Normalize to the max value


def step(size = 1025,vert=False):
    s = np.zeros((size,size))
    if vert:
        s[size//2:,:] = 1
    else:
        s[:,size // 2:] = 1
    return s


def airy(size,D,r_sc):
    x = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2) / r_sc
    # jinc = lambda x: jn(1, x) / x
    # return np.pi * 0.5 * D * D * jinc(np.pi*D*r)
    return np.pi * D * jn(1, np.pi*D*r) / (2*r)


def hypot_like(a):
    size = a.shape[-1]
    x = torch.linspace(-size/2, size/2, size,dtype=torch.float64)
    x, y = torch.meshgrid(x, x)
    return torch.sqrt(x**2 + y**2)

def getObject(obj_size,grid_size_px,name='Dog'):
    a = CIFAR10('..\CIFAR10', download=True)
    im = torch.DoubleTensor(np.array(a[4120][0].convert("L").resize((obj_size, obj_size))))
    if name in ['US', 'USAF']:
        im = torch.DoubleTensor(cv2.resize(cv2.imread('USAF.jpg', cv2.IMREAD_GRAYSCALE), (obj_size, obj_size)))
    if name in ['Boat', 'boat']:
        im = torch.DoubleTensor(np.array(a[8][0].convert("L").resize((obj_size, obj_size))))
    if name in ['Dog','dog']:
        im = torch.DoubleTensor(np.array(a[4120][0].convert("L").resize((obj_size,obj_size))))
    if name in ['Res', 'res', 'Resolution', 'resolution','US AF','USAF']:
        im = torch.DoubleTensor(cv2.resize(cv2.imread('USAF.jpg', cv2.IMREAD_GRAYSCALE), (obj_size, obj_size)))
    if name in ['Car', 'car']:
        im = torch.DoubleTensor(np.array(a[4][0].convert("L").resize((obj_size, obj_size))))
    if name in ['Point', 'point','Delta','delta']:
        return torch.DoubleTensor(unit_impulse(2*[grid_size_px],'mid')) + 0j
    if name in ['Camera','camera','camera man','cameraman','Cameraman','Camera Man']:
        im = torch.DoubleTensor(cv2.resize(cv2.imread('cameraman.bmp', cv2.IMREAD_GRAYSCALE), (obj_size, obj_size)))
    if name in ['Points','points', 'random', 'Random', 'random points']:
        im = torch.rand(2*[obj_size],dtype=torch.float64)
        p = 0.98
        im[im > p] = 1
        im[im <= p] = 0
    if name in ['Two Points','two points']:
        middle = obj_size // 2
        im = torch.zeros_like(im)
        im[middle, middle-10] = 1
        im[middle, middle+10] = 1
    im /= im.max()
    return symmetric_pad_to_shape(im, grid_size_px) + 0j


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


def create_points_obj(M, N):
    # Step 1: Create a flat tensor of size M*M with the first N entries having random complex values
    c = torch.cat((torch.ones(N), torch.zeros(M * M - N)))

    # Step 2: Randomly permute the flattened tensors
    perm = torch.randperm(M * M)
    c = c[perm]

    return c.reshape(M, M)


def generate_diffusers_and_PSFs(sz, theta, speckle_size, pixel_size, wavelength, num_diffusers=1):
    """
    Generate simulated diffusers with specified angular spread.

    Parameters:
    -----------
    sz : int
        Size of the diffuser (pixels)
    theta : float
        Desired angular spread (in degrees)
    wavelength : float
        Wavelength of light
    pixel_size : float
        Physical size of each pixel (in meters)
    speckle_size : int
        Size of the speckles produced
    num_diffusers : int
        Number of different diffuser patterns to generate

    Returns:
    --------
    diffuser : torch.Tensor
        Complex-valued phase mask(s) representing the diffusers
    """
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Special case for theta = 0
    if theta == 0:
        # Create a constant phase (ones) for each diffuser
        if num_diffusers > 1:
            initial_diffusers = torch.ones((num_diffusers, sz, sz), dtype=torch.complex64)
        else:
            initial_diffusers = torch.ones((sz, sz), dtype=torch.complex64)


        # Apply spatial envelope to control speckle characteristics
        speckle_size_pixels = speckle_size / pixel_size


        spatial_envelope = gauss2D(sz // (2 * speckle_size_pixels), size=sz) # at diffuser plane
        centered_diffusers = initial_diffusers * spatial_envelope
        diffusers = centered_diffusers
        # PSF is the Fourier transform of the diffuser
        PSFs = fftshift(ifft2(ifftshift(diffusers)))
        psf_sums = (PSFs.abs() ** 2).sum(dim=(-2, -1), keepdim=True)
        psf_sums = torch.clamp(psf_sums, min=1e-20)
        PSFs_normalized = PSFs / torch.sqrt(psf_sums)

        return centered_diffusers, PSFs_normalized

    # Normal case - proceed with original function for non-zero theta
    # Calculate correlation distance in physical units (meters)
    d_corr_physical = wavelength / theta_rad

    # Convert physical correlation distance to pixels
    d_corr_pixels = d_corr_physical / pixel_size
    speckle_size_pixels = speckle_size / pixel_size

    # Generate random phase pattern (0 to 2π)
    if num_diffusers > 1:
        random_phases = torch.exp(2j * np.pi * torch.rand((num_diffusers, sz, sz)))
    else:
        random_phases = torch.exp(2j * np.pi * torch.rand((sz, sz)))


    # Apply Fourier filtering to create phase correlation
    fourier_filter = gauss2D(sz / d_corr_pixels, size=sz)
    # Apply filtering in the Fourier domain
    filtered_fourier = fourier_filter * fftshift(ifft2(ifftshift(random_phases, dim=(-2, -1))), dim=(-2, -1))
    # display_field(filtered_fourier[0])
    initial_diffusers = fftshift(fft2(ifftshift(filtered_fourier, dim=(-2, -1))), dim=(-2, -1))
    # Normalize to unit amplitude
    initial_diffusers /= initial_diffusers.abs()


    # t_psf = gauss2D(speckle_size_pixels / 2, size=sz)
    # display_field(t_psf)
    # t_diff = fftshift(fft2(ifftshift(t_psf)))
    # display_field(t_diff)
    # psf = fftshift(ifft2(ifftshift(t_diff)))
    # display_field(psf)
    # display_field(fftshift(fft2(ifftshift(psf))))

    # Apply spatial envelope to control speckle characteristics
    spatial_envelope = gauss2D(sz // (2 * speckle_size_pixels), size=sz)
    centered_diffusers = spatial_envelope * initial_diffusers
    diffusers = centered_diffusers

    PSFs = fftshift(ifft2(ifftshift(diffusers, dim=(-2, -1))), dim=(-2, -1))
    psf_sums = (PSFs.abs() ** 2).sum(dim=(-2, -1), keepdim=True)
    psf_sums = torch.clamp(psf_sums, min=1e-20)
    PSFs_normalized = PSFs / torch.sqrt(psf_sums)

    return centered_diffusers, PSFs_normalized


