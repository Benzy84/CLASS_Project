import numpy as np
import torch
from torch.fft import *
from scipy.special import jn
import cv2
from torchvision.datasets import CIFAR10
from utils.image_processing import symmetric_pad_to_shape
from scipy.signal import unit_impulse



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


