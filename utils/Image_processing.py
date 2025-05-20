import numpy as np
import torch
import scipy
from scipy.signal import fftconvolve
from torchvision.transforms import CenterCrop



nrm = lambda x: x/np.abs(x).max()



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


