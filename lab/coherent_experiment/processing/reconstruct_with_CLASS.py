import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from CTRCLASS import CTR_CLASS
from torch.fft import *
from tkinter import filedialog
import matplotlib
from scipy.io import savemat
import h5py
import numpy as np
import scipy.io as sio





import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

matplotlib.use('TkAgg')
plt.ion()

num_iters = 1000
additional_fiff = False

# Function to get a unique filename
def get_unique_filename(base_path, base_name, extension):
    counter = 1
    file_name = f"{base_name}{extension}"
    full_path = os.path.join(base_path, file_name)

    while os.path.exists(full_path):
        file_name = f"{base_name}_{counter}{extension}"
        full_path = os.path.join(base_path, file_name)
        counter += 1

    return file_name

def CC(x,y):
    q = fft2(x) * torch.conj(fft2(y))
    return fftshift(ifft2(q/torch.abs(q)))
def shift_cross_correlation(dest, src):
    q = torch.abs(CC(dest.abs(), src.abs()))
    mxidx =np.unravel_index(torch.argmax(q), dest.shape[-2:])
    max_loc = np.array(dest.shape[-2:]) // 2 - mxidx

    src_shifted = torch.roll(src, shifts=(-max_loc[0], -max_loc[1]), dims=(-2, -1))

    return src_shifted

def gauss2D(a, size = 1025,mid = None):
    if mid is None:
        mid = [0,0]
    x = torch.linspace(-size/2, size/2, size,dtype=torch.float32)
    x, y = torch.meshgrid(x, x)
    r2 = (x-mid[0])**2 + (y-mid[1])**2
    g = torch.exp(-0.5*r2/a**2)
    return g/g.max()
postprocess = lambda x:x
nrm = lambda x:x/x.abs().max()
tounit8 = lambda x:(255*nrm(x).numpy()).astype('uint8')
phs = lambda x:x/x.abs()
mx1 = lambda x:x/x.abs().max()


# Load fields
initial_directory = 'D:\Lab Images and data local'
pth = filedialog.askdirectory(title='Select Folder', initialdir=initial_directory)
base_folder = os.path.dirname(pth)
saving_pth = os.path.join(base_folder, 'CLASS reconstructions')
# Create the directory if it does not exist
if not os.path.exists(saving_pth):
    os.makedirs(saving_pth)
    print(f"Directory created: {saving_pth}")
else:
    print(f"Directory already exists: {saving_pth}")

fields = torch.cat([torch.from_numpy(np.load(os.path.join(pth, f))).unsqueeze(0) for f in os.listdir(pth)],0)



# Rearrange dimensions to 850x850x180 (from 180x850x850)
# fields = fields.permute(1, 2, 0)  # Change dimension order

# Convert tensor to numpy array
fields_numpy = fields.numpy()
# Save as HDF5 file
save_file_path = os.path.join(saving_pth, "fields_v7_322.h5")
with h5py.File(save_file_path, 'w') as f:
    f.create_dataset('fields', data=fields_numpy)

print(f"Fields saved to {save_file_path} in HDF5 format")
# fields = fields[:,:-1,:]
# fields = fields[::2]
# Number of copies you want
# fields = fields.repeat(167, 1, 1)
#
# # Pad 2 pixels on each side
# padding = (50, 50, 50, 50)
# # Apply padding to each 2D field
# fields = F.pad(fields, padding, "constant", 0)

# field0 = fields[0]  # Directly use the tensor, no conversion needed
# fields[1:] = fields[0]
num_fields = fields.shape[0]

print(fields.shape)
# shp = field0.shape
shp = fields.shape[1:]
mx_shp = max(*shp)
#
# # Padding configuration
# # pad format: (pad_left, pad_right, pad_top, pad_bottom)
# pad_size = mx_shp // 10
#
# # Padding configuration
# # pad format: (pad_left, pad_right, pad_top, pad_bottom)
# padding = (pad_size, pad_size, pad_size, pad_size)  # Apply equal padding to all sides
#
# # Apply padding to all fields
# fields = F.pad(fields, padding, "constant", 0)
#
# # Apply Gaussian filter for smoothing
# # Convert tensor to numpy array for processing with scipy
# fields_numpy = fields.numpy()
# sigma = 2  # Standard deviation for the Gaussian kernel, adjust as needed
# fields_smoothed = torch.empty_like(fields)  # Prepare tensor to hold smoothed fields
#
# # Loop through each field in the batch
# for i in range(fields.shape[0]):
#     # Ensure that sigma is applied correctly for each 2D field
#     fields_smoothed[i] = torch.from_numpy(gaussian_filter(fields[i].numpy(), sigma=sigma))
#
# #fields = fields_smoothed
# shp = fields.shape[1:]
# mx_shp = max(*shp)

# Create M diffusers
#Diffiseers parameters
# d_corr = 10
# speckle_size = 5
# diffusers = phs(ifft2(ifftshift(gauss2D(mx_shp // d_corr, mx_shp) * fftshift(fft2(torch.exp(2j *np.pi * torch.rand(num_fields,*(2*[mx_shp]))))))))
# diffusers = (gauss2D(mx_shp//speckle_size,mx_shp)*diffusers)[:,:shp[0],:shp[1]]
# plt.imshow(diffusers[0].angle(),'hsv')
# plt.colorbar()
# plt.waitforbuttonpress(1)

# system parameters
wavelength = 632.8e-9
system_mag = 200 / 125
original_shape = 2472
blob_shape = shp[0]
mag = system_mag * blob_shape / original_shape
dx = dy = 5.5e-6 / mag  # Pixel size in x-direction (m)

#
# # creating propagated point field
# z = 0.07148 # Propagation distance in meters
# grid_size = fields.shape[1]  # Size of the grid (850x850)
# pixel_size = dx  # Pixel size in meters (adjust as needed)
#
# # Create a coordinate grid centered at (0, 0)
# x = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
# y = np.linspace(-grid_size // 2, grid_size // 2, grid_size) * pixel_size
# X, Y = np.meshgrid(x, y)
#
# # Calculate radial distance from the point source (assumed at the origin)
# r = np.sqrt(X**2 + Y**2 + z**2)
#
# # Calculate field amplitude and phase at distance z from the source
# amplitude = torch.from_numpy(1 / r)  # Amplitude falls off as 1/r
# phase = torch.from_numpy((2 * np.pi / (wavelength)) * r)  # Phase term
# propagated_point_field = amplitude * torch.exp(1j * phase)  # Complex field
# propagated_point_field /= torch.max(torch.abs(propagated_point_field))
#
# fields_without_phases = torch.zeros_like(fields)
# for i in range(num_fields):
#     fields_without_phases[i] = fields[i] * torch.exp(-1j * phase)
#

# Diffuser parameters
theta_diff = np.deg2rad(0.5)
d_corr = 2 * 632.8e-9 / theta_diff
d_corr_pix = d_corr / dx
speckle_size = 3
sigma1 = mx_shp // d_corr_pix  # Standard deviation for frequency domain Gaussian
sigma2 = mx_shp // speckle_size  # Standard deviation for spatial domain Gaussian

# Generate a random phase screen
random_phase_matrix = torch.rand(1, mx_shp, mx_shp)
complex_exponential_of_phase = torch.exp(2j * np.pi * random_phase_matrix)

# Apply Gaussian filter in the frequency domain
filtered_frequency_data = gauss2D(sigma1, mx_shp) * fftshift(fft2(complex_exponential_of_phase))

# Transform to spatial domain and extract phase
diffusers = phs(ifft2(ifftshift(filtered_frequency_data)))

# Adjust speckle size in the spatial domain and crop to the desired output size
diffusers = (gauss2D(sigma2, mx_shp) * diffusers)[:180,:shp[0],:shp[1]]
#
# plt.imshow(diffusers[0].angle(),'hsv')
# plt.colorbar()
# plt.waitforbuttonpress(1)
# plt.show(block=False)


Icam = fields
if additional_fiff:
    Icam = fields * diffusers
    additional_fiff = True
    # Iterate over each field and save it
    output_folder = 'output_fields'
    os.makedirs(output_folder, exist_ok =True)

    for i in range(Icam.shape[0]):
        field_np = Icam[i].numpy()  # Convert each field to NumPy array
        np.save(os.path.join(output_folder, f'with simulated diffuser {i}.npy'), field_np)

    print("Fields saved successfully.")
Icam = Icam.to(torch.complex64)


# # Show few measurements
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(fields[0].abs(),'hot')
# plt.title('|First field|')
# plt.subplot(1,2,2)
#
# # show fields
# img = plt.imshow(Icam[0].abs(),cmap='hot')  # Corrected variable name
# plt.title('|Distorted fields|')
# for I in Icam[:8]:  # Corrected variable name
#     img.set_data(I.angle())
#     img.set_clim(vmin=-np.pi, vmax=np.pi)
#     plt.pause(0.5)  # Pause for a second to allow GUI to update the display
# plt.close()
#




# Create matrix and apply CLASS

T = torch.permute((Icam), [2, 1, 0]).reshape(np.prod(shp), -1)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
_, PSF_std, obj_fourier_angle, obj_fourier_abs = CTR_CLASS(T, num_iters,imsize=shp)
O_est = torch.conj(obj_fourier_angle) * obj_fourier_abs
PSF_std = fftshift(PSF_std)
# O_est = fftshift(ifft2(fftshift(field)))
# O_est = fftshift(fft2((O_est)))
# O_est = ifft2(fftshift(fft2(O_est)))

plt.figure()
plt.imshow(np.abs(PSF_std))
plt.show(block=False)
plt.figure()
plt.imshow(O_est.abs())
plt.title('O_est')
plt.show(block=False)

O_est_numpy = O_est.numpy()
# Determine if the fields are from 'original_fields' or 'normalized_original_fields'
field_source = 'norm' if 'normalized_original_fields' in pth else 'original'
# Create the base filename
base_filename = f"CLASS Reconstruction {num_fields} fields and {num_iters} itr from {field_source} fields"
if additional_fiff:
    base_filename += " with additional diffuser"

# Get a unique filename
name_of_file = get_unique_filename(saving_pth, base_filename, ".npy")

print(f"Saving file as: {name_of_file}")
file_path = os.path.join(saving_pth, name_of_file)

# Save the file
np.save(file_path, O_est_numpy)
print(f"File saved successfully at: {file_path}")


mxidx_psf = np.unravel_index(torch.argmax(PSF_std.abs()),PSF_std.shape)
# Calculate the shifts needed to center the PSF
shift_cols_psf = PSF_std.shape[1] // 2 - mxidx_psf[1].item()
shift_rows_psf = PSF_std.shape[0] // 2 - mxidx_psf[0].item()
# shift_cols_psf = PSF.shape[1] // 2 - 106
# shift_rows_psf = PSF.shape[1] // 2 - 128
# Shift the PSF to center it
psf_centered = torch.roll(PSF_std, shifts=(shift_rows_psf, shift_cols_psf), dims=(0, 1))
plt.figure()
plt.imshow(np.abs(psf_centered))
plt.show()
file_path = os.path.join(saving_pth, 'psf_centered')
np.save(file_path, psf_centered)

#
# O_est_centered = torch.roll(O_est, shifts=(shift_rows_psf, shift_cols_psf), dims=(0, 1))
# plt.figure()
# plt.imshow(O_est_centered.abs())
# plt.show()

a = 5