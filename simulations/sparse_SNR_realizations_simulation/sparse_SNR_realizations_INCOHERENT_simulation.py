import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from torchvision.transforms import CenterCrop
from core.CTRCLASS import CTR_CLASS
from utils.field_utils import gauss2D, circ
from utils.field_utils import create_points_obj
from utils.image_processing import shift_cross_correlation, fourier_convolution
from scipy.ndimage import shift
from torch.fft import *
import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pointsObj(imsize, r, n):
    a = torch.zeros(2 * [imsize])
    for k in range(n):
        a += gauss2D(r, size=imsize, mid=(np.random.rand(2) - 0.5) * imsize)
    return a



# Choose an existing colormap
from matplotlib.colors import LinearSegmentedColormap

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
new_cmap = LinearSegmentedColormap.from_list('greenish_hot', list(zip(positions, colors)))


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def confocal(N):
    indices = torch.arange(N ** 2)
    row_indices = indices // N
    col_indices = indices % N
    S = torch.zeros(N ** 2, N, N)
    S[indices, row_indices, col_indices] = 1
    return S



postprocess = lambda x: x
nrm = lambda x: x / x.abs().max()
tounit8 = lambda x: (255 * nrm(x)).numpy().astype('uint8')

print(f'GPU activated {torch.cuda.is_available()}')

# Simulation parameters
d_corr = 10  # [px]
sz = 350
imsize = [sz, sz]
padding_size = 40
pi = torch.pi
dist = torch.fft.ifft2(torch.fft.ifftshift(
    circ(sz // d_corr, sz) * torch.fft.fftshift(torch.fft.fft2(torch.exp(2j * pi * torch.rand((sz, sz)))))))
dist /= dist.abs()
# dist *= circ(sz // 4, sz)

PSF = torch.fft.ifft2(torch.fft.ifftshift(dist)).abs() ** 2
OTF = nrm(torch.fft.fft2(PSF))
tounit8 = lambda x: (255 * nrm(x)).numpy().astype('uint8')

ns = 2**np.arange(17)
ns = ns[::2]
ns = ns[:-1]

Ms = 10 * (2**np.array([0, 1, 2, 3, 4, 5, 6]))
Ms = 10 * (2**np.array([6]))
# Ms = 10 * (2**np.array([0]))

SNRs = [1, 3, 10, 100, torch.inf]
SNRs = [torch.inf]

trials = 10
total_iterations = len(SNRs) * len(Ms) * len(ns) * trials
completed_iterations = 0

print(f'ns: {ns}')
print(f'Ms: {Ms}')
print(f'SNRs: {SNRs}')


def makedir(dr):
    if not os.path.isdir(dr):
        os.makedirs(dr)


speckle_size, d_corr = 4, 20

# Create a timestamped main results directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
main_results_dir = f"incoherent_results_{timestamp}"
makedir(main_results_dir)

# Create a README file with simulation parameters
with open(os.path.join(main_results_dir, "README.txt"), "w") as f:
    f.write(f"CLASS Incoherent Light Simulation Results\n")
    f.write(f"======================================\n\n")
    f.write(f"Simulation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"Image size: {sz}x{sz} pixels\n")
    f.write(f"Padding size: {padding_size} pixels\n")
    f.write(f"Speckle size: {speckle_size}\n")
    f.write(f"Correlation distance: {d_corr}\n")
    f.write(f"Number of points (n): {ns.tolist()}\n")
    f.write(f"Number of realizations (M): {Ms.tolist()}\n")
    f.write(f"SNR values: {[str(snr) if snr != torch.inf else 'inf' for snr in SNRs]}\n")
    f.write(f"Number of trials: {trials}\n\n")

    # Only document files we're actually creating
    f.write(f"File Structure:\n")
    f.write(f"  - Parameter files in main directory:\n")
    f.write(f"    - ns.npy: Array of n values (number of points)\n")
    f.write(f"    - Ms.npy: Array of M values (number of realizations)\n")
    f.write(f"    - SNRs.npy: Array of SNR values\n")
    f.write(f"  - Each SNR has its own subdirectory with the following files:\n")
    f.write(f"    - For each trial T, number of points N, and realizations M:\n")
    f.write(f"      - gt_T_N.npy: Ground truth object\n")
    f.write(f"      - widefield_T_N.npy: Widefield reference image\n")
    f.write(f"      - I_T_N_M.npy: Aberrated image with M realizations (incoherent) OR\n")
    f.write(f"      - E_T_N_M.npy: Aberrated field (coherent)\n")
    f.write(f"      - PSF_T_N_M.npy: Point spread function\n")
    f.write(f"      - O_est_T_N_M.npy: Estimated object after reconstruction\n")

# Save parameter arrays in the main directory for reference
np.save(os.path.join(main_results_dir, "ns.npy"), ns)
np.save(os.path.join(main_results_dir, "Ms.npy"), Ms)
np.save(os.path.join(main_results_dir, "SNRs.npy"),
        np.array([float(snr) if snr != torch.inf else float('inf') for snr in SNRs]))

clean_PSF = fftshift(fft2(gauss2D(sz // (2 * speckle_size), sz)).abs() ** 2).to(device)
clean_PSF /= clean_PSF.sum()

for SNR in SNRs:
    # Create SNR-specific subdirectory
    snr_dir = os.path.join(main_results_dir, f"SNR_{SNR if SNR != torch.inf else 'inf'}")
    makedir(snr_dir)

    for trial in range(trials):
        tri = trial + 1

        # Generate PSFs for this trial
        dist = ifft2(ifftshift(
            gauss2D(sz // d_corr, sz) * fftshift(fft2(torch.exp(2j * np.pi * torch.rand((Ms[-1], sz, sz)))))))
        dist /= dist.abs()
        dist *= gauss2D(sz // (2 * speckle_size), sz)
        PSFs = fftshift(fft2(dist).abs() ** 2)
        psf_sums = PSFs.sum(dim=(-2, -1), keepdim=True)  # shape will be (N,1,1)
        # Avoid division by zero if any sum is 0 (unlikely, but just in case)
        psf_sums = torch.clamp(psf_sums, min=1e-20)
        # Normalize each PSF slice so it sums to 1
        PSFs_normalized = PSFs / psf_sums
        PSFs_normalized = PSFs_normalized.to(device)

        for n in ns:
            gt = torch.nn.functional.pad(create_points_obj(sz - 2 * padding_size, n), 4 * [padding_size])
            gt_Intensity = gt / gt.sum()
            gt_Intensity = gt_Intensity.to(device)
            gt_amp = torch.sqrt(gt_Intensity)
            I = fourier_convolution(gt_Intensity, PSFs_normalized).abs()
            num_pixels = sz ** 2

            # Calculate the photon budget needed so that the average photon count per pixel gives the desired SNR
            N_photons = SNR ** 2 * num_pixels
            I_counts = I * N_photons
            I_noisy_counts = torch.poisson(I_counts)
            I_noisy = I_noisy_counts / N_photons
            if SNR == torch.inf:
                I_noisy = I

            # # Rough local estimate over the whole image:
            # noise_std_est = (I_noisy[0] - I[0]).std()
            # mean_signal = I[0].mean()
            # estimated_SNR = mean_signal / noise_std_est
            # print("Estimated per-pixel SNR (global average):", estimated_SNR.item())

            widefield = fourier_convolution(gt_Intensity, clean_PSF).abs()
            widefield = widefield.to(device)

            # Save ground truth and widefield to SNR directory (only once per n)
            np.save(os.path.join(snr_dir, f'gt_{tri}_{n}.npy'), gt.numpy())
            np.save(os.path.join(snr_dir, f'widefield_{tri}_{n}.npy'), widefield.cpu().numpy())

            M = I.shape[0] if len(I.shape) > 2 else 1  # Number of realizations
            rM = 1 + (torch.arange(1, M + 1, device='cuda') / M)  # Shape: [150]


            # Multiply each field by its corresponding weight
            I_noisy_modulated = I_noisy * rM[:, None, None]  # Shape: [150, 300, 300]

            Icam_fft = torch.fft.fftshift((torch.fft.fft2(I_noisy_modulated - I_noisy_modulated.mean(0))))
            torch.cuda.empty_cache()
            shp = I.shape[-1] * I.shape[-2]

            T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).cuda()

            for M_idx, M in enumerate(Ms):
                print(f'SNR = {SNR} trial = {tri} n = {n} M = {M}')
                print(f'Progress: {completed_iterations + 1}/{total_iterations} iterations completed')

                # Run reconstruction
                _, _, phi_tot, MTF = CTR_CLASS(T[:, :M], num_iters=1000)

                # Get result
                O_est_0 = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
                O_est = shift_cross_correlation(widefield, O_est_0.real).cpu().numpy() # O_est will be only real from this line
                O_est /= np.abs(O_est).max()

                # Save results to SNR directory
                np.save(os.path.join(snr_dir, f'I_{tri}_{n}_{M}.npy'), I[M - 1].cpu().numpy())
                np.save(os.path.join(snr_dir, f'PSF_{tri}_{n}_{M}.npy'), PSFs[M - 1].cpu().numpy())
                np.save(os.path.join(snr_dir, f'O_est_{tri}_{n}_{M}.npy'), O_est)

                # Update the completed iterations counter
                completed_iterations += 1

print(f"Simulation complete! Results saved to {main_results_dir}")

# Clean up GPU memory
del T
torch.cuda.empty_cache()
# import CLASSSimSparseIterationsDynamicCoh

