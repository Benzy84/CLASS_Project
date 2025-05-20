import matplotlib
from numpy.fft.helper import fftshift
from torch.fft import ifftshift
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from utils import *
import cv2
from torchvision.transforms import  CenterCrop
from CTRCLASS import CTR_CLASS
from utils import imshow,getObject,amp_and_hue
from utils import gauss2D
from scipy.ndimage import shift
from torch.fft import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import datetime
from torchvision.transforms import Resize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def pointsObj(imsize,r,n):
    a = torch.zeros(2*[imsize])
    for k in range(n):
        a += gauss2D(r, size = imsize,mid = (np.random.rand(2)-0.5)*imsize)
    return a



# Choose an existing colormap
from matplotlib.colors import LinearSegmentedColormap
# Define the colors for the colormap in RGB format
colors = [
    (0, 0, 0),    # Black
    (0, 0.2, 0),  # Dark Green
    (0, 0.5, 0),  # Green
    (0, 0.8, 0),  # Bright Green
    (0.7, 1, 0),  # Light Green-Yellow
    (1, 1, 1)     # White
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

postprocess = lambda x:x
nrm = lambda x:x/x.abs().max()
tounit8 = lambda x:(255*nrm(x).numpy()).astype('uint8')

print(f'GPU activated {torch.cuda.is_available()}')


d_corr = 20 # [px]
sz = 350
imsize = [sz, sz]
padding_size = 40


dist = torch.fft.ifft2(torch.fft.ifftshift(
    circ(sz // d_corr, sz) * torch.fft.fftshift(torch.fft.fft2(torch.exp(2j * pi * torch.rand((sz, sz)))))))
dist /= dist.abs()
# dist *= circ(sz // 4, sz)
tounit8 = lambda x:(255*nrm(x)).numpy().astype('uint8')

ns = 2**np.arange(17)
ns = ns[::2]
ns = ns[:-1]

Ms = 10 * (2**np.array([0, 1, 2, 3, 4, 5, 6]))
Ms = 10 * (2**np.array([5]))
# Ms[0] = 1


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
        os.mkdir(dr)

speckle_size,d_corr = 4,20

# Create a timestamped main results directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
main_results_dir = f"coherent_results_{timestamp}"
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


clean_PSF = fftshift(fft2(ifftshift(gauss2D(sz // (2 * speckle_size), sz))))
clean_PSF /= (clean_PSF.abs()**2).sum()**0.5


for SNR in SNRs:
    # Create SNR-specific subdirectory
    snr_dir = os.path.join(main_results_dir, f"SNR_{SNR if SNR != torch.inf else 'inf'}")
    makedir(snr_dir)

    for trial in range(trials):
        tri = trial+1

        dist = ifft2(ifftshift(
            gauss2D(sz // d_corr, sz) * fftshift(fft2(torch.exp(2j * np.pi * torch.rand((Ms[-1], sz, sz)))))))
        dist /= dist.abs()
        dist *= gauss2D(sz // (2 * speckle_size), sz)
        PSFs = fftshift(fft2(dist))
        psf_sums = (PSFs.abs()**2).sum(dim=(-2, -1), keepdim=True)  # shape will be (N,1,1)
        # Avoid division by zero if any sum is 0 (unlikely, but just in case)
        psf_sums = torch.clamp(psf_sums, min=1e-20)
        # Normalize each PSF slice so it sums to 1
        PSFs_normalized = PSFs / torch.sqrt(psf_sums)

        for n in ns:
            gt = torch.nn.functional.pad(create_points_obj(sz - 2 * padding_size,n), 4*[padding_size])

            gt_Intensity = gt/gt.sum()
            gt_amp = torch.sqrt(gt_Intensity)
            E = fourier_convolution(gt_amp, PSFs_normalized)
            I = E.abs()**2
            # I = fourier_convolution(gt_Intensity, PSFs_normalized.abs()**2).abs()

            num_pixels = sz ** 2
            # Calculate the photon budget needed so that the average photon count per pixel gives the desired SNR.
            # Since per-pixel SNR ≈ sqrt(average photon count), we set:
            N_photons = SNR ** 2 * num_pixels
            I_counts = I * N_photons
            I_noisy_counts = torch.poisson(I_counts)
            I_noisy = I_noisy_counts / N_photons
            E_noisy_amp = torch.sqrt(I_noisy)
            sigma_phase = 1/SNR
            phase_noise = torch.normal(0, sigma_phase, size=E.shape)
            E_noisy_phase = torch.angle(E) + phase_noise
            # Add noise to the field
            E_noisy = E_noisy_amp * torch.exp(1j*E_noisy_phase)

            # If SNR is infinite, no noise
            if SNR == torch.inf:
                E_noisy = E

            # Rough local estimate over the whole image:
            noise_std_est = (E_noisy[1].abs()**2 - I[1]).std()
            mean_signal = I[1].mean()
            estimated_SNR = mean_signal / noise_std_est
            print("Estimated per-pixel SNR (global average):", estimated_SNR.item())
            I_noisy = E_noisy.abs() ** 2
            noise_intensity = I_noisy - I
            actual_SNR = I.mean() / noise_intensity.std()

            print(f"Target SNR: {SNR}, Actual SNR: {actual_SNR.item()}")

            # # Load and prepare object
            # P = load_file_to_tensor().cpu()
            # resize = Resize((270, 270), antialias=True)
            # P = resize(P.unsqueeze(0)).squeeze(0)
            # padding_size = (sz - 270) // 2
            # P = torch.nn.functional.pad(P, 4 * [padding_size])
            #
            # # Generate reference widefield image
            # widefield = fourier_convolution(P, clean_PSF).to(device)

            widefield = fourier_convolution(gt_amp, clean_PSF).to(device)

            # Save ground truth and widefield to SNR directory
            np.save(os.path.join(snr_dir, f'gt_{tri}_{n}.npy'), gt.numpy())
            np.save(os.path.join(snr_dir, f'widefield_{tri}_{n}.npy'), widefield.cpu().numpy())

            Icam_fft = torch.fft.fftshift((torch.fft.fft2(E_noisy)))
            torch.cuda.empty_cache()
            shp = E_noisy.shape[-1] * E_noisy.shape[-2]

            T = torch.permute(Icam_fft, [2, 1, 0]).reshape(shp, -1).cuda()

            for M_idx, M in enumerate(Ms):

                print(f'SNR = {SNR} trial = {tri} n = {n} M = {M}')
                print(f'Progress: {completed_iterations+1}/{total_iterations} iterations completed')

                # Run CLASS algorithm
                _, _, phi_tot, MTF = CTR_CLASS(T[:,:M], num_iters=1000)

                # Process result
                O_est_0 = ifft2(ifftshift(torch.conj(phi_tot) * MTF))
                O_est = shift_cross_correlation(widefield, O_est_0).cpu().numpy()
                O_est /= np.abs(O_est).max()

                # Save results to SNR directory
                np.save(os.path.join(snr_dir, f'E_{tri}_{n}_{M}.npy'), E_noisy[M-1].cpu().numpy())
                np.save(os.path.join(snr_dir, f'PSF_{tri}_{n}_{M}.npy'), PSFs[M-1].cpu().numpy())
                np.save(os.path.join(snr_dir, f'O_est_{tri}_{n}_{M}.npy'), O_est)


                # if M == 90:
                #     plt.figure()
                #
                #     plt.subplot(2, 2, 1)
                #     plt.title('Ground Truth')
                #     plt.imshow(C(gt).cpu().numpy(), cmap='gray')
                #     plt.axis('off')
                #
                #     plt.subplot(2, 2, 2)
                #     plt.title('Widefield')
                #     plt.imshow(np.abs(widefield), cmap='gray')
                #     plt.axis('off')
                #
                #     plt.subplot(2, 2, 3)
                #     plt.title('aberrated')
                #     plt.imshow(np.abs(I[M-1]), cmap='gray')
                #     plt.axis('off')
                #
                #     plt.subplot(2, 2, 4)
                #     plt.title('CLASS Reconstruction')
                #     plt.imshow(np.abs(O_est), cmap='gray')
                #     plt.axis('off')
                #
                #     plt.tight_layout()
                #     plt.show()

                completed_iterations += 1

del T
torch.cuda.empty_cache()
# import CLASSSimmodulation_depth_simulation