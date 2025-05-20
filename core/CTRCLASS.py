import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import *



# Convolution using FFT
def fftconv(A, B, mode='same'):
    if mode in ['Full','full','pad','Pad']:
        # Calculate the size of the padded output
        out_size = A.size(-2) + B.size(-2) - 1, A.size(-1) + B.size(-1) - 1

        # Perform 2D FFT on both input matrices and multiply them
        f = fft2(A, s=out_size)
        f *= fft2(B, s=out_size)
    else:
        f = fft2(A)
        f *= fft2(B)
    # Perform inverse 2D FFT on the result and normalize
    return (ifft2(f)) / (torch.numel(f))


# Cross-Correlation using FFT
def fftcorr(A, B, mode='same'):
    if mode in ['Full','full','pad','Pad']:
        # Calculate the size of the padded output
        out_size = A.size(-2) + B.size(-2) - 1, A.size(-1) + B.size(-1) - 1

        # Perform 2D FFT on both input matrices and multiply them (with conjugate for B)
        f = fft2(A, s=out_size)
        f *= (fft2(B, s=out_size).conj())
    else:
        f = fft2(A)
        f *= (fft2(B).conj())

    # Perform inverse 2D FFT on the result and normalize
    return ifft2(f) / (torch.numel(f))


def TtoO(T, imsize, real=False):
    # return 3
    # Compute object
    O = torch.sqrt(torch.mean(ifft2(T).abs() ** 2, 1)).reshape(*imsize).cpu()
    torch.cuda.empty_cache()
    return O


def CTR_CLASS(T: torch.Tensor,num_iters = 100,save_path=None,save_name='0',save_every=10000,imsize = None,real=False,pos=False,device = None):

    # Determine which device to use for computations
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    T = T.to(device)

    # Set default save path if not provided, to current folder
    if save_path is None:
        save_path = os.getcwd()

    # Matrix and output image sizes
    N,M = T.shape
    if imsize is None:
        imsize = 2*[int(N**.5)]


    O0 = TtoO(T,imsize,real=real) # Estimate object before starting
    np.save(os.path.join(save_path,f'Oest0_{save_name}.npy'), O0)

    # Set initial phase mask to 0
    Cnv, OTF, phi_tot = 3*[torch.ones(N, dtype=torch.complex64,device=device)]

    # Iterative CLASS algorithm

    for k in range(1,1+num_iters):
        # Compute OTF estimation
        Cnv = fftconv(torch.conj(T.flipud()), T.fliplr())[:, M - 1]
        OTF = torch.mean((T.conj()) * torch.roll(fftcorr(T.flipud(), Cnv.unsqueeze(1).conj()).flipud(), 1, 0)[:N, :M], 1)

        phi = torch.exp(1j*(OTF.angle())) # Take only phases

        # if pos:
        #     phi = ifft2(torch.nn.functional.relu(fft2(phi.reshape(*imsize)).real)).flatten()
        #     phi /=phi.abs()

        # Update phase mask and correct matrix
        phi_tot *= phi
        T = (phi * T.T).T

        # if k % 100 == 0:
        #     # torch.cuda.empty_cache()
        #     print(f'Iteration {k}/{num_iters}')

        # Save every (save_every) iterations
        # if k % save_every == 0:
        #     O_est = TtoO(T,imsize,real=real).abs()
        #     np.save(os.path.join(save_path,f'Oest_{save_name}.npy'),O_est)

    # Final estimation of the object and other outputs
    MTF = ((T.abs()**2).sum(1))
    MTF /= MTF.max()
    MTF = torch.sqrt(MTF)


    # est = torch.ones_like(MTF)
    # est[1:] = torch.cumprod(torch.sqrt(MTF[1:] / MTF[:-1]), dim=0)

    # MTF2 = (T*T.fliplr()).sum(1).abs()
    # est2 = torch.ones_like(MTF2)
    # est2[1:] = torch.cumprod(torch.sqrt(MTF2[1:] / MTF2[:-1]), dim=0)


    # OTF = (est.cpu())
    # O_est = TtoO(T,imsize,real=real)
    # np.save(os.path.join(save_path, f'Oest_{save_name}.npy'), O_est)
    # np.save(os.path.join(save_path, f'OTF_{save_name}.npy'), MTF.abs().cpu())
    return None,None, phi_tot.reshape(*imsize).T, MTF.reshape(*imsize).T
    #
    # return T.cpu(), O_est, phi_tot.reshape(*imsize).T.cpu(), MTF.reshape(*imsize).T.cpu().abs()
