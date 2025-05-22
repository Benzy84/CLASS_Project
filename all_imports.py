import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.visualization import display_field, phplot, get_custom_colormap
from utils.field_utils import gauss2D
from utils.image_processing import fourier_convolution
from utils.io import load_file_to_tensor
from utils.image_processing import compute_similarity_score, shift_cross_correlation
from propagation.propagation import angular_spectrum_gpu
import imageio.v2 as imageio
# Add other common imports

# Create convenient aliases if needed
nwe_cmap = get_custom_colormap()