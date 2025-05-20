import numpy as np
import torch

def get_modulation_strategies(device=None, min_weight=0.01):
    """
    Get a dictionary of modulation strategies.

    Parameters:
    -----------
    device : torch.device, optional
        Device for the tensors (for PyTorch strategies)
    min_weight : float, default=0.01
        Minimum allowed weight to prevent zeros

    Returns:
    --------
    dict : Dictionary of modulation strategies
    """

    # Helper functions for more complex modulation patterns
    def get_quadratic_modulation(m, alpha, device, min_weight):
        x = torch.linspace(-1, 1, m, device=device)
        return 1.0 + alpha * (x ** 2 - 1.0) * (1.0 - min_weight)

    def get_bell_curve_modulation(m, alpha, device, min_weight):
        bell = torch.exp(-((torch.arange(m, device=device) - m / 2) / (m / 4)) ** 2)
        bell = bell / bell.max() * (2.0 - min_weight)
        return 1.0 + alpha * (bell - 1.0)

    def get_exponential_modulation(m, alpha, device, min_weight):
        exp_curve = (torch.exp(3 * torch.arange(m, device=device) / m) - 1) / (np.e ** 3 - 1)
        exp_curve = exp_curve * (2.0 - min_weight)
        return 1.0 + alpha * exp_curve

    def get_logarithmic_modulation(m, alpha, device, min_weight):
        log_curve = torch.log(1 + 9 * torch.arange(m, device=device) / m) / np.log(10)
        log_curve = log_curve * (2.0 - min_weight)
        return 1.0 + alpha * log_curve

    # Define modulation strategies where alpha=1 gives maximum depth without going negative
    modulation_strategies = {
        "none": lambda m, alpha: torch.ones(m, device=device),
        "linear": lambda m, alpha: 1.0 + alpha * torch.linspace(1.0 - min_weight, -1.0 + min_weight, m, device=device),
        "linear_2": lambda m, alpha: alpha * torch.linspace(1.0 - min_weight, -1.0 + min_weight, m, device=device),
        "quadratic": lambda m, alpha: get_quadratic_modulation(m, alpha, device, min_weight),
        "sinusoidal": lambda m, alpha: 1.0 + alpha * (1.0 - min_weight) * torch.sin(
            2 * np.pi * torch.arange(m, device=device) / m),
        "bell_curve": lambda m, alpha: get_bell_curve_modulation(m, alpha, device, min_weight),
        "step": lambda m, alpha: 1.0 + alpha * (
                (torch.arange(m, device=device) >= m / 2).float() * (2.0 - min_weight) - 1.0 + min_weight),
        "linear_ramp": lambda m, alpha: 1.0 + alpha * (torch.arange(m, device=device) / (m - 1)) * (2.0 - min_weight),
        "oscillating": lambda m, alpha: 1.0 + alpha * (1.0 - min_weight) * torch.cos(
            6 * np.pi * torch.arange(m, device=device) / m),
        "exponential": lambda m, alpha: get_exponential_modulation(m, alpha, device, min_weight),
        "logarithmic": lambda m, alpha: get_logarithmic_modulation(m, alpha, device, min_weight),
        "increasing_ramp": lambda m, alpha: 1.0 + (alpha - 1) * (torch.arange(0, m, device=device) / m),
        "modified_ramp": lambda m, alpha: 1.0 + (alpha-1) * (torch.arange(0, m, device=device) / m),
        "ratio_ramp": lambda m, alpha: torch.linspace(1.0, alpha, m, device=device),
        "reciprocal_ratio": lambda m, alpha: torch.linspace(1.0 / alpha, alpha, m, device=device),
    }

    return modulation_strategies

