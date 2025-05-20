import numpy as np
import torch
from torch.fft import fft2, ifft2


def angular_spectrum_propagation(fields, pixel_size, wavelength, distance, pad_factor=2.0, verbose=False):
    """
    Angular Spectrum propagation optimized for batch processing with automated frequency filtering.

    Parameters:
    -----------
    fields : torch.Tensor or numpy.ndarray
        Input field(s) to propagate. Can be:
        - Single field: shape (H, W)
        - Batch of fields: shape (N, H, W)
        Can be real or complex valued.
    wavelength : float
        Wavelength in meters
    pixel_size : float
        Pixel size in meters
    distance : float
        Propagation distance in meters
    pad_factor : float
        Virtual padding factor used to determine frequency cutoff
    verbose : bool
        Whether to print diagnostic information

    Returns:
    --------
    Same type and device as input (numpy.ndarray or torch.Tensor)
        Propagated field(s) with same shape as input
    """

    if distance == 0:
        return fields  # No propagation needed


    # 1. Handle input types and determine return type
    input_is_numpy = isinstance(fields, np.ndarray)
    input_device = None if input_is_numpy else fields.device
    is_complex = np.iscomplexobj(fields) if input_is_numpy else torch.is_complex(fields)

    # 2. Convert to torch tensor and move to GPU if available
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if input_is_numpy:
        if is_complex:
            fields_tensor = torch.tensor(fields, dtype=torch.complex64).to(compute_device)
        else:
            fields_tensor = torch.tensor(fields, dtype=torch.float32).to(compute_device)
    else:
        # Move to compute device if not already there
        fields_tensor = fields.to(compute_device)

    # Clean up memory after conversion
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Handle batch vs single field
    if fields_tensor.dim() == 2:
        # Single field - add batch dimension
        fields_tensor = fields_tensor.unsqueeze(0)
        was_single = True
    else:
        was_single = False

    # Get dimensions
    batch_size, H, W = fields_tensor.shape

    # Step 1: Calculate frequency limits
    # -----------------------------
    # Nyquist limit (aliasing prevention)
    f_nyquist = 1 / (2 * pixel_size)

    # Wrapping limit (based on virtual padding)
    physical_width = W * pixel_size
    f_wrap = physical_width * (pad_factor - 1) / (2 * abs(distance) * wavelength)

    # Choose the more restrictive limit
    f_limit = min(f_nyquist, f_wrap)

    # Notify user if wrapping is the limiting factor
    if verbose:
        print(f"Frequency limits (cycles/m):")
        print(f"  Nyquist limit: {f_nyquist:.2e}")
        print(f"  Wrapping limit: {f_wrap:.2e} (with pad_factor={pad_factor})")

        if f_wrap < f_nyquist:
            recommended_pad = 1 + (2 * abs(distance) * wavelength * f_nyquist) / physical_width
            print(f"  ⚠️ Wrapping limit is more restrictive than Nyquist")
            print(f"  💡 To preserve all frequencies up to Nyquist, use pad_factor ≥ {recommended_pad:.2f}")
        else:
            print(f"  ✓ Using Nyquist-limited filtering")

    # Step 2: Create frequency grid (done once for all fields)
    # -----------------------------
    fx = torch.fft.fftfreq(W, pixel_size).to(compute_device)
    fy = torch.fft.fftfreq(H, pixel_size).to(compute_device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    f_mag = torch.sqrt(FX ** 2 + FY ** 2)

    # Step 3: Create combined filter with smooth transition
    # -----------------------------
    cutoff = 0.95 * f_limit
    filter_order = 8  # Higher = sharper cutoff
    combined_filter = torch.exp(-(f_mag / cutoff) ** (2 * filter_order))

    # Clean temporary tensors
    del fx, fy, f_mag
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Step 4: Create Angular Spectrum transfer function with robust error handling
    # -----------------------------
    k = 2 * np.pi / wavelength

    # Use a small epsilon to prevent divide-by-zero or numerical issues
    epsilon = 1e-10

    # Calculate kz_squared with safeguards
    kz_squared = k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2

    # Handle evanescent waves correctly and safely
    propagating = kz_squared > epsilon  # Use epsilon instead of 0 for numerical stability

    # Initialize transfer function with zeros
    transfer = torch.zeros_like(kz_squared, dtype=torch.complex64)

    # Only calculate for propagating waves to avoid issues with sqrt of negative numbers
    if propagating.any():
        # Create a safe version of kz that avoids numerical issues
        # Use torch.clamp to ensure we don't take sqrt of negative values
        safe_kz = torch.sqrt(torch.clamp(kz_squared[propagating], min=epsilon))

        # Assign values only to propagating positions
        transfer[propagating] = torch.exp(1j * safe_kz * distance)

        # Check for any remaining NaN or inf values and replace them
        invalid_values = ~torch.isfinite(transfer)
        if invalid_values.any():
            if verbose:
                print(
                    f"⚠️ Detected {invalid_values.sum().item()} non-finite values in transfer function. Replacing with zeros.")
            transfer[invalid_values] = 0.0

    # Apply filter to transfer function (yes, equivalent to filtering field)
    filtered_transfer = transfer * combined_filter

    # Clean temporary tensors
    del FX, FY, kz_squared, propagating, transfer, combined_filter
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Make transfer function have same dimensions as fields tensor for efficient broadcasting
    filtered_transfer = filtered_transfer.unsqueeze(0)  # Add batch dimension to match fields

    # Step 5: Propagate all fields at once
    # -----------------------------
    # Convert to complex if needed
    if not is_complex:
        fields_tensor = fields_tensor.to(torch.complex64)

    # FFT of all fields
    fields_fft = torch.fft.fft2(fields_tensor)

    # Apply filtered transfer function to all fields using broadcasting
    result_fft = fields_fft * filtered_transfer

    # Clean temporary tensors
    del fields_fft, filtered_transfer
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()

    # IFFT to get results
    result = torch.fft.ifft2(result_fft)

    # Check for any NaN or inf values in the result
    if not torch.isfinite(result).all():
        if verbose:
            print("⚠️ Warning: Non-finite values detected in the result. Replacing with zeros.")
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Clean temporary tensors
    del result_fft
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Step 6: Handle output type
    # -----------------------------
    # Convert back to real if input was real
    if not is_complex:
        result = result.real

    # Remove batch dimension if input was single field
    if was_single:
        result = result.squeeze(0)

    # Convert back to original type and device
    if input_is_numpy:
        result = result.cpu().numpy()
    else:
        result = result.to(input_device)  # Move back to original device

    return result

def angular_spectrum_gpu(fields, pixel_size, wavelength, distance,
                         max_pad_factor=1.0, filtering='none', verbose=False):
    """
    GPU-optimized Angular Spectrum propagation with automatic memory management.

    Automatically:
    1. Determines optimal padding based on physics (no arbitrary limit)
    2. Uses GPU if available regardless of input device
    3. Splits processing into optimal batches if needed
    4. Returns result in same form and on same device as input

    Parameters:
    -----------
    fields : torch.Tensor or numpy.ndarray
        Input field(s) - single (H,W) or batch (N,H,W)
    pixel_size : float
        Pixel size in meters
    wavelength : float
        Wavelength in meters
    distance : float
        Propagation distance in meters
    filtering : str
        'none', 'adaptive', or 'nyquist'
    max_pad_factor : float
        Upper limit on padding factor
    verbose : bool
        Whether to print diagnostic information

    Returns:
    --------
    Same type and device as input
        Propagated field(s), same shape as input
    """
    # Save input properties to restore later
    input_is_numpy = isinstance(fields, np.ndarray)
    input_device = None if input_is_numpy else fields.device

    # Early return for z=0
    if distance == 0:
        return fields

    # Convert to torch tensor
    if input_is_numpy:
        fields_tensor = torch.tensor(fields)
    else:
        fields_tensor = fields

    # Normalize input dimensions
    original_shape = fields_tensor.shape
    was_single = len(original_shape) == 2
    if was_single:
        fields_tensor = fields_tensor.unsqueeze(0)

    # Use GPU if available, regardless of input device
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fields_tensor = fields_tensor.to(compute_device)

    # Get field shape and type
    batch_size, h, w = fields_tensor.shape

    # Step 1: Calculate physically optimal padding (NO ARBITRARY LIMIT)
    physics_pad = 1 + (abs(distance) * wavelength) / (pixel_size ** 2 * w)
    # Ensure reasonable minimum
    physics_pad = min(max(physics_pad, 1.0), max_pad_factor)

    if verbose:
        print(f"Physics-optimal padding factor: {physics_pad:.2f}x")

    # Step 2: Calculate memory requirements
    memory_per_field = estimate_field_memory(h, w, physics_pad, fields_tensor.dtype)
    total_memory_needed = memory_per_field * batch_size

    # Check available GPU memory
    if compute_device.type == 'cuda':
        available_memory = torch.cuda.get_device_properties(compute_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(compute_device)
        free_memory = available_memory - allocated_memory

        # Reserve 10% of memory as buffer
        usable_memory = free_memory * 0.9

        if verbose:
            print(f"Memory estimate: {total_memory_needed / 1e9:.2f} GB needed, {usable_memory / 1e9:.2f} GB available")

        # Step 3: Determine number of batches needed
        if total_memory_needed > usable_memory:
            # Calculate how many splits we need
            num_splits = int(np.ceil(total_memory_needed / usable_memory))
            # Calculate batch size per split
            batch_size_per_split = batch_size // num_splits
            # Ensure at least one field per batch
            batch_size_per_split = max(1, batch_size_per_split)

            if verbose:
                print(f"Splitting into {num_splits} batches with {batch_size_per_split} fields each")

            # Step 4: Process each split
            result_chunks = []
            for i in range(0, batch_size, batch_size_per_split):
                end_idx = min(i + batch_size_per_split, batch_size)
                chunk = fields_tensor[i:end_idx]

                if verbose:
                    print(f"Processing batch {i // batch_size_per_split + 1}/{num_splits}...")

                # Process this chunk with the physics-optimal padding
                # We know it will fit in memory because we calculated the batch size
                result_chunk = perform_angular_spectrum(
                    chunk, pixel_size, wavelength, distance,
                    physics_pad, filtering, verbose
                )
                result_chunks.append(result_chunk)

                # Explicitly free memory
                if compute_device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Step 5: Combine results
            result = torch.cat(result_chunks, dim=0)
        else:
            # No splitting needed, process the whole batch
            result = perform_angular_spectrum(
                fields_tensor, pixel_size, wavelength, distance,
                physics_pad, filtering, verbose
            )
    else:
        # CPU processing doesn't need splitting
        result = perform_angular_spectrum(
            fields_tensor, pixel_size, wavelength, distance,
            physics_pad, filtering, verbose
        )

    # Return in original format
    if was_single:
        result = result.squeeze(0)

    # Convert back to original device and type
    if input_is_numpy:
        result = result.cpu().numpy()
    else:
        result = result.to(device=input_device)

    return result


def estimate_field_memory(h, w, pad_factor, dtype=torch.complex64):
    """Estimate memory needed to process a single field with padding"""
    h_pad = int(h * pad_factor)
    w_pad = int(w * pad_factor)

    # Bytes per element
    bytes_per_element = 8 if dtype == torch.complex64 else 16  # complex64 or complex128

    # Major tensors per field
    memory_padded_field = h_pad * w_pad * bytes_per_element
    memory_freq_grids = 2 * h_pad * w_pad * 4  # FX, FY grids (float32)
    memory_transfer_function = h_pad * w_pad * bytes_per_element
    memory_fft_input = h_pad * w_pad * bytes_per_element
    memory_fft_output = h_pad * w_pad * bytes_per_element
    memory_result = h_pad * w_pad * bytes_per_element

    # Total with 10% overhead
    total = (memory_padded_field + memory_freq_grids + memory_transfer_function +
             memory_fft_input + memory_fft_output + memory_result)
    total = int(total * 1.1)  # Add 10% for miscellaneous allocations

    return total



def perform_angular_spectrum(fields, pixel_size, wavelength, distance,
                         pad_factor, filtering, verbose):
    """Process a single batch that is guaranteed to fit in memory"""
    batch_size, h, w = fields.shape
    device = fields.device

    # 1. Apply padding
    pad_size = int((pad_factor - 1) * h / 2)
    padded_fields = torch.nn.functional.pad(fields, (pad_size, pad_size, pad_size, pad_size))
    h_pad, w_pad = padded_fields.shape[-2:]

    # 2. Create frequency coordinates
    fx = torch.fft.fftfreq(w_pad, d=pixel_size).to(device)
    fy = torch.fft.fftfreq(h_pad, d=pixel_size).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    f_mag = torch.sqrt(FX ** 2 + FY ** 2)

    # 3. filtering
    if filtering == 'none':
        # No filtering (like diffractsim)
        filter_mask = torch.ones_like(f_mag)
    elif filtering == 'adaptive':
        # Adaptive filtering to prevent wrapping
        f_limit = pad_factor * w / (2 * wavelength * abs(distance))
        filter_order = 8
        filter_mask = torch.exp(-(f_mag / (0.9 * f_limit)) ** (2 * filter_order))
    elif filtering == 'nyquist':
        # Nyquist filtering to prevent aliasing
        f_nyquist = 1 / (2 * pixel_size)
        filter_order = 8
        filter_mask = torch.exp(-(f_mag / (0.9 * f_nyquist)) ** (2 * filter_order))
    else:
        raise ValueError(f"Unknown filtering option: {filtering}")

    # 4. Create complete transfer function
    fsq = (FX ** 2 + FY ** 2)
    H_z = torch.exp(1j * distance * 2 * torch.pi * torch.sqrt(torch.clamp(1 / wavelength ** 2 - fsq, min=0.0)))

    # 6. FFT
    fields_fft = torch.fft.fft2(padded_fields)

    # 7. Apply transfer function (properly align with FFT)
    H_expanded = H_z.unsqueeze(0)  # Add batch dimension
    result_fft = fields_fft * H_expanded

    # 8. IFFT
    result_padded = torch.fft.ifft2(result_fft)

    # 9. Crop back to original size
    start_h = (h_pad - h) // 2
    start_w = (w_pad - w) // 2
    result = result_padded[:, start_h:start_h + h, start_w:start_w + w]


    # Clean up
    del padded_fields, fields_fft, result_fft, result_padded, H_z, H_expanded
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


def perform_angular_spectrum2(fields, pixel_size, wavelength, distance, pad_factor = 1, filtering = "none", verbose=False):
    """
    Angular Spectrum Method (ASM) for scalar field propagation.

    Parameters:
    -----------
    field : np.ndarray
        Complex input field (2D array).
    pixel_size : float
        Pixel size in meters.
    wavelength : float
        Wavelength in meters.
    distance : float
        Propagation distance in meters.

    Returns:
    --------
    np.ndarray
        Propagated complex field (same shape as input).
    """
    if distance == 0:
        return fields.copy()

    # cheack if GPU id available
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Save input properties to restore later
    input_is_numpy = isinstance(fields, np.ndarray)
    input_device = None if input_is_numpy else fields.device

    # Convert to torch tensor
    if input_is_numpy:
        fields_tensor = torch.tensor(fields)
    else:
        fields_tensor = fields

    fields = fields_tensor
    del fields_tensor
    # Normalize input dimensions
    original_shape = fields.shape
    was_single = len(original_shape) == 2
    if was_single:
        fields = fields.unsqueeze(0)

    fields = fields.to(compute_device)

    batch_size, H, W = fields.shape
    k = 2 * np.pi / wavelength

    fx = torch.fft.fftfreq(W, d=pixel_size) # frequency coordinates
    fy = torch.fft.fftfreq(H, d=pixel_size)
    FX, FY = torch.meshgrid(fx, fy)

    # Angular spectrum transfer function
    fsq = FX ** 2 + FY ** 2
    H_z = np.exp(1j * distance * 2 * torch.pi * np.sqrt(np.maximum(0, 1 / wavelength ** 2 - fsq))).to(compute_device)

    # Forward FFT
    F_field = fft2(fields)
    # Multiply by transfer function
    F_propagated = F_field * H_z
    # Inverse FFT to get back to spatial domain
    propagated_field = ifft2(F_propagated)

    # Return in original format
    if was_single:
        propagated_field = propagated_field.squeeze(0)

    # Convert back to original device and type
    if input_is_numpy:
        propagated_field = propagated_field.cpu().numpy()
    else:
        propagated_field = propagated_field.to(device=input_device)


    return propagated_field
