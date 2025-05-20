import os
import time

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, TextBox
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
import datetime
from tqdm import tqdm
import torch
import concurrent.futures
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import queue
import threading
from natsort import natsorted



def check_for_reference_images(image_folder):
    ref_folder = os.path.join(os.path.dirname(image_folder), "only ref")
    if os.path.isdir(ref_folder):
        ref_files = []
        for image_type in ['*.jpg', '*.tif', '*.png', '*.npy']:
            ref_files.extend(glob.glob(os.path.join(ref_folder, image_type)))
        return True, ref_folder, ref_files
    return False, None, None

def check_for_noise_images(image_folder):
    noise_folder = os.path.join(os.path.dirname(image_folder), "only noise")
    if os.path.isdir(noise_folder):
        noise_files = []
        for image_type in ['*.jpg', '*.tif', '*.png', '*.npy']:
            noise_files.extend(glob.glob(os.path.join(noise_folder, image_type)))
        return True, noise_folder, noise_files
    return False, None, None

def crop_tensor(tensor, crop_coords):
    return tensor[:, crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]

def process_fft_tensor(fft_tensor, fft_crop_coords):
    # Ensure the input is on the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fft_tensor = fft_tensor.to(device)

    # Perform computations on GPU
    cropped_fft = crop_tensor(fft_tensor, fft_crop_coords)
    reconstructed_field = ifft2(ifftshift(cropped_fft, dim=(-2, -1)), dim=(-2, -1))

    # Move results back to CPU
    return reconstructed_field.cpu()

def perform_fft_on_tensor_individually(images_tensor, use_gpu=True):
    fft_tensors = []
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    for img in tqdm(images_tensor, desc="Processing FFTs"):
        img = img.to(device)
        fft = fftshift(fft2(img.float()))
        fft_tensors.append(fft.cpu())  # Move result back to CPU

    return torch.stack(fft_tensors)


def load_array_or_image(file_path, device):
    if file_path.endswith('.npy'):
        return torch.from_numpy(np.load(file_path)).to(device)
    else:
        image = Image.open(file_path)
        if image.mode == 'I;16':
            image_array = np.array(image, dtype=np.uint16)
            image_array = image_array.astype(np.float32)  # Convert to float32 for PyTorch
        else:
            image_array = np.array(image, dtype=np.float32)
        return torch.from_numpy(image_array).to(device)


def load_images_as_tensor(files, device):
    images = []
    # files = files[::2]
    with tqdm(total=len(files), desc="Loading images") as pbar:
        for file in files:
            image = load_array_or_image(file, device)
            images.append(image)
            pbar.update(1)
    return torch.stack(images)


def select_files_or_folder():
    image_types = ['*.jpg', '*.tif', '*.png', '*.npy']
    root = tk.Tk()
    root.withdraw()
    messagebox_result = messagebox.askquestion('Choose', 'Select a Folder?', icon='question')
    if messagebox_result == 'yes':
        folder = filedialog.askdirectory()
        if folder:
            files = []
            for image_type in image_types:
                files.extend(glob.glob(os.path.join(folder, image_type)))
            has_ref_images, ref_folder, ref_files = check_for_reference_images(folder)
            has_noise_images, noise_folder, noise_files = check_for_noise_images(folder)
            return files, folder, has_ref_images, ref_folder, ref_files, has_noise_images, noise_folder, noise_files
        return [], None, False, None, None, False, None, None
    else:
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.npy;*.tif;*.png")])
        folder = os.path.dirname(files[0]) if files else None
        has_ref_images, ref_folder, ref_files = check_for_reference_images(folder) if folder else (False, None, None)
        has_noise_images, noise_folder, noise_files = check_for_noise_images(folder)
        return files, folder, has_ref_images, ref_folder, ref_files, has_noise_images, noise_folder, noise_files


def crop_image_interactively(image_array, title='Select Crop Area', crop_coords=None):
    if crop_coords is None:
        crop_coords = (0, 0, image_array.shape[1], image_array.shape[0])
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.set_title(title)

        # Initialize variables
        cropped_image = None

        # Add a text box to show the dimensions of the selected area
        ax_box = plt.axes([0.8, 0.15, 0.15, 0.075])
        dim_text = TextBox(ax_box, 'Dimensions', initial=f"{image_array.shape[1]}x{image_array.shape[0]}")

        # Function to update the cropping rectangle
        def update_rect(eclick, erelease):
            nonlocal x1, y1, x2, y2, rect, dim_text
            dx = abs(erelease.xdata - eclick.xdata)
            dy = abs(erelease.ydata - eclick.ydata)
            span = int(min(dx, dy))  # Ensure span is an integer

            x1 = int(eclick.xdata)
            y1 = int(eclick.ydata)

            x2 = x1 + span if erelease.xdata >= eclick.xdata else x1 - span
            y2 = y1 + span if erelease.ydata >= eclick.ydata else y1 - span

            if rect is None:
                rect = ax.add_patch(plt.Rectangle((x1, y1), span, span, fill=False, edgecolor='r', linewidth=2))
            else:
                rect.set_width(span)
                rect.set_height(span)
                rect.set_xy((x1, y1))

            # Update the dimensions in the text box
            dim_text.set_val(f"{span}x{span}")

            fig.canvas.draw()

        # Function to confirm the selection
        def confirm_selection(event):
            nonlocal cropped_image, crop_coords, x1, y1, x2, y2
            crop_coords = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
            cropped_image = image_array[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
            plt.close(fig)

        # Create the rectangle selector
        x1, y1, x2, y2 = crop_coords
        rect = ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2))
        rs = RectangleSelector(ax, update_rect, useblit=True,
                               button=[1, 3],  # Left click to start, right click to stop
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)

        # Add a confirm button
        ax_confirm = plt.axes([0.8, 0.05, 0.1, 0.075])
        button = Button(ax_confirm, 'Confirm')
        button.on_clicked(confirm_selection)

        plt.show(block=True)
        return torch.tensor(cropped_image), crop_coords
    else:
        return image_array[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]], crop_coords


def create_results_directory(base_folder, has_ref_images):
    # If we have reference images, create the results folder in the parent directory
    if has_ref_images:
        base_folder = os.path.dirname(base_folder)

    # Create a time-stamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_folder, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Create subdirectories for each category of output
    subfolders = ['cropped_data', 'original_fields']
    if has_ref_images:
        subfolders.append('normalized_original_fields')

    paths = {}
    for subfolder in subfolders:
        path = os.path.join(results_dir, subfolder)
        os.makedirs(path, exist_ok=True)
        paths[subfolder] = path

    return paths

def save_file(path, data):
    np.save(path, data)


def save_results_parallel(paths, base_filename, cropped_image, reconstructed_field, normalized_field=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3 if normalized_field is not None else 2) as executor:
        futures = [
            executor.submit(np.save, os.path.join(paths['cropped_data'], f"{base_filename}_cropped.npy"),
                            cropped_image),
            executor.submit(np.save, os.path.join(paths['original_fields'], f"{base_filename}_matrix.npy"),
                            reconstructed_field)
        ]

        if normalized_field is not None:
            futures.append(executor.submit(np.save,
                                           os.path.join(paths['normalized_original_fields'], f"{base_filename}_normalized.npy"),
                                           normalized_field))

        concurrent.futures.wait(futures)

def process_batch(batch_fft, fft_crop_coords, mean_abs_ref_field=None):
    reconstructed_fields = process_fft_tensor(batch_fft, fft_crop_coords)

    normalized_fields = None
    if mean_abs_ref_field is not None:
        # Normalize each field by the square root of the mean reference intensity
        normalized_fields = reconstructed_fields / torch.sqrt(mean_abs_ref_field)

    return reconstructed_fields, normalized_fields


def save_batch_results(batch_data, results_paths, pbar_save):
    base_filenames, cropped_images, reconstructed_fields, normalized_fields = batch_data
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for j, base_filename in enumerate(base_filenames):
            future = executor.submit(
                save_results_parallel,
                results_paths,
                base_filename,
                cropped_images[j],
                reconstructed_fields[j],
                normalized_fields[j] if normalized_fields is not None else None
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()
            pbar_save.update(1)

    pbar_save.refresh()

def saving_worker(processed_queue, results_paths, pbar_save):
    while True:
        batch_data = processed_queue.get()
        if batch_data is None:
            break
        save_batch_results(batch_data, results_paths, pbar_save)
        processed_queue.task_done()


# Main execution
files, folder, has_ref_images, ref_folder, ref_files, has_noise_images, noise_folder, noise_files = select_files_or_folder()
files = natsorted(files)
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images_tensor = load_images_as_tensor(files, device='cpu')
ref_images_tensor = None
noise_images_tensor = None
if has_ref_images:
    print(f"Reference images found in {ref_folder}, \nLoading {len(ref_files)} reference images")
    ref_images_tensor = load_images_as_tensor(ref_files, device='cpu')
else:
    print("No reference images found. Processing without reference normalization.")

if has_noise_images:
    print(f"Noise images found in {noise_folder}, \nLoading {len(noise_files)} noise images")
    noise_images_tensor = load_images_as_tensor(noise_files, device='cpu')
    # average_noise = torch.mean(noise_images_tensor, axis=0)
    images_tensor = images_tensor-noise_images_tensor
else:
    print("No noise images found. Processing without noise normalization.")


results_paths = create_results_directory(folder, has_ref_images)
sample_image = images_tensor[0].cpu().numpy()
time_1 = time.time() - start_time
cropped_sample, crop_coords = crop_image_interactively(sample_image, 'Select Original Area')
fft_result = fftshift(fft2(cropped_sample))
fft_magnitude = torch.log(torch.abs(fft_result) + 1).cpu()
cropped_fft_magnitude, fft_crop_coords = crop_image_interactively(fft_magnitude, 'Select FFT Area for Reconstruction')
start_time = time.time()

if crop_coords != (0, 0, images_tensor[0].shape[1], images_tensor[0].shape[0]):
    cropped_images_tensor = crop_tensor(images_tensor, crop_coords)
    if has_ref_images:
        cropped_ref_images_tensor = crop_tensor(ref_images_tensor, crop_coords)
else:
    cropped_images_tensor = images_tensor
    if has_ref_images:
        cropped_ref_images_tensor = ref_images_tensor
del images_tensor
images_fft_tensor = perform_fft_on_tensor_individually(cropped_images_tensor)
if has_ref_images:
    ref_images_fft_tensor = perform_fft_on_tensor_individually(cropped_ref_images_tensor)

    # Calculate the center coordinates for the reference crop
    h, w = ref_images_fft_tensor.shape[-2:]
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = fft_crop_coords[3] - fft_crop_coords[1], fft_crop_coords[2] - fft_crop_coords[0]

    ref_crop_coords = (
        center_w - crop_w // 2,
        center_h - crop_h // 2,
        center_w + crop_w // 2 + (crop_w % 2),  # Add 1 if crop_w is odd
        center_h + crop_h // 2 + (crop_h % 2)  # Add 1 if crop_h is odd
    )

    print(f"Main fft_crop_coords: {fft_crop_coords}")
    print(f"Reference ref_crop_coords: {ref_crop_coords}")

    reconstructed_ref_fields = process_fft_tensor(ref_images_fft_tensor, ref_crop_coords)
    mean_abs_ref_field = torch.mean(torch.abs(reconstructed_ref_fields) ** 2, dim=0)
else:
    mean_abs_ref_field = None

torch.cuda.empty_cache()
batch_size = 10  # Adjust based on your GPU memory
total_batches = (len(files) + batch_size - 1) // batch_size

# Create a queue to hold processed batches
processed_queue = queue.Queue(maxsize=2)  # Adjust maxsize as needed

# Progress bars
pbar_process = tqdm(total=total_batches, desc="Processing batches", position=0, leave=True, ncols=100)
pbar_save = tqdm(total=len(files), desc="Saving results     ", position=1, leave=True, ncols=100, mininterval=0.5)

# Start a thread to handle saving
save_thread = threading.Thread(target=saving_worker, args=(processed_queue, results_paths, pbar_save))
save_thread.start()

try:
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_fft = images_fft_tensor[i:i + batch_size].to(device)

        reconstructed_fields, normalized_fields = process_batch(batch_fft, fft_crop_coords, mean_abs_ref_field)

        base_filenames = [os.path.basename(os.path.splitext(f)[0]) for f in batch_files]

        # Move results to CPU and convert to numpy arrays
        cropped_images = cropped_images_tensor[i:i + batch_size].cpu().numpy()
        reconstructed_fields = reconstructed_fields.cpu().numpy()

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Put processed batch in the queue
        processed_queue.put((base_filenames, cropped_images, reconstructed_fields, normalized_fields))

        pbar_process.update(1)

finally:
    # Signal the saving thread to finish and wait for it
    processed_queue.put(None)
    save_thread.join()

    # Close the progress bars
    pbar_process.close()
    pbar_save.close()


print(f"Processing and saving completed in {time.time() - start_time + time_1} seconds.")