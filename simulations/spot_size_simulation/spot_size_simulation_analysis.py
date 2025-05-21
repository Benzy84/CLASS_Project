import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from utils.visualization import display_field
from utils.image_processing import center_crop
from matplotlib.pyplot import imshow

nrm = lambda x: x/np.abs(x).max()

def select_results_directory():
    """Open a file dialog to select the results directory"""
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="Select Spot Size Simulation Results Directory",
        mustexist=True
    )

    if not folder_path:
        print("No directory selected. Exiting.")
        return None

    print(f"Selected directory: {folder_path}")
    return folder_path


def main():
    """Main function to load data"""
    # Select results directory
    results_dir = select_results_directory()
    if not results_dir:
        return

    # Load parameters if available
    param_file = os.path.join(results_dir, "parameters.npy")
    if os.path.exists(param_file):
        params = np.load(param_file, allow_pickle=True).item()
        print(f"Loaded parameters: {params}")
        ratios = params['ratios']
    else:
        # Try to find ratios from file names
        ratio_files = [f for f in os.listdir(results_dir) if f.startswith("reconstruction_") and f.endswith(".npy")]
        ratios = []
        for f in ratio_files:
            try:
                ratio = float(f.split("_")[1].split(".npy")[0])
                ratios.append(ratio)
            except:
                pass
        ratios.sort()
        print(f"Found ratios from filenames: {ratios}")

    # Load reference data
    ground_truth = None
    widefield = None
    diffuser = None

    if os.path.exists(os.path.join(results_dir, "ground_truth.npy")):
        ground_truth = np.load(os.path.join(results_dir, "ground_truth.npy"))
        print("Loaded ground truth object")

    if os.path.exists(os.path.join(results_dir, "widefield.npy")):
        widefield = np.load(os.path.join(results_dir, "widefield.npy"))
        print("Loaded widefield reference")

    if os.path.exists(os.path.join(results_dir, "diffuser.npy")):
        diffuser = np.load(os.path.join(results_dir, "diffuser.npy"))
        print("Loaded diffuser")

    # Load all data dictionaries if available
    data_loaded = False

    try:
        all_illuminations = np.load(os.path.join(results_dir, "all_illuminations.npy"), allow_pickle=True).item()
        all_ill_at_diff = np.load(os.path.join(results_dir, "all_illuminations_at_diffuser.npy"),
                                  allow_pickle=True).item()
        all_ill_at_obj = np.load(os.path.join(results_dir, "all_illuminations_at_object.npy"), allow_pickle=True).item()
        all_eff_obj = np.load(os.path.join(results_dir, "all_effective_objects.npy"), allow_pickle=True).item()
        all_distorted_eff_obj = np.load(os.path.join(results_dir, "all_distorted_objects.npy"),
                                        allow_pickle=True).item()
        all_recons = np.load(os.path.join(results_dir, "all_reconstructions.npy"), allow_pickle=True).item()

        print("Successfully loaded all data dictionaries")
        data_loaded = True

        # Display the keys of the loaded dictionaries
        print(f"Available ratios: {list(all_recons.keys())}")

    except Exception as e:
        print(f"Could not load compiled data dictionaries: {str(e)}")
        print("Will attempt to load individual files")

    # If compiled dictionaries weren't loaded, load individual files
    if not data_loaded:
        all_illuminations = {}
        all_ill_at_diff = {}
        all_ill_at_obj = {}
        all_eff_obj = {}
        all_distorted_eff_obj = {}
        all_recons = {}

        for ratio in ratios:
            ratio_str = f"{ratio}"
            try:
                # Try to load each file for this ratio
                if os.path.exists(os.path.join(results_dir, f"illumination_{ratio_str}.npy")):
                    all_illuminations[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"illumination_{ratio_str}.npy"))

                if os.path.exists(os.path.join(results_dir, f"illumination_at_diffuser_{ratio_str}.npy")):
                    all_ill_at_diff[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"illumination_at_diffuser_{ratio_str}.npy"))

                if os.path.exists(os.path.join(results_dir, f"illumination_at_object_{ratio_str}.npy")):
                    all_ill_at_obj[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"illumination_at_object_{ratio_str}.npy"))

                if os.path.exists(os.path.join(results_dir, f"effective_object_{ratio_str}.npy")):
                    all_eff_obj[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"effective_object_{ratio_str}.npy"))

                if os.path.exists(os.path.join(results_dir, f"distorted_object_{ratio_str}.npy")):
                    all_distorted_eff_obj[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"distorted_object_{ratio_str}.npy"))

                if os.path.exists(os.path.join(results_dir, f"reconstruction_{ratio_str}.npy")):
                    all_recons[f'spot_size/d_corr = {ratio_str}'] = np.load(
                        os.path.join(results_dir, f"reconstruction_{ratio_str}.npy"))

                print(f"Loaded data for ratio {ratio_str}")
            except Exception as e:
                print(f"Error loading data for ratio {ratio_str}: {str(e)}")

    # Print summary of loaded data
    print("\nData loading summary:")
    print(f"ground truth: {'Loaded' if ground_truth is not None else 'Not found'}")
    print(f"Widefield: {'Loaded' if widefield is not None else 'Not found'}")
    print(f"diffuser: {'Loaded' if diffuser is not None else 'Not found'}")
    print(f"Illuminations: {len(all_illuminations)} ratios")
    print(f"Illuminations at diffuser: {len(all_ill_at_diff)} ratios")
    print(f"Illuminations at object: {len(all_ill_at_obj)} ratios")
    print(f"Effective objects: {len(all_eff_obj)} ratios")
    print(f"Distorted objects: {len(all_distorted_eff_obj)} ratios")
    print(f"Reconstructions: {len(all_recons)} ratios")

    # Return all the loaded data
    return {
        'ground_truth': ground_truth,
        'widefield': widefield,
        'diffuser': diffuser,
        'all_illuminations': all_illuminations,
        'all_ill_at_diff': all_ill_at_diff,
        'all_ill_at_obj': all_ill_at_obj,
        'all_eff_obj': all_eff_obj,
        'all_distorted_eff_obj': all_distorted_eff_obj,
        'all_recons': all_recons,
        'ratios': ratios
    }


if __name__ == "__main__":
    data = main()
    ratios = data['ratios']
    ground_truth = data['ground_truth']
    widefield = data['widefield']
    diffuser = data['diffuser']
    all_illuminations = data['all_illuminations']
    all_ill_at_diff = data['all_ill_at_diff']
    all_ill_at_obj = data['all_ill_at_obj']
    all_eff_obj = data['all_eff_obj']
    all_distorted_eff_obj = data['all_distorted_eff_obj']
    all_recons = data['all_recons']
    del data
    print("\nData loaded and ready. You can now use the 'data' variable to access everything.")

    plt.figure()
    plt.imshow(np.abs(nrm(center_crop(ground_truth, 90))), cmap='gray')
    plt.title('ground_truth')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.abs(nrm(center_crop(widefield, 90))), cmap='gray')
    plt.title('widefield')
    plt.show()

    plt.figure()
    plt.imshow(np.angle(center_crop(diffuser, 90)), cmap='hsv')
    plt.title('diffuser')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle('Illumination')
    for idx, ratio in enumerate(ratios):
        pos = idx + 1

        image = all_illuminations[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(nrm(center_crop(image,90)), cmap='gray', vmin=0, vmax=1)
        plt.title(f'spot_size/d_corr = {ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('Illumination + diffuser')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        image = all_illuminations[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.angle(center_crop(diffuser, 90)), cmap='hsv')
        plt.imshow(np.abs(nrm(center_crop(image, 90))), cmap='gray', alpha=0.85)
        plt.title(f'spot_size/d_corr = {ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('Illumination after diffuser')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        ill_at_diff = all_ill_at_diff[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.abs(nrm(center_crop(ill_at_diff, 90))), cmap='gray', vmin=0, vmax=1)
        plt.title(f'spot_size/d_corr = {ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('Illumination after diffuser (phase)')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        ill_at_diff = all_ill_at_diff[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.angle(nrm(ill_at_diff)), cmap='hsv')
        plt.title(f'spot_size/d_corr = {ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('illumination at object (mag)')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        ill_at_obj = all_ill_at_obj[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.round(np.abs(nrm(center_crop(ill_at_obj, 90))), 4), cmap='gray', vmin=0, vmax=1)
        plt.title(f'spot_size/d_corr\n={ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('illumination at object (phase)')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        ill_at_obj = all_ill_at_obj[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.angle(nrm(center_crop(ill_at_obj, 90))), cmap='gray')
        plt.title(f'spot_size/d_corr\n={ratio}')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle('Effective object')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        eff_obj = all_eff_obj[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.abs(nrm(center_crop(eff_obj, 90))), cmap='gray')
        plt.title(f'spot_size/d_corr\n={ratio}')
    plt.show()

    plt.figure()
    plt.suptitle('reconstructed object')
    for idx, ratio in enumerate(ratios):

        pos = idx + 1

        reconstruction = all_recons[f'spot_size/d_corr = {ratio}']

        plt.subplot(3, 4, pos)
        plt.imshow(np.abs(nrm(center_crop(reconstruction, 90))), cmap='gray')
        plt.title(f'spot_size/d_corr\n={ratio}')
    plt.show()



a = 5




