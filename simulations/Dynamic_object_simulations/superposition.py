import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import transforms
import matplotlib.patches as patches
plt.ion()

def phplot(field, amp=1):
    """
    Implementation of phplot for phase visualization
    """
    # Calculate phase and amplitude with log scaling
    phase = np.unwrap(np.angle(field))
    amplitude = np.log(np.abs(field) + 1)  # Added logarithmic scaling

    if amp != 0:
        # Add check for zero maximum
        max_amp = np.max(amplitude)
        if max_amp > 0:
            amplitude = amplitude / max_amp  # Normalize after log scaling
        else:
            amplitude = np.zeros_like(amplitude)
    else:
        amplitude = np.ones_like(amplitude)

    # Normalize phase to [0, 2π] range after unwrapping
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    if phase_max > phase_min:
        phase = (phase - phase_min) / (phase_max - phase_min) * 2 * np.pi
    else:
        phase = np.zeros_like(phase)

    # Handle any remaining NaN values
    amplitude = np.nan_to_num(amplitude)
    phase = np.nan_to_num(phase)

    # Create RGB array
    A = np.zeros((*field.shape, 3))

    # Map phase to RGB using trigonometric functions
    A[..., 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
    A[..., 1] = 0.5 * (np.sin(phase + np.pi/2) + 1) * amplitude  # Green
    A[..., 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

    # Ensure values are in valid range
    A = np.clip(A, 0, 1)

    return A

def apply_lpf(image, sigma=2.0):
    """
    Apply Gaussian Low Pass Filter to image
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    mask = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    filtered_fshift = fshift * mask
    filtered_f = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.real(np.fft.ifft2(filtered_f))

    # Add checks for invalid values
    min_val = np.nanmin(filtered_image)
    max_val = np.nanmax(filtered_image)

    if max_val == min_val:
        filtered_image = np.zeros_like(filtered_image)
    else:
        filtered_image = (filtered_image - min_val) / (max_val - min_val)

    # Ensure no NaN values
    filtered_image = np.nan_to_num(filtered_image)

    return filtered_image

def create_frames(num_frames, width=500, height=500):
    frames = []  # frames with road (for display)
    ffts = []  # Store FFT data from car-only frames

    # Create base figure for rendering
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    # Road polygon points
    road_points = np.array([
        [1, 0],  # Bottom left
        [9, 0],  # Bottom right
        [6.5, 5],  # Top right
        [3.5, 5],  # Top left
        [1, 0]  # Close polygon
    ])

    # Initial car properties
    car_size = 2.0
    car_x = 4.0
    rotation_start = 0.3
    num_rotations = 3
    rotation_speed = 8

    def get_position_scale_and_rotation(frame):
        progress = frame / num_frames

        # Base position (perspective motion)
        start_x, start_y = car_x, 0.5
        end_x, end_y = 5.0, 4.5

        # Current base position
        current_x = start_x + (end_x - start_x) * progress
        current_y = start_y + (end_y - start_y) * progress

        # Add lateral movement
        lateral_amplitude = 1.5 * (1 - progress)
        lateral_frequency = 3
        lateral_offset = lateral_amplitude * np.sin(2 * np.pi * lateral_frequency * progress)

        # Rotation
        if progress < rotation_start:
            rotation = 0
        else:
            adjusted_progress = (progress - rotation_start) / (1 - rotation_start)
            rotation = adjusted_progress * 360 * num_rotations * rotation_speed

        # Combine movements
        final_x = current_x + lateral_offset
        scale = 1.0 - (0.99 * progress)

        return final_x, current_y, scale, rotation

    for frame_number in range(num_frames):
        # Generate two frames for each iteration: one with road, one without
        for include_road in [True, False]:
            # Clear the figure for new frame
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.set_axis_off()

            # Draw road only if include_road is True
            if include_road:
                road = plt.Polygon(road_points, color='gray', zorder=1)
                ax.add_patch(road)

            # Get current position, scale, and rotation
            x, y, scale, rotation = get_position_scale_and_rotation(frame_number)

            # Calculate center point for rotation
            center_x = x + (car_size * scale) / 2
            center_y = y + (car_size * scale) / 2

            # Create transform
            t = (transforms.Affine2D()
                 .translate(-car_x - car_size / 2, -0.5 - car_size / 2)
                 .rotate_deg(rotation)
                 .scale(scale)
                 .translate(center_x, center_y))

            # Create car parts
            car_body = patches.Rectangle((car_x, 0.5), car_size, car_size,
                                         color="blue", zorder=2)
            wheel1 = patches.Ellipse((car_x + 0.3, 0.5), 0.15, 0.4,
                                     color="black", zorder=2)
            wheel2 = patches.Ellipse((car_x + car_size - 0.3, 0.5), 0.15, 0.4,
                                     color="black", zorder=2)
            window_size = car_size * 0.7
            window_x = car_x + (car_size - window_size) / 2
            window_y = 0.5 + (car_size - window_size) / 2
            rear_window = patches.Rectangle((window_x, window_y), window_size,
                                            window_size, color="lightblue", zorder=3)

            # Apply transform and add all car parts
            for part in [car_body, wheel1, wheel2, rear_window]:
                part.set_transform(t + ax.transData)
                ax.add_patch(part)

            # Render frame to array
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            image = image[:, :, :3]

            # Convert to grayscale and normalize
            frame = np.mean(image, axis=2) / 255.0

            if include_road:
                # Store the frame with road for display
                frames.append(frame)
            else:
                # Calculate FFT from the car-only frame
                filtered_frame = apply_lpf(frame, sigma=75)
                fft = np.fft.fftshift(np.fft.fft2(filtered_frame))
                ffts.append(fft)

    plt.close(fig)
    return np.array(frames), np.array(ffts)

def save_combined_animation(frames, ffts, filename, fps=20):
    """
    Save a combined animation with original and FFT visualization side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize images
    img1 = ax1.imshow(frames[0], cmap='gray')
    img2 = ax2.imshow(phplot(ffts[0]))

    # Set titles
    ax1.set_title('Original')
    ax2.set_title('FFT Phase (phplot)')

    # Turn off axes
    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()

    def update(frame):
        img1.set_array(frames[frame])
        img2.set_array(phplot(ffts[frame]))
        return img1, img2

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                       interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_combined_mag_plot_animation(frames, ffts, filename, fps=20):
    """
    Save animation with original and FFT magnitude visualizations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize images
    frame = frames[0]
    magnitude = np.log(np.abs(ffts[0]) + 1)

    img1 = ax1.imshow(frame, cmap='gray')
    img2 = ax2.imshow(magnitude, cmap='gray')

    # Set titles
    ax1.set_title('Original')
    ax2.set_title('FFT Magnitude (log scale)')

    # Turn off axes
    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()

    def update(frame_idx):  # Changed parameter name to frame_idx
        frame = frames[frame_idx]  # Use frame_idx for indexing
        magnitude = np.log(np.abs(ffts[frame_idx]) + 1)  # Use frame_idx for indexing

        img1.set_array(frame)
        img2.set_array(magnitude)
        return img1, img2

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_triple_animation(frames, ffts, filename, fps=20):
    """
    Save animation with original, FFT magnitude, and FFT phase visualizations
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Initialize images
    frame = frames[0]
    magnitude = np.log(np.abs(ffts[0]) + 1)
    phase = np.unwrap(np.angle(ffts[0]))

    img1 = ax1.imshow(frame, cmap='gray')
    img2 = ax2.imshow(magnitude, cmap='viridis')
    img3 = ax3.imshow(phase, cmap='hsv')

    # Set titles
    ax1.set_title('Original')
    ax2.set_title('FFT Magnitude (log scale)')
    ax3.set_title('FFT Phase')

    # Turn off axes
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.tight_layout()

    def update(frame_idx):  # Changed parameter name to frame_idx
        frame = frames[frame_idx]  # Use frame_idx for indexing
        magnitude = np.log(np.abs(ffts[frame_idx]) + 1)  # Use frame_idx for indexing
        phase = np.unwrap(np.angle(ffts[frame_idx]))  # Use frame_idx for indexing

        img1.set_array(frame)
        img2.set_array(magnitude)
        img3.set_array(phase)
        return img1, img2, img3

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

def init():
    return img1, img2

def update(frame_idx):
    from time import time
    start = time()

    img1.set_array(frames[frame_idx])
    img2.set_array(np.abs(np.log(ffts[frame_idx])+1))

    print(f"Frame: {frame_idx}, Time taken: {(time() - start) * 1000:.1f}ms")
    return img1, img2

# Generate frames
num_frames = 60
frames, ffts = create_frames(num_frames)

# Save combined synchronized animation
# save_single_animation(frames,  'single_animation_superposition.gif', title='Amp', fps=20)
save_combined_mag_plot_animation(frames, ffts, 'combined_animation_superposition.gif', fps=10 )
# save_triple_animation(frames, ffts, 'combined_animation_superposition_triple_view.gif', fps=10)

# Setup figure for live visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Initialize empty images
img1 = ax1.imshow(frames[0], cmap='gray')
img2 = ax2.imshow(np.log(np.abs(ffts[0])+1), cmap='gray')

# Set titles
ax1.set_title('Original')
ax2.set_title('FFT Amplidude')

# Turn off axes
ax1.axis('off')
ax2.axis('off')

plt.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames,
                   init_func=init, blit=True,
                   interval=50)

plt.show()