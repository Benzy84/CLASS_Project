import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

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

def create_frames(num_frames, width=500, height=500):
    frames = []
    ffts = []  # Store FFT data directly

    sky_value = 0.9
    road_value = 0.5
    car_value = 0.3
    wheel_value = 0.1

    for frame_number in range(num_frames):
        frame = np.full((height, width), sky_value)

        # Add road (at 54% from top, 2% thick)
        road_y = int(height * 0.54)
        road_thickness = int(height * 0.02)
        frame[road_y - road_thickness // 2:road_y + road_thickness // 2, :] = road_value

        # Car dimensions
        car_width = int(width * 0.14)
        car_height = int(height * 0.08)
        car_y = int(height * 0.44)

        # Center for rotation
        center_x = width // 2
        center_y = car_y + car_height // 2

        # Calculate rotation angle
        angle_degrees = (frame_number / num_frames) * 360
        angle_radians = np.radians(angle_degrees)

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])

        # Car body corners relative to center
        car_points = np.array([
            [-car_width // 2, -car_height // 2],
            [car_width // 2, -car_height // 2],
            [car_width // 2, car_height // 2],
            [-car_width // 2, car_height // 2]
        ])

        # Rotate car points
        rotated_car = np.dot(car_points, rotation_matrix.T)
        rotated_car += np.array([center_x, center_y])
        rotated_car = rotated_car.astype(int)

        # Draw car body using mask
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv_points = np.array([rotated_car], dtype=np.int32)
        pts = cv_points.reshape((-1, 2))
        for i in range(height):
            for j in range(width):
                point = np.array([j, i])
                if point_in_polygon(point, pts):
                    frame[i, j] = car_value

        # Add wheels
        wheel_radius = int(width * 0.02)
        wheel_offsets = [
            (width * 0.03, int(height * 0.52) - center_y),
            (width * 0.11, int(height * 0.52) - center_y)
        ]

        # Draw wheels
        for offset_x, offset_y in wheel_offsets:
            rel_x = offset_x - car_width / 2
            wheel_pos = [rel_x, offset_y]
            rotated_wheel = np.dot(wheel_pos, rotation_matrix.T)
            wheel_center = rotated_wheel + np.array([center_x, center_y])
            wheel_center = wheel_center.astype(int)

            for dy in range(-wheel_radius, wheel_radius + 1):
                for dx in range(-wheel_radius, wheel_radius + 1):
                    if dx * dx + dy * dy <= wheel_radius * wheel_radius:
                        py = wheel_center[1] + dy
                        px = wheel_center[0] + dx
                        if 0 <= py < height and 0 <= px < width:
                            frame[py, px] = wheel_value

        filtered_frame = apply_lpf(frame, sigma=75)
        frames.append(frame)
        fft = np.fft.fftshift(np.fft.fft2(filtered_frame))
        ffts.append(fft)

    return np.array(frames), np.array(ffts)

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i

    return inside

def save_combined_phplot_animation(frames, ffts, filename, fps=20):
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
    return img1, img2, img3

def update(frame_idx):
    img1.set_array(frames[frame_idx])
    img2.set_array(np.log(np.abs(ffts[frame_idx])+1))
    img3.set_array(np.unwrap(np.angle(ffts[frame_idx])))
    return img1, img2, img3

# Generate frames
num_frames = 45
frames, ffts = create_frames(num_frames)

# Save combined synchronized animation
save_combined_mag_plot_animation(frames, ffts, 'combined_animation_rotating_car.gif', fps=15)
# save_triple_animation(frames, ffts, 'combined_animation_rotating_car_triple_view.gif', fps=10)

# Setup figure for live visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Initialize empty images
img1 = ax1.imshow(frames[0], cmap='gray')
img2 = ax2.imshow(np.log(np.abs(ffts[0])+1), cmap='gray')
img3 = ax3.imshow(np.unwrap(np.angle(ffts[0])))

# Set titles
ax1.set_title('Original')
ax2.set_title('FFT Amplidude')
ax3.set_title('FFT Phase')

# Turn off axes
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')

plt.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames,
                   init_func=init, blit=True,
                   interval=100)

plt.show()