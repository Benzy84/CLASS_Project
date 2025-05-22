import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def phplot(field, amp=1):
    """
    Implementation of phplot for phase visualization
    Args:
        field: Complex array to visualize
        amp: If 0, amplitude is not plotted (default=1)
    Returns:
        RGB array
    """
    # Calculate phase and amplitude with log scaling
    phase = np.unwrap(np.angle(field))
    amplitude = np.log(np.abs(field) + 1)  # Added logarithmic scaling

    if amp != 0:
        amplitude = amplitude / np.max(amplitude)  # Normalize after log scaling
    else:
        amplitude = np.ones_like(amplitude)

    # Normalize phase to [0, 2π] range after unwrapping
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * 2 * np.pi

    # Create RGB array
    A = np.zeros((*field.shape, 3))

    # Map phase to RGB using trigonometric functions
    A[..., 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
    A[..., 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amplitude  # Green
    A[..., 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

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

        # Car position (moves from 0 to width - car_width)
        car_width = int(width * 0.14)  # Car width 14% of screen
        x = (frame_number / num_frames) * (width - car_width)
        car_x = int(x)
        car_y = int(height * 0.44)  # Car at 44% from top
        car_height = int(height * 0.08)  # Car height 8% of screen

        # Draw car body
        frame[car_y:car_y + car_height, car_x:car_x + car_width] = car_value

        # Add wheels
        wheel_radius = int(width * 0.02)  # Wheel size 2% of width
        wheel_centers = [
            (int(car_x + width * 0.03), int(height * 0.52)),  # Front wheel
            (int(car_x + width * 0.11), int(height * 0.52))  # Back wheel
        ]

        # Draw wheels
        for wheel_x, wheel_y in wheel_centers:
            for dy in range(-wheel_radius, wheel_radius + 1):
                for dx in range(-wheel_radius, wheel_radius + 1):
                    if dx * dx + dy * dy <= wheel_radius * wheel_radius:
                        py = wheel_y + dy
                        px = wheel_x + dx
                        if 0 <= py < height and 0 <= px < width:
                            frame[py, px] = wheel_value

        frames.append(frame)
        # Store raw FFT data
        fft = np.fft.fftshift(np.fft.fft2(frame))
        ffts.append(fft)

    return np.array(frames), np.array(ffts)

def save_single_animation(frames_array,  filename, title='No title', fps=20):
    """
    Save a combined animation with original and FFT visualization side by side
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Initialize images
    img = ax.imshow(frames_array[0], cmap='gray')

    # Set titles
    ax.set_title(title)

    # Turn off axes
    ax.axis('off')

    plt.tight_layout()

    def update(frame):
        img.set_array(frames[frame])
        return [img]

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                       interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

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


def save_animation_with_colorbars(frames, ffts, filename, fps=20):
    """
    Save animation with original, FFT magnitude, and FFT phase visualizations,
    including colorbars and consistent scaling
    """
    # Pre-calculate all values for consistent scaling
    magnitude_values = [np.log(np.abs(fft) + 1) for fft in ffts]
    phase_values = [np.unwrap(np.angle(fft)) for fft in ffts]
    height, width = frames[0].shape

    # Calculate global magnitude range
    magnitude_min = min(np.min(mag) for mag in magnitude_values)
    magnitude_max = max(np.max(mag) for mag in magnitude_values)

    # Calculate global phase range after center column subtraction
    adjusted_phases = []
    for phase in phase_values:
        center_column = phase[:, width // 2:width // 2 + 1]
        adjusted_phase = phase - center_column
        adjusted_phases.append(adjusted_phase)

    phase_min = min(np.min(phase) for phase in adjusted_phases)
    phase_max = max(np.max(phase) for phase in adjusted_phases)
    v_min, v_max = phase_min, phase_max

    # Setup figure with space for colorbars
    fig = plt.figure(figsize=(16, 5))
    gs = plt.GridSpec(1, 7, width_ratios=[4, 0.2, 4, 0.2, 4, 0.2, 0.2])
    ax1 = fig.add_subplot(gs[0])
    cax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax2 = fig.add_subplot(gs[3])
    ax3 = fig.add_subplot(gs[4])
    cax3 = fig.add_subplot(gs[5])

    # Initialize images with consistent scaling
    img1 = ax1.imshow(frames[0], cmap='gray', vmin=0.1, vmax=0.9)
    plt.colorbar(img1, cax=cax1, label='Intensity')

    img2 = ax2.imshow(magnitude_values[0], cmap='viridis',
                      vmin=magnitude_min, vmax=magnitude_max)
    plt.colorbar(img2, cax=cax2, label='Log Magnitude')

    img3 = ax3.imshow(adjusted_phases[0], cmap='RdBu',
                      vmin=v_min, vmax=v_max)
    plt.colorbar(img3, cax=cax3, label='Phase (rad)')

    # Set titles and turn off axes
    ax1.set_title('Original')
    ax2.set_title('FFT Magnitude (log scale)')
    ax3.set_title('FFT Phase (centered)')
    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    plt.tight_layout()

    def update(frame_idx):
        # Update original frame
        img1.set_array(frames[frame_idx])

        # Update magnitude
        img2.set_array(magnitude_values[frame_idx])

        # Update phase
        img3.set_array(adjusted_phases[frame_idx])

        return img1, img2, img3

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_animation_with_colorbars_and_lineplot(frames, ffts, filename, fps=20):
    """
    Save animation with original, FFT magnitude, FFT phase visualizations,
    and a line plot of a single row of phase values
    """
    # Pre-calculate all values for consistent scaling
    magnitude_values = [np.log(np.abs(fft) + 1) for fft in ffts]
    phase_values = [np.unwrap(np.angle(fft)) for fft in ffts]
    height, width = frames[0].shape

    # Calculate global magnitude range
    magnitude_min = min(np.min(mag) for mag in magnitude_values)
    magnitude_max = max(np.max(mag) for mag in magnitude_values)

    # Calculate global phase range after center column subtraction
    adjusted_phases = []
    for phase in phase_values:
        center_column = phase[:, width // 2:width // 2 + 1]
        adjusted_phase = phase - center_column
        adjusted_phases.append(adjusted_phase)

    phase_min = min(np.min(phase) for phase in adjusted_phases)
    phase_max = max(np.max(phase) for phase in adjusted_phases)
    v_min, v_max = phase_min, phase_max

    # Setup figure with space for colorbars and line plot
    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 9, width_ratios=[4, 0.2, 4, 0.2, 4, 0.2, 6, 0.2, 0.2])
    ax1 = fig.add_subplot(gs[0])
    cax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax2 = fig.add_subplot(gs[3])
    ax3 = fig.add_subplot(gs[4])
    cax3 = fig.add_subplot(gs[5])
    ax4 = fig.add_subplot(gs[6])  # Line plot

    # Initialize images with consistent scaling
    img1 = ax1.imshow(frames[0], cmap='gray', vmin=0.1, vmax=0.9)
    plt.colorbar(img1, cax=cax1, label='Intensity')

    img2 = ax2.imshow(magnitude_values[0], cmap='viridis',
                      vmin=magnitude_min, vmax=magnitude_max)
    plt.colorbar(img2, cax=cax2, label='Log Magnitude')

    img3 = ax3.imshow(adjusted_phases[0], cmap='RdBu',
                      vmin=v_min, vmax=v_max)
    plt.colorbar(img3, cax=cax3, label='Phase (rad)')

    # Initialize line plot
    row_to_plot = height // 2  # Middle row
    x = np.arange(width) - width // 2  # Centered x-axis
    line, = ax4.plot(x, adjusted_phases[0][row_to_plot, :])
    ax4.set_ylim(v_min, v_max)
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Phase (rad)')
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    # Set titles and turn off axes for image plots
    ax1.set_title('Original')
    ax2.set_title('FFT Magnitude (log scale)')
    ax3.set_title('FFT Phase (centered)')
    ax4.set_title('Phase Values Along Middle Row')
    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    plt.tight_layout()

    def update(frame_idx):
        # Update original frame
        img1.set_array(frames[frame_idx])

        # Update magnitude
        img2.set_array(magnitude_values[frame_idx])

        # Update phase
        img3.set_array(adjusted_phases[frame_idx])

        # Update line plot
        line.set_ydata(adjusted_phases[frame_idx][row_to_plot, :])

        return img1, img2, img3, line

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_original_animation(frames, filename, fps=20):
    """Save animation of original frames"""
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    img = ax.imshow(frames[0], cmap='gray', vmin=0.1, vmax=0.9)

    ax.set_title('Original')
    ax.axis('off')
    plt.tight_layout()

    def update(frame_idx):
        img.set_array(frames[frame_idx])
        return [img]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_magnitude_animation(ffts, filename, fps=20):
    """Save animation of FFT magnitude"""
    magnitude_values = [np.log(np.abs(fft) + 1) for fft in ffts]
    magnitude_min = min(np.min(mag) for mag in magnitude_values)
    magnitude_max = max(np.max(mag) for mag in magnitude_values)

    fig = plt.figure(figsize=(6, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    img = ax.imshow(magnitude_values[0], cmap='gray',
                    vmin=magnitude_min, vmax=magnitude_max)
    plt.colorbar(img, cax=cax, label='Log Magnitude')

    ax.set_title('FFT Magnitude (log scale)')
    ax.axis('off')
    plt.tight_layout()

    def update(frame_idx):
        img.set_array(magnitude_values[frame_idx])
        return [img]

    ani = FuncAnimation(fig, update, frames=len(ffts), interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_phase_animation(ffts, filename, fps=20):
    """Save animation of FFT phase"""
    phase_values = [np.unwrap(np.angle(fft)) for fft in ffts]
    height, width = ffts[0].shape

    adjusted_phases = []
    for phase in phase_values:
        center_column = phase[:, width // 2:width // 2 + 1]
        adjusted_phase = phase - center_column
        adjusted_phases.append(adjusted_phase)

    phase_min = min(np.min(phase) for phase in adjusted_phases)
    phase_max = max(np.max(phase) for phase in adjusted_phases)

    fig = plt.figure(figsize=(6, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    img = ax.imshow(adjusted_phases[0], cmap='seismic',
                    vmin=phase_min, vmax=phase_max)
    plt.colorbar(img, cax=cax, label='Phase (rad)')

    ax.set_title('FFT Phase (centered)')
    ax.axis('off')
    plt.tight_layout()

    def update(frame_idx):
        img.set_array(adjusted_phases[frame_idx])
        return [img]

    ani = FuncAnimation(fig, update, frames=len(ffts), interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_phase_line_animation(ffts, filename, fps=20):
    """Save animation of phase line plot"""
    phase_values = [np.unwrap(np.angle(fft)) for fft in ffts]
    height, width = ffts[0].shape

    adjusted_phases = []
    for phase in phase_values:
        center_column = phase[:, width // 2:width // 2 + 1]
        adjusted_phase = phase - center_column
        adjusted_phases.append(adjusted_phase)

    phase_min = min(np.min(phase) for phase in adjusted_phases)
    phase_max = max(np.max(phase) for phase in adjusted_phases)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    row_to_plot = height // 2  # Middle row
    x = np.arange(width) - width // 2  # Centered x-axis
    line, = ax.plot(x, adjusted_phases[0][row_to_plot, :])

    ax.set_ylim(phase_min, phase_max)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Phase (rad)')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Phase Values Along Middle Row')

    plt.tight_layout()

    def update(frame_idx):
        line.set_ydata(adjusted_phases[frame_idx][row_to_plot, :])
        return [line]

    ani = FuncAnimation(fig, update, frames=len(ffts), interval=50, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


# Example usage:
def save_all_animations(frames, ffts, fps=20):
    """Save all four animations"""
    save_original_animation(frames, 'moving_car_original.gif', fps)
    save_magnitude_animation(ffts, 'moving_car_magnitude.gif', fps)
    save_phase_animation(ffts, 'moving_car_phase.gif', fps)
    save_phase_line_animation(ffts, 'moving_car_phase_line.gif', fps)

def update(frame_idx):
    # Update original frame
    img1.set_array(frames[frame_idx])

    # Update magnitude
    img2.set_array(magnitude_values[frame_idx])

    # Update phase
    img3.set_array(adjusted_phases[frame_idx])

    return img1, img2, img3


# Generate frames and FFTs
num_frames = 90
frames, ffts = create_frames(num_frames)

# Save animation with colorbars
save_all_animations(frames, ffts, fps=15)

height, width = frames[0].shape

# Pre-calculate all values for consistent scaling
magnitude_values = [np.log(np.abs(fft) + 1) for fft in ffts]
phase_values = [np.unwrap(np.angle(fft)) for fft in ffts]

# Calculate global magnitude range
magnitude_min = min(np.min(mag) for mag in magnitude_values)
magnitude_max = max(np.max(mag) for mag in magnitude_values)

# Calculate global phase range after center column subtraction
adjusted_phases = []
for phase in phase_values:
    center_column = phase[:, width // 2:width // 2 + 1]
    adjusted_phase = phase - center_column
    adjusted_phases.append(adjusted_phase)

phase_min = min(np.min(phase) for phase in adjusted_phases)
phase_max = max(np.max(phase) for phase in adjusted_phases)
v_min, v_max = phase_min, phase_max

# Setup figure with space for colorbars
fig = plt.figure(figsize=(16, 5))
gs = plt.GridSpec(1, 7, width_ratios=[4, 0.2, 4, 0.2, 4, 0.2, 0.2])
ax1 = fig.add_subplot(gs[0])
cax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
cax2 = fig.add_subplot(gs[3])
ax3 = fig.add_subplot(gs[4])
cax3 = fig.add_subplot(gs[5])

# Initialize images with consistent scaling
img1 = ax1.imshow(frames[0], cmap='gray', vmin=0.1, vmax=0.9)
plt.colorbar(img1, cax=cax1, label='Intensity')

img2 = ax2.imshow(magnitude_values[0], cmap='viridis',
                  vmin=magnitude_min, vmax=magnitude_max)
plt.colorbar(img2, cax=cax2, label='Log Magnitude')

img3 = ax3.imshow(adjusted_phases[0], cmap='seismic',
                  vmin=v_min, vmax=v_max)
plt.colorbar(img3, cax=cax3, label='Phase (rad)')

# Set titles and turn off axes
ax1.set_title('Original')
ax2.set_title('FFT Magnitude (log scale)')
ax3.set_title('FFT Phase (centered)')
for ax in (ax1, ax2, ax3):
    ax.axis('off')

plt.tight_layout()


# Create animation
ani = FuncAnimation(fig, update, frames=num_frames,
                    interval=50, blit=True)

plt.show()

a=5