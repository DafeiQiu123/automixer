import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the experiment directories
exp_dir = Path(__file__).parent
hidden_dims = [32, 128, 512]
groups = ['none', 'bpm', 'naive']

# Define colors for each group
colors = {
    'none': '#1f77b4',  # blue
    'bpm': '#ff7f0e',   # orange
    'naive': '#2ca02c',  # green
    'vanilla': '#2ca02c'  # green
}

# Define line styles for each group
linestyles = {
    'none': '-',      # solid
    'bpm': '--',      # dashed
    'naive': '-.',     # dash-dot
    'vanilla': '-.'     # dash-dot
}

def smooth_exponential(x, beta=0.9):
    """Apply exponential smoothing to the data."""
    y = []
    last = x[0]
    for val in x:
        last = beta * last + (1 - beta) * val
        y.append(last)
    return y

def smooth_moving_avg(x, window=50):
    """Apply moving average smoothing to the data."""
    return np.convolve(x, np.ones(window)/window, mode='valid')

def load_iter_losses(group, hidden_dim):
    """Load the iteration losses for a given group and hidden dimension."""
    checkpoint_dir = exp_dir / f"checkpoints_d{hidden_dim}_L1_pe_{group}"
    loss_file = checkpoint_dir / "loss_logs.json"

    with open(loss_file, 'r') as f:
        data = json.load(f)

    train_losses = data['train_iter_losses']
    valid_losses = data['valid_iter_losses']

    return train_losses, valid_losses

# Create a figure with 2 rows (train/valid) and 3 columns (hidden dims)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot for each hidden dimension
for col_idx, hidden_dim in enumerate(hidden_dims):
    # Training losses (top row)
    ax_train = axes[0, col_idx]
    # Validation losses (bottom row)
    ax_valid = axes[1, col_idx]

    for group in groups:
        train_losses, valid_losses = load_iter_losses(group, hidden_dim)

        # Apply exponential smoothing
        train_smooth = smooth_exponential(train_losses, beta=0.9)
        valid_smooth = smooth_exponential(valid_losses, beta=0.9)
        display_label = 'vanilla' if group == 'naive' else ('baseline' if group == 'none' else group)
        style_group = group

        # Plot training losses
        ax_train.plot(range(len(train_smooth)), train_smooth,
                     color=colors[style_group],
                     label=display_label,
                     linewidth=2.5,
                     linestyle=linestyles[style_group],
                     alpha=0.9)

        # Plot validation losses
        ax_valid.plot(range(len(valid_smooth)), valid_smooth,
                     color=colors[style_group],
                     label=display_label,
                     linewidth=2.5,
                     linestyle=linestyles[style_group],
                     alpha=0.9)

    # Configure training plot
    ax_train.set_title(f'Training Loss (d={hidden_dim})', fontsize=13, fontweight='bold')
    ax_train.set_xlabel('Iteration', fontsize=11)
    ax_train.set_ylabel('Loss (log scale)', fontsize=11)
    ax_train.set_yscale('log')
    ax_train.legend(fontsize=10)
    ax_train.grid(True, alpha=0.3, which='both')

    # Configure validation plot
    ax_valid.set_title(f'Validation Loss (d={hidden_dim})', fontsize=13, fontweight='bold')
    ax_valid.set_xlabel('Iteration', fontsize=11)
    ax_valid.set_ylabel('Loss (log scale)', fontsize=11)
    ax_valid.set_yscale('log')
    ax_valid.legend(fontsize=10)
    ax_valid.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(exp_dir / 'iter_losses_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved combined plot to {exp_dir / 'iter_losses_comparison.png'}")

# Create separate plots for training losses
fig_train, axes_train = plt.subplots(1, 3, figsize=(18, 5))

for col_idx, hidden_dim in enumerate(hidden_dims):
    ax = axes_train[col_idx]

    for group in groups:
        train_losses, _ = load_iter_losses(group, hidden_dim)
        train_smooth = smooth_exponential(train_losses, beta=0.9)
        display_label = 'vanilla' if group == 'naive' else ('baseline' if group == 'none' else group)
        style_group = group
        ax.plot(range(len(train_smooth)), train_smooth,
               color=colors[style_group],
               label=display_label,
               linewidth=2.5,
               linestyle=linestyles[style_group],
               alpha=0.9)

    ax.set_title(f'Training Loss (d={hidden_dim})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(exp_dir / 'train_iter_losses.png', dpi=300, bbox_inches='tight')
print(f"Saved training plot to {exp_dir / 'train_iter_losses.png'}")

# Create separate plots for validation losses
fig_valid, axes_valid = plt.subplots(1, 3, figsize=(18, 5))

for col_idx, hidden_dim in enumerate(hidden_dims):
    ax = axes_valid[col_idx]

    for group in groups:
        _, valid_losses = load_iter_losses(group, hidden_dim)
        valid_smooth = smooth_exponential(valid_losses, beta=0.9)
        display_label = 'vanilla' if group == 'naive' else ('baseline' if group == 'none' else group)
        style_group = group
        ax.plot(range(len(valid_smooth)), valid_smooth,
               color=colors[style_group],
               label=display_label,
               linewidth=2.5,
               linestyle=linestyles[style_group],
               alpha=0.9)

    ax.set_title(f'Validation Loss (d={hidden_dim})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(exp_dir / 'valid_iter_losses.png', dpi=300, bbox_inches='tight')
print(f"Saved validation plot to {exp_dir / 'valid_iter_losses.png'}")

# Create individual plots for each hidden dimension (train and valid separately)
for hidden_dim in hidden_dims:
    # Training plot
    fig_train_single = plt.figure(figsize=(10, 6))
    ax_train = fig_train_single.add_subplot(111)

    for group in groups:
        train_losses, _ = load_iter_losses(group, hidden_dim)
        train_smooth = smooth_exponential(train_losses, beta=0.9)
        display_label = 'vanilla' if group == 'naive' else ('baseline' if group == 'none' else group)
        style_group = group
        ax_train.plot(range(len(train_smooth)), train_smooth,
                     color=colors[style_group],
                     label=display_label,
                     linewidth=3,
                     linestyle=linestyles[style_group],
                     alpha=0.9)

    ax_train.set_title(f'Training Loss (d={hidden_dim})', fontsize=15, fontweight='bold')
    ax_train.set_xlabel('Iteration', fontsize=13)
    ax_train.set_ylabel('Loss (log scale)', fontsize=13)
    ax_train.set_yscale('log')
    ax_train.legend(fontsize=12, framealpha=0.9)
    ax_train.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(exp_dir / f'train_d{hidden_dim}.png', dpi=300, bbox_inches='tight')
    print(f"Saved training plot to {exp_dir / f'train_d{hidden_dim}.png'}")
    plt.close()

    # Validation plot
    fig_valid_single = plt.figure(figsize=(10, 6))
    ax_valid = fig_valid_single.add_subplot(111)

    for group in groups:
        _, valid_losses = load_iter_losses(group, hidden_dim)
        valid_smooth = smooth_exponential(valid_losses, beta=0.9)
        display_label = 'vanilla' if group == 'naive' else ('baseline' if group == 'none' else group)
        style_group = group
        ax_valid.plot(range(len(valid_smooth)), valid_smooth,
                     color=colors[style_group],
                     label=display_label,
                     linewidth=3,
                     linestyle=linestyles[style_group],
                     alpha=0.9)

    ax_valid.set_title(f'Validation Loss (d={hidden_dim})', fontsize=15, fontweight='bold')
    ax_valid.set_xlabel('Iteration', fontsize=13)
    ax_valid.set_ylabel('Loss (log scale)', fontsize=13)
    ax_valid.set_yscale('log')
    ax_valid.legend(fontsize=12, framealpha=0.9)
    ax_valid.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(exp_dir / f'valid_d{hidden_dim}.png', dpi=300, bbox_inches='tight')
    print(f"Saved validation plot to {exp_dir / f'valid_d{hidden_dim}.png'}")
    plt.close()

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. iter_losses_comparison.png - Combined 2x3 grid")
print("  2. train_iter_losses.png - Training losses in 1x3 grid")
print("  3. valid_iter_losses.png - Validation losses in 1x3 grid")
print("  4-9. Individual plots for each dimension:")
print("     - train_d32.png, valid_d32.png")
print("     - train_d128.png, valid_d128.png")
print("     - train_d512.png, valid_d512.png")
print("="*60)
