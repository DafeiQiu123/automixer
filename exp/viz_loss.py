import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the experiment directories
exp_dir = Path(__file__).parent
hidden_dims = [32, 128]
groups = ['none', 'bpm', 'naive']

# Define colors for each group
colors = {
    'none': '#1f77b4',  # blue
    'bpm': '#ff7f0e',   # orange
    'naive': '#2ca02c'  # green
}

# Define markers for each group
markers = {
    'none': 'o',
    'bpm': 's',
    'vanilla': '^'
}

def load_final_losses(group, hidden_dim):
    """Load the average of the last 20 iterations for a given group and hidden dimension."""
    checkpoint_dir = exp_dir / f"checkpoints_d{hidden_dim}_L1_pe_{group}"
    loss_file = checkpoint_dir / "loss_logs.json"

    with open(loss_file, 'r') as f:
        data = json.load(f)

    # Get the average of the last 20 iterations
    train_loss = np.mean(data['train_iter_losses'][-20:])
    valid_loss = np.mean(data['valid_iter_losses'][-20:])

    return train_loss, valid_loss

# Collect data
train_data = {group: [] for group in groups}
valid_data = {group: [] for group in groups}

for group in groups:
    for hidden_dim in hidden_dims:
        train_loss, valid_loss = load_final_losses(group, hidden_dim)
        train_data[group].append(train_loss)
        valid_data[group].append(valid_loss)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training losses
for group in groups:
    ax1.plot(hidden_dims, train_data[group],
             color=colors[group],
             marker=markers[group],
             markersize=10,
             linewidth=2,
             label=group)

ax1.set_xlabel('Hidden Dimension', fontsize=12)
ax1.set_ylabel('Training Loss (Last 20 Iters Avg)', fontsize=12)
ax1.set_title('Training Loss vs Hidden Dimension (Last 20 Iters Avg)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_xticks(hidden_dims)
ax1.set_xticklabels(hidden_dims)

# Plot validation losses
for group in groups:
    ax2.plot(hidden_dims, valid_data[group],
             color=colors[group],
             marker=markers[group],
             markersize=10,
             linewidth=2,
             label=group)

ax2.set_xlabel('Hidden Dimension', fontsize=12)
ax2.set_ylabel('Validation Loss (Last 20 Iters Avg)', fontsize=12)
ax2.set_title('Validation Loss vs Hidden Dimension (Last 20 Iters Avg)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_xticks(hidden_dims)
ax2.set_xticklabels(hidden_dims)

plt.tight_layout()
plt.savefig(exp_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {exp_dir / 'loss_comparison.png'}")

# Also create separate plots
fig1, ax1 = plt.subplots(figsize=(8, 6))
for group in groups:
    ax1.plot(hidden_dims, train_data[group],
             color=colors[group],
             marker=markers[group],
             markersize=12,
             linewidth=2.5,
             label=group)

ax1.set_xlabel('Hidden Dimension', fontsize=13)
ax1.set_ylabel('Training Loss (Last 20 Iters Avg)', fontsize=13)
ax1.set_title('Training Loss vs Hidden Dimension (Last 20 Iters Avg)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_xticks(hidden_dims)
ax1.set_xticklabels(hidden_dims)
plt.tight_layout()
plt.savefig(exp_dir / 'train_loss_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {exp_dir / 'train_loss_comparison.png'}")

fig2, ax2 = plt.subplots(figsize=(8, 6))
for group in groups:
    ax2.plot(hidden_dims, valid_data[group],
             color=colors[group],
             marker=markers[group],
             markersize=12,
             linewidth=2.5,
             label=group)

ax2.set_xlabel('Hidden Dimension', fontsize=13)
ax2.set_ylabel('Validation Loss (Last 20 Iters Avg)', fontsize=13)
ax2.set_title('Validation Loss vs Hidden Dimension (Last 20 Iters Avg)', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_xticks(hidden_dims)
ax2.set_xticklabels(hidden_dims)
plt.tight_layout()
plt.savefig(exp_dir / 'valid_loss_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {exp_dir / 'valid_loss_comparison.png'}")

# Print summary statistics
print("\n" + "="*60)
print("Summary Statistics (Average of Last 20 Iterations)")
print("="*60)
print("\nTraining Losses:")
for group in groups:
    print(f"\n{group.upper()}:")
    for i, dim in enumerate(hidden_dims):
        print(f"  d={dim}: {train_data[group][i]:.6f}")

print("\nValidation Losses:")
for group in groups:
    print(f"\n{group.upper()}:")
    for i, dim in enumerate(hidden_dims):
        print(f"  d={dim}: {valid_data[group][i]:.6f}")
print("="*60)
