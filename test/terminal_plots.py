#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def simple_terminal_plot(epochs, losses, width=60, height=20):
    """Create a simple ASCII plot in the terminal"""
    print(f"\n{'='*70}")
    print("TRAINING LOSS OVER TIME (Terminal Plot)")
    print(f"{'='*70}")

    # Normalize losses to fit in terminal height
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1

    # Print header
    print(f"Loss Range: {min_loss:.6f} to {max_loss:.6f}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):+.2f}%")
    print()

    # Create the plot
    for row in range(height):
        # Calculate loss value for this row (top to bottom = high to low loss)
        row_loss = max_loss - (row * loss_range / (height - 1))

        # Print y-axis label
        print(f"{row_loss:8.4f} │", end="")

        # Plot the line
        for col in range(width):
            if col < len(losses):
                # Determine which epoch this column represents
                epoch_idx = (
                    int(col * len(losses) / width) if width > len(losses) else col
                )
                if epoch_idx < len(losses):
                    loss_val = losses[epoch_idx]

                    # Check if this point should be plotted at this row
                    normalized_pos = (max_loss - loss_val) / loss_range * (height - 1)
                    if abs(normalized_pos - row) < 0.5:
                        print("●", end="")
                    else:
                        print(" ", end="")
                else:
                    print(" ", end="")
            else:
                print(" ", end="")

        print("│")

    # Print x-axis
    print(f"{'':9}└{'─' * width}┘")
    print(f"{'':10}", end="")
    for i in range(0, width, 10):
        epoch_num = int(i * len(epochs) / width) if width > len(epochs) else i
        if epoch_num < len(epochs):
            print(f"{epoch_num:<10}", end="")
        else:
            print(f"{'':10}", end="")
    print()
    print(f"{'':12}{'Epochs':^{width-4}}")


def bar_chart_terminal(epochs, losses, width=50):
    """Create a horizontal bar chart showing loss progression"""
    print(f"\n{'='*70}")
    print("LOSS PROGRESSION BAR CHART")
    print(f"{'='*70}")

    # Take samples if too many epochs
    if len(epochs) > 20:
        step = len(epochs) // 20
        sample_epochs = epochs[::step]
        sample_losses = losses[::step]
    else:
        sample_epochs = epochs
        sample_losses = losses

    # Normalize for bar lengths
    max_loss = max(sample_losses)
    min_loss = min(sample_losses)

    for i, (epoch, loss) in enumerate(zip(sample_epochs, sample_losses)):
        # Calculate bar length
        if max_loss > min_loss:
            bar_length = int((loss - min_loss) / (max_loss - min_loss) * width)
        else:
            bar_length = width // 2

        # Color coding for improvement
        if i == 0:
            bar_char = "█"  # First epoch
        elif i < len(sample_losses) // 3:
            bar_char = "▓"  # Early epochs
        elif i < 2 * len(sample_losses) // 3:
            bar_char = "▒"  # Middle epochs
        else:
            bar_char = "░"  # Later epochs (hopefully lower loss)

        # Print the bar
        bar = bar_char * bar_length
        print(f"Epoch {epoch:3d} │{bar:<{width}} │ {loss:.6f}")

    print(f"{'':9}└{'─' * width}┘")
    print(f"Legend: █ Start ▓ Early ▒ Middle ░ Late")


def sparkline_terminal(epochs, losses):
    """Create a sparkline (mini chart) in terminal"""
    print(f"\n{'='*70}")
    print("SPARKLINE LOSS VISUALIZATION")
    print(f"{'='*70}")

    # Unicode block characters for sparkline
    chars = " ▁▂▃▄▅▆▇█"

    # Normalize losses
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1

    # Create sparkline
    sparkline = ""
    for loss in losses:
        # Normalize to 0-8 range for character index
        normalized = (loss - min_loss) / loss_range
        char_idx = int(normalized * (len(chars) - 1))
        sparkline += chars[char_idx]

    print(f"Loss Trend: {sparkline}")
    print(f"Range: {min_loss:.6f} (low) to {max_loss:.6f} (high)")
    print(f"Epochs: {len(epochs)} total")
    print(f"Trend: {'Decreasing ↓' if losses[-1] < losses[0] else 'Increasing ↑'}")


def detailed_terminal_plot(epochs, losses):
    """Detailed terminal output with statistics"""
    print(f"\n{'='*70}")
    print("DETAILED TRAINING ANALYSIS")
    print(f"{'='*70}")

    # Basic statistics
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    avg_loss = sum(losses) / len(losses)

    # Find best epoch
    best_epoch = epochs[losses.index(min_loss)]

    # Calculate improvement rate
    improvement = initial_loss - final_loss
    improvement_pct = (improvement / initial_loss * 100) if initial_loss != 0 else 0

    print(f"📊 TRAINING SUMMARY")
    print(f"   Total Epochs:     {len(epochs)}")
    print(f"   Initial Loss:     {initial_loss:.6f}")
    print(f"   Final Loss:       {final_loss:.6f}")
    print(f"   Best Loss:        {min_loss:.6f} (Epoch {best_epoch})")
    print(f"   Average Loss:     {avg_loss:.6f}")
    print(f"   Improvement:      {improvement:+.6f} ({improvement_pct:+.2f}%)")

    print(f"\n📈 LOSS TREND")
    # Show trend in chunks
    chunk_size = len(losses) // 10 if len(losses) >= 10 else 1
    for i in range(0, len(losses), chunk_size):
        end_idx = min(i + chunk_size, len(losses))
        chunk_avg = sum(losses[i:end_idx]) / len(losses[i:end_idx])
        epoch_range = f"Epochs {epochs[i]}-{epochs[end_idx-1]}"

        # Visual indicator
        if i == 0:
            indicator = "🔴"  # Start
        elif chunk_avg < losses[0] * 0.8:
            indicator = "🟢"  # Good improvement
        elif chunk_avg < losses[0] * 0.9:
            indicator = "🟡"  # Some improvement
        else:
            indicator = "🟠"  # Little improvement

        print(f"   {indicator} {epoch_range:15} │ Avg Loss: {chunk_avg:.6f}")

    # Performance indicators
    print(f"\n⚡ TRAINING PERFORMANCE")
    if improvement_pct > 50:
        status = "🚀 Excellent convergence!"
    elif improvement_pct > 20:
        status = "✅ Good training progress"
    elif improvement_pct > 5:
        status = "🔄 Moderate improvement"
    elif improvement_pct > 0:
        status = "🐌 Slow convergence"
    else:
        status = "❌ No improvement detected"

    print(f"   Status: {status}")


def test_all_visualizations():
    """Test all visualization methods"""
    # Create test data with decreasing loss
    epochs = list(range(50))
    # Simulate realistic loss decay
    losses = [2.5 * np.exp(-0.05 * x) + 0.1 + 0.05 * np.random.random() for x in epochs]

    # Show all visualization types
    detailed_terminal_plot(epochs, losses)
    simple_terminal_plot(epochs, losses)
    sparkline_terminal(epochs, losses)
    bar_chart_terminal(epochs, losses)


if __name__ == "__main__":
    test_all_visualizations()
