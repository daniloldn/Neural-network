#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def simple_terminal_graph(epochs, losses, width=60, height=15):
    """Create a simple ASCII graph in the terminal"""
    print(f"\n{'='*70}")
    print("TRAINING LOSS GRAPH")
    print(f"{'='*70}")

    if not losses or len(losses) == 0:
        print("No loss data to display")
        return

    # Basic stats
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1

    print(f"Epochs: {len(epochs)}")
    print(f"Initial Loss: {losses[0]:.6f}")
    print(f"Final Loss:   {losses[-1]:.6f}")
    print(f"Best Loss:    {min_loss:.6f}")
    print(f"Improvement:  {((losses[0] - losses[-1]) / losses[0] * 100):+.1f}%")
    print()

    # Create the graph
    for row in range(height):
        # Calculate loss value for this row (top = high loss, bottom = low loss)
        row_loss = max_loss - (row * loss_range / (height - 1))

        # Print y-axis label
        print(f"{row_loss:8.4f} │", end="")

        # Plot points
        for col in range(width):
            if col < len(losses):
                loss_val = losses[col] if col < len(losses) else losses[-1]

                # Calculate where this loss should appear vertically
                normalized_pos = (max_loss - loss_val) / loss_range * (height - 1)

                # If this loss value is close to current row, plot it
                if abs(normalized_pos - row) < 0.7:
                    print("*", end="")
                elif abs(normalized_pos - row) < 1.0:
                    print(".", end="")
                else:
                    print(" ", end="")
            else:
                print(" ", end="")

        print("│")

    # Print x-axis
    print(f"{'':9}└{'─' * width}┘")
    print(f"{'':10}0{'':<{width-10}}{len(epochs)-1}")
    print(f"{'':15}Epochs")


def loss_table_view(epochs, losses):
    """Show loss data in a clean table format"""
    print(f"\n{'='*50}")
    print("TRAINING PROGRESS TABLE")
    print(f"{'='*50}")
    print(f"{'Epoch':<8} {'Loss':<12} {'Change':<10} {'% Change':<10}")
    print("-" * 50)

    # Show every 5th epoch if many epochs, otherwise show all
    step = max(1, len(epochs) // 20)

    for i in range(0, len(epochs), step):
        epoch = epochs[i]
        loss = losses[i]

        if i == 0:
            change = 0.0
            pct_change = 0.0
        else:
            prev_i = max(0, i - step)
            change = loss - losses[prev_i]
            pct_change = (change / losses[prev_i] * 100) if losses[prev_i] != 0 else 0

        # Color coding with symbols
        if i == 0:
            symbol = "🔴"  # Start
        elif change < -0.001:
            symbol = "🟢"  # Good decrease
        elif change < 0:
            symbol = "🟡"  # Small decrease
        else:
            symbol = "🟠"  # Increase or no change

        print(f"{epoch:<8} {loss:<12.6f} {change:+10.6f} {pct_change:+10.2f}% {symbol}")

    # Final summary
    print("-" * 50)
    final_change = losses[-1] - losses[0]
    final_pct = (final_change / losses[0] * 100) if losses[0] != 0 else 0
    print(f"{'TOTAL':<8} {losses[-1]:<12.6f} {final_change:+10.6f} {final_pct:+10.2f}%")


def progress_bar_view(epochs, losses):
    """Show progress as horizontal bars"""
    print(f"\n{'='*60}")
    print("LOSS REDUCTION PROGRESS")
    print(f"{'='*60}")

    if len(losses) < 2:
        print("Need at least 2 epochs to show progress")
        return

    # Group epochs into chunks for visualization
    num_bars = min(20, len(losses))
    chunk_size = len(losses) // num_bars

    max_loss = max(losses)
    min_loss = min(losses)

    for i in range(num_bars):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(losses))

        # Average loss for this chunk
        chunk_losses = losses[start_idx:end_idx]
        avg_loss = sum(chunk_losses) / len(chunk_losses)

        # Calculate bar length (inverse - lower loss = longer bar = better)
        if max_loss > min_loss:
            # Invert so lower loss gives longer bar
            progress = (max_loss - avg_loss) / (max_loss - min_loss)
        else:
            progress = 0.5

        bar_length = int(progress * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)

        epoch_range = (
            f"{epochs[start_idx]}-{epochs[end_idx-1]}"
            if start_idx != end_idx - 1
            else f"{epochs[start_idx]}"
        )

        print(f"E{epoch_range:>6} │{bar}│ {avg_loss:.6f}")

    print(f"{'':8}└{'─' * 40}┘")
    print(f"{'':10}Better ←{'─' * 32}→ Worse")


def mini_dashboard(epochs, losses):
    """A compact dashboard view"""
    print(f"\n╔{'═' * 68}╗")
    print(f"║{'NEURAL NETWORK TRAINING DASHBOARD':^68}║")
    print(f"╚{'═' * 68}╝")

    # Quick stats in a box
    initial = losses[0]
    final = losses[-1]
    improvement = initial - final
    improvement_pct = (improvement / initial * 100) if initial != 0 else 0
    best = min(losses)

    # Status determination
    if improvement_pct > 30:
        status = "🚀 EXCELLENT"
        status_color = "🟢"
    elif improvement_pct > 10:
        status = "✅ GOOD"
        status_color = "🟡"
    elif improvement_pct > 0:
        status = "📈 IMPROVING"
        status_color = "🟠"
    else:
        status = "⚠️  NO PROGRESS"
        status_color = "🔴"

    print(f"┌─────────────────┬──────────────┬─────────────┬──────────────┐")
    print(
        f"│ Epochs: {len(epochs):>7} │ Initial: {initial:>8.4f} │ Final: {final:>8.4f} │ Best: {best:>9.4f} │"
    )
    print(f"├─────────────────┼──────────────┼─────────────┼──────────────┤")
    print(
        f"│ Status: {status:<7} │ Improvement: {improvement_pct:>6.1f}% │ Change: {improvement:>7.4f} │ {status_color} │"
    )
    print(f"└─────────────────┴──────────────┴─────────────┴──────────────┘")

    # Mini sparkline
    print(f"\nLoss Trend: ", end="")
    chars = "▁▂▃▄▅▆▇█"
    if len(losses) > 1:
        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1

        # Sample points if too many
        sample_size = min(50, len(losses))
        step = len(losses) // sample_size
        sampled = [losses[i] for i in range(0, len(losses), max(1, step))]

        for loss in sampled[:50]:  # Max 50 characters
            normalized = (loss - min_loss) / loss_range if loss_range > 0 else 0
            char_idx = min(len(chars) - 1, int(normalized * (len(chars) - 1)))
            print(chars[char_idx], end="")

    print(f" ({len(epochs)} epochs)")
    print("(Lower is better) ▁▂▃▄▅▆▇█ (Higher is worse)")


def show_training_plot(epochs, losses):
    """Main function to display training visualization"""
    if not epochs or not losses:
        print("No training data available to plot")
        return

    # Show different views
    mini_dashboard(epochs, losses)
    simple_terminal_graph(epochs, losses)
    loss_table_view(epochs, losses)
    progress_bar_view(epochs, losses)


# Test with sample data
if __name__ == "__main__":
    # Create sample decreasing loss data
    import random

    random.seed(42)

    epochs = list(range(25))
    losses = []
    loss = 1.5

    for i in range(25):
        # Simulate decreasing loss with some noise
        loss = loss * 0.95 + random.uniform(-0.02, 0.01)
        losses.append(max(0.01, loss))  # Don't go below 0.01

    print("DEMO: Terminal Plotting for Neural Network Training")
    show_training_plot(epochs, losses)
