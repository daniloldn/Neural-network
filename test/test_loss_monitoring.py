#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.loss import mse_loss
from neural_network.network import Network


def test_training_with_loss_monitoring():
    """Test training while monitoring loss progression"""
    print("=" * 70)
    print("TRAINING WITH LOSS MONITORING")
    print("=" * 70)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create training data
    print("Setting up training data...")
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.3]])  # Target closer to 0 for better loss visualization

    print(f"Input: {x}")
    print(f"Target: {y}")

    # Create network
    net = Network(x, depth=1, width=2)
    print(f"Network: depth={net.depth}, width=2")

    # Training parameters
    epochs = 10
    learning_rate = 0.1

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")

    print(f"\n{'-'*70}")
    print("TRAINING PROGRESS")
    print(f"{'-'*70}")
    print(f"{'Epoch':<6} {'Loss':<12} {'Prediction':<15} {'Change':<12}")
    print(f"{'-'*50}")

    # Track loss progression
    losses = []
    predictions = []

    for epoch in range(epochs):
        # Get current prediction and loss
        pred = net.feedforward(x)
        loss = float(mse_loss(y, pred))

        losses.append(loss)
        predictions.append(float(pred[0, 0]))

        # Calculate change from previous epoch
        if epoch == 0:
            change_str = "Initial"
        else:
            change = loss - losses[epoch - 1]
            change_str = f"{change:+.6f}"

        print(f"{epoch:<6} {loss:<12.6f} {pred[0,0]:<15.6f} {change_str:<12}")

        # Train for one epoch
        if epoch < epochs - 1:  # Don't train after the last measurement
            try:
                net.train(x, y, epochs=1, batch_size=1, learning_rate=learning_rate)
            except Exception as e:
                print(f"Training failed at epoch {epoch}: {e}")
                break

    print(f"{'-'*50}")

    # Summary statistics
    print(f"\nTRAINING SUMMARY")
    print(f"{'-'*30}")
    print(f"Initial loss:    {losses[0]:.6f}")
    print(f"Final loss:      {losses[-1]:.6f}")
    print(f"Total reduction: {losses[0] - losses[-1]:.6f}")
    print(f"Percent change:  {((losses[-1] - losses[0]) / losses[0] * 100):+.2f}%")

    print(f"\nInitial prediction: {predictions[0]:.6f}")
    print(f"Final prediction:   {predictions[-1]:.6f}")
    print(f"Target value:       {y[0,0]:.6f}")
    print(f"Final error:        {abs(predictions[-1] - y[0,0]):.6f}")

    # Check if training is working
    if losses[-1] < losses[0]:
        print(f"\n✅ SUCCESS: Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
        print(f"   Training is working correctly! 🎉")
    else:
        print(f"\n⚠️  WARNING: Loss increased or stayed the same")
        print(f"   This might indicate learning rate issues or other problems")

    return losses


def test_different_learning_rates():
    """Test training with different learning rates"""
    print(f"\n{'='*70}")
    print("TESTING DIFFERENT LEARNING RATES")
    print(f"{'='*70}")

    learning_rates = [0.01, 0.1, 0.5, 1.0]

    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        print(f"{'-'*30}")

        # Reset network and data
        np.random.seed(42)  # Same initialization for fair comparison
        x = np.array([[1.0, 2.0]])
        y = np.array([[0.3]])
        net = Network(x, depth=1, width=2)

        # Record initial and final loss after 5 epochs
        initial_pred = net.feedforward(x)
        initial_loss = float(mse_loss(y, initial_pred))

        try:
            # Train for 5 epochs
            for epoch in range(5):
                net.train(x, y, epochs=1, batch_size=1, learning_rate=lr)

            final_pred = net.feedforward(x)
            final_loss = float(mse_loss(y, final_pred))

            improvement = initial_loss - final_loss

            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Final loss:   {final_loss:.6f}")
            print(f"  Improvement:  {improvement:+.6f}")

            if improvement > 0:
                print(f"  Status: ✅ Loss decreased")
            else:
                print(f"  Status: ⚠️ Loss increased/unchanged")

        except Exception as e:
            print(f"  Status: ❌ Training failed: {e}")


def test_different_network_sizes():
    """Test loss monitoring with different network architectures"""
    print(f"\n{'='*70}")
    print("TESTING DIFFERENT NETWORK ARCHITECTURES")
    print(f"{'='*70}")

    configs = [
        {"name": "Small (1x1)", "depth": 1, "width": 1},
        {"name": "Medium (1x3)", "depth": 1, "width": 3},
        {"name": "Deep (2x2)", "depth": 2, "width": 2},
    ]

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"Depth: {config['depth']}, Width: {config['width']}")
        print(f"{'-'*40}")

        try:
            np.random.seed(42)
            x = np.array([[1.0, 2.0]])
            y = np.array([[0.4]])

            net = Network(x, depth=config["depth"], width=config["width"])

            print(f"{'Epoch':<6} {'Loss':<12}")
            print(f"{'-'*18}")

            for epoch in range(5):
                pred = net.feedforward(x)
                loss = float(mse_loss(y, pred))
                print(f"{epoch:<6} {loss:<12.6f}")

                if epoch < 4:  # Train after each measurement except the last
                    net.train(x, y, epochs=1, batch_size=1, learning_rate=0.1)

            print(f"  ✅ {config['name']} completed successfully")

        except Exception as e:
            print(f"  ❌ {config['name']} failed: {e}")


def test_longer_training():
    """Test longer training to see convergence behavior"""
    print(f"\n{'='*70}")
    print("LONGER TRAINING TEST (20 EPOCHS)")
    print(f"{'='*70}")

    np.random.seed(42)
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.2]])  # Even further target for more dramatic loss change

    net = Network(x, depth=1, width=2)

    print(f"Target: {y[0,0]}")
    print(f"\n{'Epoch':<6} {'Loss':<12} {'Prediction':<15} {'Error':<12}")
    print(f"{'-'*55}")

    # Train for 20 epochs, showing every 2nd epoch
    for epoch in range(0, 21, 2):  # 0, 2, 4, ..., 20
        pred = net.feedforward(x)
        loss = float(mse_loss(y, pred))
        error = abs(float(pred[0, 0]) - y[0, 0])

        print(f"{epoch:<6} {loss:<12.6f} {pred[0,0]:<15.6f} {error:<12.6f}")

        if epoch < 20:  # Train 2 epochs unless we're at the end
            for _ in range(2):
                net.train(x, y, epochs=1, batch_size=1, learning_rate=0.1)

    print(f"\n📈 Training completed! Check if prediction approaches target value.")


if __name__ == "__main__":
    print("NEURAL NETWORK TRAINING LOSS MONITORING")
    print("=" * 70)

    # Run all tests
    losses = test_training_with_loss_monitoring()
    test_different_learning_rates()
    test_different_network_sizes()
    test_longer_training()

    print(f"\n{'='*70}")
    print("MONITORING COMPLETE")
    print(f"{'='*70}")
    print("Check the results above to see:")
    print("  1. Is loss decreasing during training?")
    print("  2. How do different learning rates affect convergence?")
    print("  3. Do different network architectures train successfully?")
    print("  4. Does longer training lead to better convergence?")
    print(f"\nIf losses are decreasing, your training function is working! 🚀")
