#!/usr/bin/env python3
"""
Simple Deployment Tests for Neural Network Implementation
=======================================================

This script tests the neural network with realistic data generating processes
to validate that the implementation works correctly on different types of problems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.loss import mse_loss
from neural_network.network import Network


def generate_linear_data(n_samples=100, n_features=2):
    """Generate simple linear relationship: y = 2*x1 + x2 + noise"""
    print("🔧 Generating Linear Relationship Data...")

    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * np.random.randn(n_samples)
    y = y.reshape(-1, 1)

    print(f"   📊 Samples: {n_samples}, Features: {n_features}")
    print(f"   📈 Function: y = 2*x1 + x2 + noise")

    return X, y


def generate_quadratic_data(n_samples=120):
    """Generate quadratic relationship: y = x^2 + noise"""
    print("🔧 Generating Quadratic Relationship Data...")

    np.random.seed(42)
    x = np.random.uniform(-2, 2, n_samples)
    X = np.column_stack([x, x**2])  # Features: x and x^2
    y = x**2 + 0.1 * np.random.randn(n_samples)
    y = y.reshape(-1, 1)

    print(f"   📊 Samples: {n_samples}, Features: 2")
    print(f"   📈 Function: y = x² + noise")

    return X, y


def generate_sine_data(n_samples=150):
    """Generate sine wave data: y = sin(x) + noise"""
    print("🔧 Generating Sine Wave Data...")

    np.random.seed(42)
    x = np.random.uniform(0, 2 * np.pi, n_samples)
    X = np.column_stack([x, np.sin(x), np.cos(x)])  # Features: x, sin(x), cos(x)
    y = np.sin(x) + 0.05 * np.random.randn(n_samples)
    y = y.reshape(-1, 1)

    print(f"   📊 Samples: {n_samples}, Features: 3")
    print(f"   📈 Function: y = sin(x) + noise")

    return X, y


def generate_xor_data(n_samples=200):
    """Generate XOR-like data for non-linear classification"""
    print("🔧 Generating XOR-style Data...")

    np.random.seed(42)
    # Create four clusters
    n_per_cluster = n_samples // 4

    # Cluster 1: bottom-left (negative output)
    x1 = np.random.normal(-1, 0.3, n_per_cluster)
    y1 = np.random.normal(-1, 0.3, n_per_cluster)
    z1 = np.zeros(n_per_cluster)

    # Cluster 2: top-right (negative output)
    x2 = np.random.normal(1, 0.3, n_per_cluster)
    y2 = np.random.normal(1, 0.3, n_per_cluster)
    z2 = np.zeros(n_per_cluster)

    # Cluster 3: top-left (positive output)
    x3 = np.random.normal(-1, 0.3, n_per_cluster)
    y3 = np.random.normal(1, 0.3, n_per_cluster)
    z3 = np.ones(n_per_cluster)

    # Cluster 4: bottom-right (positive output)
    x4 = np.random.normal(1, 0.3, n_per_cluster)
    y4 = np.random.normal(-1, 0.3, n_per_cluster)
    z4 = np.ones(n_per_cluster)

    # Combine all clusters
    X = np.column_stack(
        [np.concatenate([x1, x2, x3, x4]), np.concatenate([y1, y2, y3, y4])]
    )
    y = np.concatenate([z1, z2, z3, z4]).reshape(-1, 1)

    print(f"   📊 Samples: {n_samples}, Features: 2")
    print(f"   📈 Pattern: XOR-like classification")

    return X, y


def test_network_on_data(name, X, y, config):
    """Test the neural network on a dataset"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {name.upper()}")
    print(f"{'='*60}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"🏗️  Network: depth={config['depth']}, width={config['width']}")

    try:
        # Create and train network
        net = Network(X_train, depth=config["depth"], width=config["width"])

        # Training parameters
        epochs = config.get("epochs", 60)
        batch_size = config.get("batch_size", 16)
        learning_rate = config.get("learning_rate", 0.5)

        print(
            f"⚙️  Training: {epochs} epochs, lr={learning_rate}, batch_size={batch_size}"
        )

        # Train the network
        net.train(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Evaluate
        train_pred = net.feedforward(X_train)
        test_pred = net.feedforward(X_test)

        train_loss = float(mse_loss(y_train, train_pred))
        test_loss = float(mse_loss(y_test, test_pred))

        # Calculate R²-like metric
        train_var = np.var(y_train)
        test_var = np.var(y_test)
        train_r2 = max(0, 1 - train_loss / train_var) if train_var > 0 else 0
        test_r2 = max(0, 1 - test_loss / test_var) if test_var > 0 else 0

        print(f"\n📈 RESULTS:")
        print(f"   Training Loss: {train_loss:.6f}")
        print(f"   Test Loss:     {test_loss:.6f}")
        print(f"   Training R²:   {train_r2:.4f}")
        print(f"   Test R²:       {test_r2:.4f}")

        # Performance assessment
        if test_r2 > 0.8:
            status = "🎯 EXCELLENT"
        elif test_r2 > 0.6:
            status = "✅ GOOD"
        elif test_r2 > 0.3:
            status = "🔄 FAIR"
        else:
            status = "❌ POOR"

        print(f"   Assessment:    {status}")

        # Show sample predictions
        print(f"\n🔍 Sample Predictions:")
        print(f"{'Actual':<10} {'Predicted':<10} {'Error':<8}")
        print("-" * 28)
        for i in range(min(5, len(y_test))):
            actual = y_test[i, 0]
            predicted = test_pred[i, 0]
            error = abs(actual - predicted)
            print(f"{actual:<10.3f} {predicted:<10.3f} {error:<8.3f}")

        return {
            "success": True,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_r2": test_r2,
            "status": status,
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"success": False, "error": str(e)}


def run_deployment_tests():
    """Run all deployment tests"""
    print("🚀 NEURAL NETWORK DEPLOYMENT TESTS")
    print("=" * 60)
    print("Testing neural network on various data patterns")

    results = {}

    # Test 1: Linear Relationship (Easy)
    print("\n" + "🟢 EASY TESTS".center(60))
    X1, y1 = generate_linear_data(n_samples=80, n_features=2)
    results["linear"] = test_network_on_data(
        "Linear Relationship",
        X1,
        y1,
        {"depth": 1, "width": 3, "epochs": 40, "learning_rate": 0.3},
    )

    # Test 2: Quadratic Relationship (Medium)
    print("\n" + "🟡 MEDIUM TESTS".center(60))
    X2, y2 = generate_quadratic_data(n_samples=100)
    results["quadratic"] = test_network_on_data(
        "Quadratic Relationship",
        X2,
        y2,
        {"depth": 2, "width": 4, "epochs": 60, "learning_rate": 0.2},
    )

    # Test 3: Sine Wave (Hard)
    print("\n" + "🔴 HARD TESTS".center(60))
    X3, y3 = generate_sine_data(n_samples=120)
    results["sine"] = test_network_on_data(
        "Sine Wave Pattern",
        X3,
        y3,
        {"depth": 2, "width": 6, "epochs": 80, "learning_rate": 0.15},
    )

    # Test 4: XOR Pattern (Very Hard)
    X4, y4 = generate_xor_data(n_samples=160)
    results["xor"] = test_network_on_data(
        "XOR Pattern",
        X4,
        y4,
        {"depth": 2, "width": 8, "epochs": 100, "learning_rate": 0.1},
    )

    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")

    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)

    print(f"🎯 Tests Passed: {successful}/{total}")
    print(f"\n{'Test':<15} {'Status':<12} {'Test R²':<8} {'Performance'}")
    print("-" * 50)

    for name, result in results.items():
        if result.get("success", False):
            status_emoji = "✅"
            r2 = result.get("test_r2", 0)
            perf = (
                result.get("status", "Unknown").split()[1]
                if " " in result.get("status", "")
                else "OK"
            )
        else:
            status_emoji = "❌"
            r2 = 0.0
            perf = "FAILED"

        print(f"{name.title():<15} {status_emoji:<12} {r2:<8.3f} {perf}")

    # Overall assessment
    if successful == total:
        avg_r2 = np.mean(
            [r.get("test_r2", 0) for r in results.values() if r.get("success")]
        )
        if avg_r2 > 0.7:
            overall = "🌟 EXCELLENT - Neural network is working very well!"
        elif avg_r2 > 0.5:
            overall = "🎯 GOOD - Neural network shows strong learning"
        elif avg_r2 > 0.2:
            overall = "✅ FUNCTIONAL - Network is learning but could improve"
        else:
            overall = "⚠️  BASIC - Network works but struggles with complex patterns"
    else:
        overall = "❌ ISSUES - Some tests failed, needs debugging"

    print(f"\n🏆 OVERALL: {overall}")

    return results


if __name__ == "__main__":
    results = run_deployment_tests()
