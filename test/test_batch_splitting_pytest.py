import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network


def test_split_batch_normal_case():
    """Test _split_batch with normal batch size."""
    x = np.array([[i, i + 1] for i in range(20)])  # 20 observations
    net = Network(x, depth=1, width=3)

    batch_size = 5
    batches = net._split_batch(batch_size, x)

    # Verify basic properties
    assert len(batches) == 4  # 20/5 = 4 batches
    assert sum(len(batch) for batch in batches) == len(x)  # All data preserved
    assert all(len(batch) == 5 for batch in batches)  # Equal batch sizes


def test_split_batch_large_batch_size():
    """Test _split_batch when batch size is larger than data."""
    x = np.array([[i, i + 1] for i in range(5)])  # 5 observations
    net = Network(x, depth=1, width=3)

    batch_size = 10
    batches = net._split_batch(batch_size, x)

    assert len(batches) == 1  # Only one batch
    assert len(batches[0]) == len(x)  # Contains all data


def test_split_batch_size_one():
    """Test _split_batch with batch size of 1."""
    x = np.array([[i, i + 1] for i in range(5)])
    net = Network(x, depth=1, width=3)

    batch_size = 1
    batches = net._split_batch(batch_size, x)

    assert len(batches) == len(x)  # One batch per sample
    assert all(len(batch) == 1 for batch in batches)  # Each batch has one sample


def test_split_batch_data_integrity():
    """Test that _split_batch preserves all data after shuffling."""
    x = np.array([[i, i + 1] for i in range(10)])
    net = Network(x, depth=1, width=3)

    batches = net._split_batch(3, x)
    reconstructed = np.vstack(batches)

    # Sort both arrays for comparison (since order is shuffled)
    original_sorted = np.sort(x.view(np.void), axis=0).view(np.int_).reshape(x.shape)
    reconstructed_sorted = (
        np.sort(reconstructed.view(np.void), axis=0)
        .view(np.int_)
        .reshape(reconstructed.shape)
    )

    assert reconstructed.shape == x.shape
    assert np.array_equal(original_sorted, reconstructed_sorted)


def test_split_batch_shuffles_data():
    """Test that _split_batch shuffles the data."""
    x = np.array([[i, i + 1] for i in range(20)])
    net = Network(x, depth=1, width=3)

    # Run multiple times and check if we get different first elements
    first_elements = []
    for _ in range(10):
        batches = net._split_batch(5, x)
        first_element = batches[0][0, 0]  # First element of first batch
        first_elements.append(first_element)

    # We should get some variation in first elements (not always the same)
    unique_elements = len(set(first_elements))
    assert unique_elements > 1, "Should see some variation due to shuffling"


def test_split_batch_uneven_division():
    """Test _split_batch when data doesn't divide evenly into batches."""
    x = np.array([[i, i + 1] for i in range(7)])  # 7 observations
    net = Network(x, depth=1, width=3)

    batch_size = 3
    batches = net._split_batch(batch_size, x)

    # Should create round(7/3) = 2 batches
    assert len(batches) == 2
    # Total data should be preserved
    assert sum(len(batch) for batch in batches) == len(x)
    # Batches might have different sizes due to uneven split
    batch_sizes = [len(batch) for batch in batches]
    assert all(size > 0 for size in batch_sizes)  # No empty batches


def test_train_method_executes():
    """Test that train method runs without errors using batch splitting."""
    x = np.array([[i, i + 1] for i in range(6)])
    y = np.array([[i] for i in range(6)])

    net = Network(x, depth=1, width=3)

    # This should run without errors (currently just prints batches)
    try:
        net.train(x, y, epochs=1, batch_size=2, learning_rate=0.01)
    except Exception as e:
        pytest.fail(f"Train method should not raise exception: {e}")


def test_split_batch_returns_numpy_arrays():
    """Test that _split_batch returns numpy arrays, not lists."""
    x = np.array([[i, i + 1] for i in range(8)])
    net = Network(x, depth=1, width=3)

    batches = net._split_batch(3, x)

    # Each batch should be a numpy array
    assert all(isinstance(batch, np.ndarray) for batch in batches)
    # Each batch should have the same number of features as original
    assert all(batch.shape[1] == x.shape[1] for batch in batches)
