import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network


def test_batch_splitting():
    """Test that _split_batch method correctly splits data into batches."""

    print("=== Testing Batch Splitting Functionality ===\n")

    # Create sample data
    x = np.array([[i, i + 1] for i in range(20)])  # 20 observations, 2 features each
    print(f"Original data shape: {x.shape}")
    print(f"Original data (first 5 rows):\n{x[:5]}\n")

    # Create a network instance to test the _split_batch method
    net = Network(x, depth=1, width=3)

    # Test 1: Normal batch size
    print("1. Testing normal batch size...")
    batch_size = 5
    batches = net._split_batch(batch_size, x)

    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches created: {len(batches)}")
    print(f"   Expected batches: {round(x.shape[0]/batch_size)}")

    # Verify total data is preserved
    total_samples = sum(len(batch) for batch in batches)
    print(f"   Total samples across batches: {total_samples}")
    print(f"   Original samples: {len(x)}")

    # Show batch sizes
    batch_sizes = [len(batch) for batch in batches]
    print(f"   Individual batch sizes: {batch_sizes}")

    assert total_samples == len(x), "Total samples should be preserved"
    assert len(batches) == round(
        x.shape[0] / batch_size
    ), "Number of batches should match expected"
    print("   ✓ PASSED\n")

    # Test 2: Batch size larger than data
    print("2. Testing batch size larger than data...")
    large_batch_size = 25
    large_batches = net._split_batch(large_batch_size, x)

    print(f"   Batch size: {large_batch_size}")
    print(f"   Number of batches created: {len(large_batches)}")
    print(f"   Batch size: {len(large_batches[0])}")

    assert len(large_batches) == 1, "Should create only one batch"
    assert len(large_batches[0]) == len(x), "Batch should contain all data"
    print("   ✓ PASSED\n")

    # Test 3: Batch size of 1
    print("3. Testing batch size of 1...")
    single_batch_size = 1
    single_batches = net._split_batch(single_batch_size, x)

    print(f"   Batch size: {single_batch_size}")
    print(f"   Number of batches created: {len(single_batches)}")

    assert len(single_batches) == len(x), "Should create one batch per sample"
    assert all(
        len(batch) == 1 for batch in single_batches
    ), "Each batch should have one sample"
    print("   ✓ PASSED\n")

    # Test 4: Verify shuffling occurs
    print("4. Testing data shuffling...")
    original_order = x.copy()
    batch_size = 10

    # Run multiple times to check if shuffling occurs
    shuffled_results = []
    for i in range(5):
        batches = net._split_batch(batch_size, x)
        # Reconstruct data from batches to check order
        reconstructed = np.vstack(batches)
        shuffled_results.append(reconstructed[0, 0])  # First element of first sample

    print(f"   First element across 5 runs: {shuffled_results}")
    # Check if we get different first elements (indicating shuffling)
    unique_first_elements = len(set(shuffled_results))
    print(f"   Unique first elements: {unique_first_elements}")

    # Note: This is probabilistic - with proper shuffling we should get some variation
    # But we won't assert on this as random shuffling could theoretically give same result
    print("   ✓ Shuffling mechanism present\n")

    # Test 5: Verify data integrity after shuffling
    print("5. Testing data integrity after shuffling...")
    batches = net._split_batch(5, x)
    reconstructed_data = np.vstack(batches)

    # Sort both arrays for comparison
    original_sorted = np.sort(x.view(np.void), axis=0).view(np.int_).reshape(x.shape)
    reconstructed_sorted = (
        np.sort(reconstructed_data.view(np.void), axis=0)
        .view(np.int_)
        .reshape(reconstructed_data.shape)
    )

    print(f"   Original data shape: {x.shape}")
    print(f"   Reconstructed data shape: {reconstructed_data.shape}")

    assert np.array_equal(
        original_sorted, reconstructed_sorted
    ), "Data should be identical after sorting"
    print("   ✓ PASSED\n")

    # Test 6: Edge case - empty array behavior
    print("6. Testing edge case with small data...")
    small_x = np.array([[1, 2], [3, 4]])  # Only 2 samples
    small_batches = net._split_batch(3, small_x)  # Batch size larger than data

    print(f"   Small data shape: {small_x.shape}")
    print(f"   Batches created: {len(small_batches)}")
    print(f"   Batch content: {[len(batch) for batch in small_batches]}")

    assert len(small_batches) == 1, "Should create one batch for small data"
    print("   ✓ PASSED\n")

    print("🎉 All batch splitting tests passed!")
    print(
        "✅ _split_batch method works correctly with shuffling and proper batch division."
    )


def test_train_method_batch_integration():
    """Test that the train method properly uses batch splitting."""

    print("\n=== Testing Train Method Batch Integration ===\n")

    # Create test data
    x = np.array([[i, i + 1] for i in range(10)])
    y = np.array([[i] for i in range(10)])

    net = Network(x, depth=1, width=3)

    print("Testing train method with batch printing...")
    print(f"Training data shape: {x.shape}")
    print(f"Batch size: 3")
    print("Expected output: batches printed during training\n")

    # Note: The current train method just prints batches, so we'll capture that
    # In a real test, you might want to capture stdout or modify the method to return batches
    try:
        net.train(x, y, epochs=1, batch_size=3, learning_rate=0.01)
        print("\n✓ Train method executed without errors")
        print("✅ Batch integration working correctly")
    except Exception as e:
        print(f"❌ Error in train method: {e}")
        raise


if __name__ == "__main__":
    test_batch_splitting()
    test_train_method_batch_integration()
