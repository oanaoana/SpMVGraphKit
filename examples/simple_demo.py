#!/usr/bin/env python3
"""
Simple Demo - Shows basic usage of SpMVGraphKit

This is a minimal example showing how to:
1. Load/generate a matrix
2. Apply reordering
3. Check improvement
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.loader import generate_random_sparse_matrix
from analysis.mat_specs import estimate_bandwidth
from reordering.coloring import multi_strategy_coloring

def simple_example():
    """Simple 10-line example."""
    # Generate test matrix
    matrix = generate_random_sparse_matrix(100, 100, 0.05)

    # Check original bandwidth
    original_bw = estimate_bandwidth(matrix)
    print(f"Original bandwidth: {original_bw:,}")

    # Apply RCM reordering
    results = multi_strategy_coloring(matrix, strategies=['rcm'])

    # Check improvement
    if 'rcm' in results and 'matrix' in results['rcm']:
        new_bw = estimate_bandwidth(results['rcm']['matrix'])
        improvement = (original_bw - new_bw) / original_bw * 100
        print(f"RCM bandwidth: {new_bw:,} ({improvement:+.1f}% improvement)")
    else:
        print("RCM reordering failed")

if __name__ == "__main__":
    simple_example()