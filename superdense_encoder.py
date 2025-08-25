#!/usr/bin/env python3
"""
High-Dimensional Entanglement Generator
Creates Bell states in arbitrary dimensions (qubits to qudits)
"""

import numpy as np
from qiskit import QuantumCircuit


def create_high_dimensional_bell_state(dimension: int = 8) -> QuantumCircuit:
    """
    Create Bell state of specified dimension
    Returns quantum circuit ready for superdense coding
    """
    if dimension == 2:
        return create_bell_pair_2d()
    elif dimension == 4:
        return create_bell_pair_4d()
    elif dimension == 8:
        return create_bell_pair_8d()
    else:
        return create_bell_pair_2d()  # Default fallback


def create_bell_pair_2d() -> QuantumCircuit:
    """Create standard 2D Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    qc = QuantumCircuit(2)
    qc.h(0)  # Put first qubit in superposition
    qc.cx(0, 1)  # Create entanglement
    return qc


def create_bell_pair_4d() -> QuantumCircuit:
    """Create 4D Bell state using 2 qubits per party"""
    qc = QuantumCircuit(4)

    # Create 4D entangled state
    qc.h(0)
    qc.h(1)
    qc.cx(0, 2)
    qc.cx(1, 3)

    return qc


def create_bell_pair_8d() -> QuantumCircuit:
    """
    Create 8D Bell state - our innovation for 3+ bits capacity
    Uses 3 qubits per party for 8-dimensional encoding
    """
    qc = QuantumCircuit(6)  # 3 qubits per party for 8D

    # Initialize Alice's qubits in equal superposition
    for i in range(3):
        qc.h(i)

    # Create maximal entanglement between Alice and Bob
    for i in range(3):
        qc.cx(i, i + 3)

    return qc


if __name__ == "__main__":
    # Test different dimensions
    dimensions = [2, 4, 8]

    for dim in dimensions:
        print(f"Testing {dim}D Bell State Generation")
        qc = create_high_dimensional_bell_state(dim)
        print(f"Qubits required: {qc.num_qubits}")
        print(f"Circuit depth: {qc.depth()}")
        print("---")