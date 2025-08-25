#!/usr/bin/env python3
"""
High-Dimensional Superdense Decoder
Implements Bell measurement decoding for 8D protocol
"""

from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute
import numpy as np
from typing import Tuple, List


def decode_quantum_state(encoded_circuit: QuantumCircuit, dimension: int = 8) -> Tuple[List[int], float]:
    """
    Decode received quantum state back to classical bits

    Args:
        encoded_circuit: Quantum circuit with encoded message
        dimension: Quantum system dimension

    Returns:
        Tuple of (decoded_bits, fidelity)
    """
    decoder = HighDimensionalSuperdenseDecoder(dimension)
    return decoder.decode_quantum_state(encoded_circuit)


class HighDimensionalSuperdenseDecoder:
    """
    Decode quantum states back to classical bits
    Supports 2D, 4D, and 8D protocols
    """

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.num_qubits = int(np.ceil(np.log2(dimension)))

    def decode_quantum_state(self, encoded_circuit: QuantumCircuit) -> Tuple[List[int], float]:
        """Decode received quantum state back to classical bits"""
        qc = encoded_circuit.copy()

        # Add Bell state measurement for different dimensions
        if self.dimension == 2:
            qc = self._add_2d_measurement(qc)
        elif self.dimension == 4:
            qc = self._add_4d_measurement(qc)
        elif self.dimension == 8:
            qc = self._add_8d_measurement(qc)

        # Execute measurement
        backend = Aer.get_backend('aer_simulator')
        job = execute(qc, backend, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Extract most probable result
        if counts:
            decoded_state = max(counts.keys(), key=counts.get)
            # Take the first half of bits as the decoded message
            message_length = self._get_message_length()
            decoded_bits = [int(bit) for bit in decoded_state[:message_length]]

            # Calculate fidelity based on measurement counts
            fidelity = counts[decoded_state] / 1024
        else:
            # Fallback if no results
            decoded_bits = [0] * self._get_message_length()
            fidelity = 0.0

        return decoded_bits, fidelity

    def _add_2d_measurement(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Add Bell state measurement for 2D protocol"""
        # Standard Bell measurement
        qc.cx(0, 1)
        qc.h(0)

        # Add classical registers and measure
        qc.add_register(ClassicalRegister(2, 'c'))
        qc.measure_all()

        return qc

    def _add_4d_measurement(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Add Bell state measurement for 4D protocol"""
        # 4D Bell measurement
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.h(0)
        qc.h(1)

        # Add classical registers and measure
        qc.add_register(ClassicalRegister(4, 'c'))
        qc.measure_all()

        return qc

    def _add_8d_measurement(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Add Bell state measurement for 8D protocol"""
        # 8D Bell measurement - our innovation
        for i in range(3):
            qc.cx(i, i + 3)

        for i in range(3):
            qc.h(i)

        # Add classical registers and measure
        qc.add_register(ClassicalRegister(6, 'c'))
        qc.measure_all()

        return qc

    def _get_message_length(self) -> int:
        """Get expected message length for dimension"""
        if self.dimension == 2:
            return 2
        elif self.dimension == 4:
            return 2
        elif self.dimension == 8:
            return 3
        else:
            return int(np.log2(self.dimension))


if __name__ == "__main__":
    print("Testing High-Dimensional Superdense Decoding")

    # Test with simple circuit
    from entanglement_generator import create_high_dimensional_bell_state

    dimensions = [2, 4, 8]

    for dim in dimensions:
        print(f"\nTesting {dim}D decoding")

        # Create Bell state
        bell_circuit = create_high_dimensional_bell_state(dim)

        # Decode (without encoding for simplicity)
        decoded_bits, fidelity = decode_quantum_state(bell_circuit, dim)

        print(f"Decoded bits: {decoded_bits}")
        print(f"Fidelity: {fidelity:.3f}")