"""
Enhanced Quantum Circuit Simulator with Educational Insights - Complete Implementation
===================================================================================

This module provides comprehensive quantum circuit simulation capabilities using Qiskit
with advanced educational features, BB84 protocol integration, and detailed
visualizations with educational feedback.

Features:
- Complete gate library with all implementations
- Real-time statevector calculation with visualization  
- Circuit diagram generation with professional styling
- BB84-specific gates and protocol templates
- Advanced measurement analysis and statistics
- Comprehensive educational insights system
- Adaptive learning recommendations
- Multi-backend compatibility with automatic fallbacks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# Qiskit imports with error handling
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.visualization import circuit_drawer, plot_histogram, plot_bloch_multivector
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
    from qiskit.circuit.library import UnitaryGate
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Some features will be limited.")

# Import BlochSphereVisualizer (assuming it's in the same directory)
try:
    from bloch_sphere_visualizer import BlochSphereVisualizer
    BLOCH_VISUALIZER_AVAILABLE = True
except ImportError:
    BLOCH_VISUALIZER_AVAILABLE = False
    print("Warning: BlochSphereVisualizer not available.")

logger = logging.getLogger(__name__)


class EnhancedQuantumCircuitSimulator:
    """
    Enhanced quantum circuit simulator with advanced visualization and comprehensive educational features
    """
    
    def __init__(self, backend='qiskit'):
        """Initialize the enhanced circuit simulator"""
        self.simulator = AerSimulator() if QISKIT_AVAILABLE else None
        self.backend = backend
        self.max_qubits = 15
        self.max_shots = 10000
        
        # Initialize BlochSphere visualizer if available
        if BLOCH_VISUALIZER_AVAILABLE:
            self.bloch_visualizer = BlochSphereVisualizer(style_theme='educational')
        else:
            self.bloch_visualizer = None
        
        # Enhanced gate definitions with metadata
        self.supported_gates = {
            # Basic single-qubit gates
            'H': {
                'name': 'Hadamard',
                'description': 'Creates superposition',
                'matrix': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_hadamard
            },
            'X': {
                'name': 'Pauli-X (NOT)',
                'description': 'Bit flip operation',
                'matrix': np.array([[0, 1], [1, 0]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_pauli_x
            },
            'Y': {
                'name': 'Pauli-Y',
                'description': 'Bit and phase flip',
                'matrix': np.array([[0, -1j], [1j, 0]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_pauli_y
            },
            'Z': {
                'name': 'Pauli-Z',
                'description': 'Phase flip operation',
                'matrix': np.array([[1, 0], [0, -1]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_pauli_z
            },
            'I': {
                'name': 'Identity',
                'description': 'No operation',
                'matrix': np.array([[1, 0], [0, 1]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_identity
            },
            # Two-qubit gates
            'CNOT': {
                'name': 'Controlled-NOT',
                'description': 'Two-qubit entangling gate',
                'parameters': ['control', 'target'],
                'qubits_required': 2,
                'implementation': self._add_cnot
            },
            'CZ': {
                'name': 'Controlled-Z',
                'description': 'Controlled phase flip',
                'parameters': ['control', 'target'],
                'qubits_required': 2,
                'implementation': self._add_cz
            },
            'SWAP': {
                'name': 'SWAP',
                'description': 'Exchange two qubits',
                'parameters': ['qubit1', 'qubit2'],
                'qubits_required': 2,
                'implementation': self._add_swap
            },
            # Rotation gates
            'RX': {
                'name': 'X-Rotation',
                'description': 'Rotation around X-axis',
                'parameters': ['qubit', 'angle'],
                'qubits_required': 1,
                'implementation': self._add_rotation_x
            },
            'RY': {
                'name': 'Y-Rotation',
                'description': 'Rotation around Y-axis',
                'parameters': ['qubit', 'angle'],
                'qubits_required': 1,
                'implementation': self._add_rotation_y
            },
            'RZ': {
                'name': 'Z-Rotation',
                'description': 'Rotation around Z-axis',
                'parameters': ['qubit', 'angle'],
                'qubits_required': 1,
                'implementation': self._add_rotation_z
            },
            # Phase gates
            'S': {
                'name': 'S Gate',
                'description': 'Phase gate (π/2)',
                'matrix': np.array([[1, 0], [0, 1j]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_s_gate
            },
            'T': {
                'name': 'T Gate',
                'description': 'Phase gate (π/4)',
                'matrix': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]),
                'parameters': ['qubit'],
                'qubits_required': 1,
                'implementation': self._add_t_gate
            },
            # BB84 specific gates
            'PREPARE_0': {
                'name': 'Prepare |0⟩',
                'description': 'BB84 computational 0 state',
                'parameters': ['qubit'],
                'qubits_required': 1,
                'bb84_relevant': True,
                'implementation': self._prepare_computational_0
            },
            'PREPARE_1': {
                'name': 'Prepare |1⟩',
                'description': 'BB84 computational 1 state',
                'parameters': ['qubit'],
                'qubits_required': 1,
                'bb84_relevant': True,
                'implementation': self._prepare_computational_1
            },
            'PREPARE_PLUS': {
                'name': 'Prepare |+⟩',
                'description': 'BB84 superposition + state',
                'parameters': ['qubit'],
                'qubits_required': 1,
                'bb84_relevant': True,
                'implementation': self._prepare_superposition_plus
            },
            'PREPARE_MINUS': {
                'name': 'Prepare |−⟩',
                'description': 'BB84 superposition - state',
                'parameters': ['qubit'],
                'qubits_required': 1,
                'bb84_relevant': True,
                'implementation': self._prepare_superposition_minus
            }
        }
        
        # Circuit templates for famous algorithms
        self.algorithm_templates = {
            'bell_state': self._create_bell_state_template,
            'ghz_state': self._create_ghz_state_template,
            'quantum_teleportation': self._create_teleportation_template,
            'deutsch_jozsa': self._create_deutsch_jozsa_template,
            'grover_search': self._create_grover_template,
            'quantum_fourier_transform': self._create_qft_template
        }
    
    def simulate_circuit(self, num_qubits: int, gates: List[Dict], shots: int = 1024,
                        include_statevector: bool = True, include_density_matrix: bool = False,
                        include_bloch_sphere: bool = True, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced circuit simulation with comprehensive analysis and visualization
        
        Args:
            num_qubits: Number of qubits in the circuit
            gates: List of gate specifications
            shots: Number of measurement shots
            include_statevector: Whether to calculate statevector
            include_density_matrix: Whether to calculate density matrix
            include_bloch_sphere: Whether to generate Bloch sphere visualization
            user_context: User context for personalized insights
            
        Returns:
            dict: Comprehensive simulation results with visualizations and educational insights
        """
        try:
            # Validation
            if not QISKIT_AVAILABLE:
                return {'error': 'Qiskit not available', 'success': False}
            
            validation_result = self.validate_circuit(num_qubits, gates)
            if not validation_result[0]:
                return {'error': validation_result[1], 'success': False}
            
            if shots > self.max_shots:
                shots = self.max_shots
                logger.warning(f"Shots limited to {self.max_shots}")
            
            # Create quantum circuit
            qc = QuantumCircuit(num_qubits, num_qubits)
            
            # Add gates to circuit
            measurement_operations = []
            for gate in gates:
                if gate.get('type', '').startswith('MEASURE'):
                    measurement_operations.append(gate)
                else:
                    self._add_gate_to_circuit(qc, gate)
            
            # Handle measurements
            if measurement_operations:
                for meas_op in measurement_operations:
                    self._add_measurement_to_circuit(qc, meas_op)
            else:
                qc.measure_all()
            
            # Calculate statevector (before measurement)
            statevector_data = None
            pure_circuit = None
            
            if include_statevector:
                pure_circuit = QuantumCircuit(num_qubits)
                for gate in gates:
                    if not gate.get('type', '').startswith('MEASURE'):
                        self._add_gate_to_circuit(pure_circuit, gate)
                
                try:
                    statevector = Statevector.from_instruction(pure_circuit)
                    statevector_data = [complex(amp) for amp in statevector.data]
                except Exception as e:
                    logger.warning(f"Statevector calculation failed: {str(e)}")
                    statevector_data = None
            
            # Run measurement simulation
            job = self.simulator.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate probabilities
            probabilities = {state: count/shots for state, count in counts.items()}
            
            # Generate enhanced visualizations
            visualizations = self._generate_enhanced_visualizations(
                qc, pure_circuit, counts, statevector_data, num_qubits, 
                include_bloch_sphere
            )
            
            # Calculate advanced metrics
            metrics = self._calculate_advanced_metrics(
                qc, statevector_data, counts, shots, num_qubits
            )
            
            # Generate ENHANCED educational insights (NEW)
            insights = self._generate_enhanced_educational_insights(
                gates, statevector_data, counts, num_qubits, user_context or {}
            )
            
            return {
                'simulation_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'backend': 'Qiskit',
                    'num_qubits': num_qubits,
                    'shots': shots,
                    'circuit_depth': qc.depth(),
                    'gate_count': qc.size()
                },
                'statevector': self._serialize_complex_list(statevector_data) if statevector_data else None,
                'counts': counts,
                'probabilities': probabilities,
                'metrics': metrics,
                'visualizations': visualizations,
                'insights': insights,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Enhanced circuit simulation failed: {str(e)}")
            return {'error': str(e), 'success': False}

    # ================================================================
    # NEW EDUCATIONAL INSIGHTS SYSTEM
    # ================================================================

    def _generate_enhanced_educational_insights(
        self,
        gates: List[Dict],
        statevector: Optional[List[complex]],
        counts: Dict[str, int],
        num_qubits: int,
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Central orchestrator for all educational insights"""
        return {
            "gate_by_gate_insights": [self._analyze_gate_specific_insights(g, gates) for g in gates],
            "quantum_phenomena": self._detect_quantum_phenomena_detailed(gates, statevector, counts),
            "circuit_story": self._generate_circuit_story(gates),
            "difficulty_assessment": self._assess_circuit_difficulty(gates),
            "misconception_warnings": self._detect_and_correct_misconceptions(gates),
            "adaptive_challenges": self._generate_adaptive_challenges(gates),
            "visualization_guides": {
                "histogram": self._generate_visualization_insights("histogram", counts, gates),
                "circuit_diagram": self._generate_visualization_insights("circuit_diagram", None, gates),
                "statevector": self._generate_visualization_insights("statevector", statevector, gates)
            },
            "real_world_applications": self._suggest_real_world_applications(gates),
            "learning_path": self._suggest_learning_path(gates, user_context)
        }

    def _analyze_gate_specific_insights(self, gate: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Return rich pedagogy per gate"""
        g = gate.get("type", "").upper()
        info = {"gate": g}

        if g == "H":
            info.update({
                "physics_explanation": "Creates equal superposition of |0⟩ and |1⟩ states",
                "visual_analogy": "Like a spinning coin - both heads and tails simultaneously",
                "mathematical_detail": "H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2",
                "practical_tips": [
                    "Applying H twice returns to original state",
                    "Use before CNOT to create entanglement",
                    "Essential for quantum algorithms initialization"
                ],
                "common_applications": ["Quantum random number generation", "Algorithm initialization", "Creating Bell states"]
            })
        elif g == "X":
            info.update({
                "physics_explanation": "Bit-flip: maps |0⟩→|1⟩ and |1⟩→|0⟩",
                "visual_analogy": "Classical NOT gate in quantum form",
                "mathematical_detail": "X|0⟩ = |1⟩, X|1⟩ = |0⟩",
                "practical_tips": [
                    "Combine with H to get |−⟩ superposition",
                    "Used in quantum error correction",
                    "Building block for more complex operations"
                ]
            })
        elif g == "CNOT":
            info.update({
                "physics_explanation": "Entangles control and target qubits (conditional flip)",
                "visual_analogy": "Quantum 'if' statement - flips target IF control is |1⟩",
                "mathematical_detail": "CNOT|00⟩=|00⟩, CNOT|01⟩=|01⟩, CNOT|10⟩=|11⟩, CNOT|11⟩=|10⟩",
                "practical_tips": [
                    "Use H on control then CNOT to create Bell states",
                    "Essential for quantum teleportation",
                    "Creates 'spooky action at a distance'"
                ],
                "entanglement_creation": True
            })
        elif g in ["RX", "RY", "RZ"]:
            angle = gate.get("angle", "θ")
            info.update({
                "physics_explanation": f"Rotation around {g[1]}-axis by angle {angle}",
                "practical_tips": [
                    "Controls precise quantum state manipulation",
                    "Used for creating arbitrary superpositions",
                    "Essential for quantum optimization algorithms"
                ]
            })
        
        return info

    def _detect_quantum_phenomena_detailed(
        self, gates: List[Dict], statevector: Optional[List[complex]], counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Enhanced detection of quantum phenomena with explanations"""
        phenomena = {}
        gate_types = {g["type"].upper() for g in gates}
        
        # Superposition detection
        if "H" in gate_types or any(g.startswith("R") for g in gate_types):
            phenomena["superposition"] = {
                "detected": True,
                "explanation": "Your circuit creates quantum superposition - qubits exist in multiple states simultaneously",
                "evidence": "Hadamard or rotation gates create superposition",
                "visualization_tip": "Look for equal probabilities in measurement results",
                "real_world_analogy": "Like Schrödinger's cat being alive AND dead until observed"
            }
        
        # Entanglement detection
        if any(g in gate_types for g in ["CNOT", "CZ", "SWAP"]):
            phenomena["entanglement"] = {
                "detected": True,
                "explanation": "Your circuit creates quantum entanglement - qubits become mysteriously connected",
                "evidence": "Two-qubit gates create correlations between qubits",
                "einstein_quote": "Spooky action at a distance",
                "applications": ["Quantum teleportation", "Quantum cryptography", "Quantum computing speedup"]
            }
            
            # Measure entanglement strength if statevector available
            if statevector and len(statevector) == 4:  # Two-qubit system
                entropy = self._calculate_entanglement_entropy(statevector)
                phenomena["entanglement"]["strength"] = f"{entropy:.2f} bits"
        
        # Interference detection
        if len(set(counts.values())) > 1 and len(counts) > 1:
            phenomena["interference"] = {
                "detected": True,
                "explanation": "Quantum interference affects measurement probabilities",
                "evidence": "Unequal measurement outcomes suggest wave-like interference",
                "analogy": "Like waves in water - they can add up or cancel out"
            }
        
        return phenomena

    def _generate_circuit_story(self, gates: List[Dict]) -> Dict[str, str]:
        """Create educational narrative of circuit execution"""
        story_parts = []
        quantum_state = "|000...⟩"  # Initial state
        
        for i, gate in enumerate(gates, 1):
            gate_type = gate["type"].upper()
            if gate_type == "H":
                story_parts.append(f"Step {i}: Applied Hadamard gate, creating superposition")
                quantum_state = "superposition state"
            elif gate_type == "CNOT":
                story_parts.append(f"Step {i}: Applied CNOT gate, creating entanglement")
                quantum_state = "entangled state"
            elif gate_type.startswith("MEASURE"):
                story_parts.append(f"Step {i}: Measured qubit, collapsing quantum state")
                quantum_state = "classical state"
            else:
                story_parts.append(f"Step {i}: Applied {gate_type} gate")
        
        return {
            "step_by_step": " → ".join(story_parts),
            "final_state": quantum_state,
            "physics_summary": self._generate_physics_summary(gates)
        }

    def _generate_physics_summary(self, gates: List[Dict]) -> str:
        """Generate physics summary of what the circuit accomplishes"""
        gate_types = {g["type"].upper() for g in gates}
        
        if "H" in gate_types and "CNOT" in gate_types:
            return "This circuit demonstrates quantum superposition and entanglement - the two key quantum phenomena that give quantum computers their power."
        elif "H" in gate_types:
            return "This circuit explores quantum superposition - the ability of qubits to exist in multiple states simultaneously."
        elif "CNOT" in gate_types:
            return "This circuit creates quantum entanglement - mysterious correlations between qubits."
        else:
            return "This circuit performs quantum operations on qubits."

    def _assess_circuit_difficulty(self, gates: List[Dict]) -> Dict[str, Any]:
        """Assess circuit difficulty and provide guidance"""
        complexity_factors = {
            "gate_count": len(gates),
            "gate_variety": len(set(g["type"].upper() for g in gates)),
            "entangling_gates": sum(1 for g in gates if g["type"].upper() in ["CNOT", "CZ", "SWAP"]),
            "rotation_gates": sum(1 for g in gates if g["type"].upper().startswith("R")),
        }
        
        # Calculate difficulty score
        score = (
            complexity_factors["gate_count"] +
            2 * complexity_factors["entangling_gates"] +
            complexity_factors["rotation_gates"]
        )
        
        if score < 4:
            level = "beginner"
            guidance = "Great starting circuit! Try adding more gates to explore quantum phenomena."
        elif score < 10:
            level = "intermediate"
            guidance = "Nice complexity! You're exploring interesting quantum effects."
        else:
            level = "advanced"
            guidance = "Complex circuit! Consider breaking it into smaller parts for analysis."
        
        return {
            "level": level,
            "score": score,
            "factors": complexity_factors,
            "guidance": guidance,
            "next_steps": self._suggest_next_steps(level, gates)
        }

    def _suggest_next_steps(self, level: str, gates: List[Dict]) -> List[str]:
        """Suggest next learning steps based on current level"""
        gate_types = {g["type"].upper() for g in gates}
        
        if level == "beginner":
            suggestions = []
            if "H" not in gate_types:
                suggestions.append("Try adding a Hadamard gate to create superposition")
            if "CNOT" not in gate_types:
                suggestions.append("Add a CNOT gate to explore entanglement")
            return suggestions
            
        elif level == "intermediate":
            return [
                "Experiment with rotation gates (RX, RY, RZ) for precise control",
                "Try creating different Bell states",
                "Explore quantum algorithms like Deutsch-Jozsa"
            ]
        else:
            return [
                "Implement quantum algorithms like Grover's search",
                "Explore quantum error correction",
                "Try multi-qubit entangled states like GHZ"
            ]

    def _detect_and_correct_misconceptions(self, gates: List[Dict]) -> List[Dict[str, str]]:
        """Identify and address common quantum computing misconceptions"""
        misconceptions = []
        
        # Check for measurement misconceptions
        measure_positions = [i for i, g in enumerate(gates) if g["type"].upper().startswith("MEASURE")]
        if measure_positions and measure_positions[0] < len(gates) - 1:
            misconceptions.append({
                "type": "measurement_collapse",
                "issue": "Measurement found mid-circuit",
                "explanation": "Measurement collapses the quantum state irreversibly!",
                "correction": "Move measurements to the end to preserve quantum effects",
                "demonstration": "Try the same circuit with measurements at different positions"
            })
        
        # Check for classical thinking
        gate_types = {g["type"].upper() for g in gates}
        if not any(g in gate_types for g in ["H", "RX", "RY", "RZ"]):
            misconceptions.append({
                "type": "classical_mindset",
                "issue": "No superposition gates detected",
                "explanation": "Your circuit operates only on computational basis states",
                "correction": "Add Hadamard or rotation gates to explore quantum superposition",
                "quantum_advantage": "Quantum computers excel when using superposition and entanglement"
            })
        
        return misconceptions

    def _generate_adaptive_challenges(self, gates: List[Dict]) -> List[Dict[str, str]]:
        """Generate personalized challenges based on current circuit"""
        challenges = []
        gate_types = {g["type"].upper() for g in gates}
        
        # Entanglement challenge
        if "H" in gate_types and "CNOT" not in gate_types:
            challenges.append({
                "title": "Entanglement Explorer",
                "description": "You've mastered superposition! Now create quantum entanglement.",
                "task": "Add a CNOT gate after your Hadamard to create a Bell state",
                "success_criteria": "Measure correlated outcomes between qubits",
                "physics_goal": "Experience 'spooky action at a distance'"
            })
        
        # Bell state variants
        if "H" in gate_types and "CNOT" in gate_types:
            challenges.append({
                "title": "Bell State Master",
                "description": "Create all four Bell states",
                "task": "Use X and Z gates to create |Φ-⟩, |Ψ+⟩, and |Ψ-⟩ states",
                "learning_outcome": "Understand the Bell state basis"
            })
        
        # Algorithm challenge
        if len(gates) < 5:
            challenges.append({
                "title": "Quantum Algorithm Pioneer",
                "description": "Build your first quantum algorithm",
                "task": "Implement Deutsch's algorithm to distinguish constant vs balanced functions",
                "real_world_connection": "This demonstrates quantum computational advantage"
            })
        
        return challenges

    def _generate_visualization_insights(
        self, viz_type: str, data: Any, gates: List[Dict]
    ) -> Dict[str, str]:
        """Generate insights for specific visualizations"""
        insights = {}
        
        if viz_type == "histogram":
            insights = {
                "how_to_read": "Bar height shows measurement frequency",
                "equal_bars": "Equal bars suggest perfect superposition",
                "skewed_bars": "Unequal bars indicate interference or bias",
                "interpretation_tip": "Compare with classical expectations"
            }
            
        elif viz_type == "circuit_diagram":
            insights = {
                "how_to_read": "Time flows left to right, each horizontal line is a qubit",
                "gate_symbols": "H = Hadamard, ⊕ = CNOT target, • = CNOT control",
                "depth_meaning": "Circuit depth affects decoherence in real quantum computers"
            }
            
        elif viz_type == "statevector":
            insights = {
                "amplitude_bars": "Height = √probability, shows quantum amplitudes",
                "phase_information": "Phase differences create interference effects",
                "complex_plane": "Distance from origin = amplitude, angle = phase"
            }
        
        return insights

    def _suggest_real_world_applications(self, gates: List[Dict]) -> List[Dict[str, str]]:
        """Connect circuit concepts to real-world applications"""
        gate_types = {g["type"].upper() for g in gates}
        applications = []
        
        if "H" in gate_types:
            applications.append({
                "concept": "Superposition (Hadamard gates)",
                "application": "Quantum random number generation",
                "description": "True randomness for cryptography and simulations",
                "companies": "ID Quantique, QuintessenceLabs"
            })
        
        if "CNOT" in gate_types:
            applications.append({
                "concept": "Entanglement (CNOT gates)",
                "application": "Quantum key distribution",
                "description": "Unbreakable communication security",
                "real_deployment": "Chinese quantum satellite network"
            })
        
        return applications

    def _suggest_learning_path(self, gates: List[Dict], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest personalized learning progression"""
        current_level = user_context.get("level", "beginner")
        completed_concepts = set(g["type"].upper() for g in gates)
        
        learning_path = {
            "current_concepts": list(completed_concepts),
            "mastered_skills": [],
            "next_concepts": [],
            "long_term_goals": []
        }
        
        # Assess mastered skills
        if "H" in completed_concepts:
            learning_path["mastered_skills"].append("Quantum superposition")
        if "CNOT" in completed_concepts:
            learning_path["mastered_skills"].append("Quantum entanglement")
        
        # Suggest next concepts
        if "H" not in completed_concepts:
            learning_path["next_concepts"].append("Learn Hadamard gates for superposition")
        elif "CNOT" not in completed_concepts:
            learning_path["next_concepts"].append("Explore CNOT gates for entanglement")
        else:
            learning_path["next_concepts"].append("Try quantum algorithms like Deutsch-Jozsa")
        
        return learning_path

    def _calculate_entanglement_entropy(self, statevector: List[complex]) -> float:
        """Calculate entanglement entropy for two-qubit systems"""
        try:
            if len(statevector) != 4:
                return 0.0
            
            # Calculate reduced density matrix for first qubit
            psi = np.array(statevector).reshape((2, 2))
            rho = np.outer(psi.flatten(), np.conj(psi.flatten()))
            
            # Partial trace over second qubit
            rho_reduced = np.array([
                [rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]
            ])
            
            # Von Neumann entropy
            eigenvals = np.linalg.eigvals(rho_reduced)
            eigenvals = eigenvals[eigenvals > 1e-12]
            entropy = -sum(ev * np.log2(ev) for ev in eigenvals)
            
            return float(entropy)
        except:
            return 0.0

    # ================================================================
    # ORIGINAL METHODS (unchanged but complete implementations)
    # ================================================================
    
    def _serialize_complex_list(self, complex_list: List[complex]) -> List[Dict]:
        """Convert complex numbers to JSON-serializable format"""
        if not complex_list:
            return None
        
        return [
            {
                'real': float(c.real),
                'imag': float(c.imag),
                'magnitude': float(abs(c)),
                'phase': float(np.angle(c))
            }
            for c in complex_list
        ]
    
    def _generate_enhanced_visualizations(self, circuit: QuantumCircuit, 
                                        pure_circuit: Optional[QuantumCircuit],
                                        counts: Dict[str, int], 
                                        statevector: Optional[List],
                                        num_qubits: int,
                                        include_bloch: bool = True) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        visualizations = {}
        
        try:
            # Circuit diagram
            visualizations['circuit_diagram'] = self._generate_circuit_diagram(circuit)
            
            # Enhanced histogram
            if counts:
                visualizations['histogram'] = self._generate_enhanced_histogram(counts, num_qubits)
            
            # Statevector visualization
            if statevector and num_qubits <= 5:
                visualizations['statevector_plot'] = self._generate_statevector_plot(
                    statevector, num_qubits
                )
            
            # Bloch sphere visualization for single and two-qubit systems
            if include_bloch and statevector and self.bloch_visualizer:
                if num_qubits == 1:
                    # Single qubit Bloch sphere
                    bloch_image = self.bloch_visualizer.generate_bloch_sphere(
                        [statevector[0], statevector[1]],
                        title="Single Qubit State",
                        show_state_info=True
                    )
                    visualizations['bloch_sphere'] = bloch_image
                elif num_qubits == 2:
                    # Two-qubit Bloch spheres (individual qubits)
                    bloch_images = self._generate_two_qubit_bloch_spheres(statevector)
                    visualizations['bloch_spheres_2q'] = bloch_images
            
            # Probability distribution chart
            if counts:
                visualizations['probability_chart'] = self._generate_probability_chart(counts)
                
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_two_qubit_bloch_spheres(self, statevector: List[complex]) -> Dict[str, str]:
        """Generate Bloch spheres for individual qubits in a two-qubit system"""
        if not self.bloch_visualizer or len(statevector) != 4:
            return {}
        
        try:
            # Calculate reduced density matrices
            sv_array = np.array(statevector).reshape((2, 2))
            rho = np.outer(sv_array.flatten(), np.conj(sv_array.flatten()))
            
            # Reduced density matrix for qubit 0
            rho_0 = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                             [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
            
            # Reduced density matrix for qubit 1  
            rho_1 = np.array([[rho[0,0] + rho[2,2], rho[0,1] + rho[2,3]],
                             [rho[1,0] + rho[3,2], rho[1,1] + rho[3,3]]])
            
            # Extract state vectors from reduced density matrices (approximate)
            evals_0, evecs_0 = np.linalg.eigh(rho_0)
            evals_1, evecs_1 = np.linalg.eigh(rho_1)
            
            max_idx_0 = np.argmax(evals_0)
            max_idx_1 = np.argmax(evals_1)
            
            state_0 = evecs_0[:, max_idx_0]
            state_1 = evecs_1[:, max_idx_1]
            
            bloch_0 = self.bloch_visualizer.generate_bloch_sphere(
                state_0, title="Qubit 0 State", show_state_info=True
            )
            bloch_1 = self.bloch_visualizer.generate_bloch_sphere(
                state_1, title="Qubit 1 State", show_state_info=True
            )
            
            return {
                'qubit_0': bloch_0,
                'qubit_1': bloch_1
            }
            
        except Exception as e:
            logger.error(f"Two-qubit Bloch sphere generation failed: {str(e)}")
            return {}

    def _create_teleportation_template(self, **kwargs) -> List[Dict]:
        """Create quantum teleportation circuit template"""
        return [
            # Create Bell pair between qubits 1 and 2
            {'type': 'H', 'qubit': 1},
            {'type': 'CNOT', 'control': 1, 'target': 2},
            
            # Prepare state to teleport on qubit 0 (example: |+⟩ state)
            {'type': 'H', 'qubit': 0},
            {'type': 'T', 'qubit': 0},  # Add some phase
            
            # Bell measurement on qubits 0 and 1
            {'type': 'CNOT', 'control': 0, 'target': 1},
            {'type': 'H', 'qubit': 0}
            
            # Note: Classical control operations would follow in real implementation
        ]

    def _create_deutsch_jozsa_template(self, num_qubits: int = 3, oracle_type: str = 'constant') -> List[Dict]:
        """Create Deutsch-Jozsa algorithm circuit template"""
        gates = []
        
        # Initialize ancilla qubit (last qubit) to |1⟩
        gates.append({'type': 'X', 'qubit': num_qubits - 1})
        
        # Apply Hadamard to all qubits
        for i in range(num_qubits):
            gates.append({'type': 'H', 'qubit': i})
        
        # Oracle implementation (simplified)
        if oracle_type == 'balanced':
            # Example balanced oracle - flips ancilla for first half of inputs
            gates.append({'type': 'CNOT', 'control': 0, 'target': num_qubits - 1})
        # For constant oracle, do nothing (identity)
        
        # Apply Hadamard to input qubits
        for i in range(num_qubits - 1):
            gates.append({'type': 'H', 'qubit': i})
        
        return gates

    def _create_grover_template(self, num_qubits: int = 2, target_state: str = '11') -> List[Dict]:
        """Create Grover's search algorithm circuit template"""
        gates = []
        
        # Initialize superposition
        for i in range(num_qubits):
            gates.append({'type': 'H', 'qubit': i})
        
        # Grover iterations (simplified - one iteration)
        # Oracle marking target state
        if target_state == '11' and num_qubits >= 2:
            gates.append({'type': 'CZ', 'control': 0, 'target': 1})
        
        # Diffusion operator (inversion about average)
        for i in range(num_qubits):
            gates.append({'type': 'H', 'qubit': i})
            gates.append({'type': 'X', 'qubit': i})
        
        if num_qubits >= 2:
            gates.append({'type': 'CZ', 'control': 0, 'target': 1})
        
        for i in range(num_qubits):
            gates.append({'type': 'X', 'qubit': i})
            gates.append({'type': 'H', 'qubit': i})
        
        return gates

    def _create_qft_template(self, num_qubits: int = 3) -> List[Dict]:
        """Create Quantum Fourier Transform circuit template"""
        gates = []
        
        for i in range(num_qubits):
            # Hadamard gate
            gates.append({'type': 'H', 'qubit': i})
            
            # Controlled phase rotations
            for j in range(i + 1, num_qubits):
                angle = np.pi / (2 ** (j - i))
                # Simplified: use RZ gate instead of controlled phase
                gates.append({'type': 'RZ', 'qubit': j, 'angle': angle})
        
        # SWAP gates to reverse qubit order
        for i in range(num_qubits // 2):
            gates.append({'type': 'SWAP', 'qubit1': i, 'qubit2': num_qubits - 1 - i})
        
        return gates

    def _generate_enhanced_histogram(self, counts: Dict[str, int], num_qubits: int) -> Optional[str]:
        """Generate enhanced histogram with better styling and analysis"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            states = list(counts.keys())
            values = list(counts.values())
            total_shots = sum(values)
            
            # Sort by binary value for better visualization
            state_pairs = list(zip(states, values))
            state_pairs.sort(key=lambda x: int(x[0], 2))
            states, values = zip(*state_pairs)
            
            # Histogram
            colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
            bars = ax1.bar(range(len(states)), values, color=colors, edgecolor='white', linewidth=2)
            
            ax1.set_xlabel('Quantum States', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Measurement Counts', fontsize=14, fontweight='bold')
            ax1.set_title('Measurement Results Histogram', fontsize=16, fontweight='bold')
            ax1.set_xticks(range(len(states)))
            ax1.set_xticklabels([f'|{s}⟩' for s in states], rotation=45)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{int(value)}', ha='center', va='bottom', fontweight='bold')
            
            # Probability pie chart
            ax2.pie(values, labels=[f'|{s}⟩' for s in states], autopct='%1.1f%%', 
                colors=colors, startangle=90)
            ax2.set_title('Probability Distribution', fontsize=16, fontweight='bold')
            
            # Add statistics text
            fig.suptitle(f'Circuit Results ({num_qubits} Qubits, {total_shots} Shots)', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Enhanced histogram generation failed: {str(e)}")
            return None
        
    def _generate_statevector_plot(self, statevector: List[complex], num_qubits: int) -> Optional[str]:
        """Generate comprehensive statevector visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            amplitudes = [abs(amp) for amp in statevector]
            phases = [np.angle(amp) for amp in statevector]
            probabilities = [abs(amp)**2 for amp in statevector]
            
            num_states = len(statevector)
            state_labels = [f'|{format(i, f"0{num_qubits}b")}⟩' for i in range(num_states)]
            
            # Amplitude plot
            bars1 = ax1.bar(range(num_states), amplitudes, color='skyblue', 
                           edgecolor='navy', alpha=0.8)
            ax1.set_xlabel('Quantum States', fontweight='bold')
            ax1.set_ylabel('Amplitude Magnitude', fontweight='bold')
            ax1.set_title('Statevector Amplitudes', fontweight='bold', fontsize=14)
            ax1.set_xticks(range(num_states))
            ax1.set_xticklabels(state_labels, rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            # Phase plot
            bars2 = ax2.bar(range(num_states), phases, color='lightcoral', 
                           edgecolor='darkred', alpha=0.8)
            ax2.set_xlabel('Quantum States', fontweight='bold')
            ax2.set_ylabel('Phase (radians)', fontweight='bold')
            ax2.set_title('Statevector Phases', fontweight='bold', fontsize=14)
            ax2.set_xticks(range(num_states))
            ax2.set_xticklabels(state_labels, rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Probability plot
            bars3 = ax3.bar(range(num_states), probabilities, color='lightgreen', 
                           edgecolor='darkgreen', alpha=0.8)
            ax3.set_xlabel('Quantum States', fontweight='bold')
            ax3.set_ylabel('Probability', fontweight='bold')
            ax3.set_title('Measurement Probabilities', fontweight='bold', fontsize=14)
            ax3.set_xticks(range(num_states))
            ax3.set_xticklabels(state_labels, rotation=45)
            ax3.grid(axis='y', alpha=0.3)
            
            # Complex plane representation
            real_parts = [amp.real for amp in statevector]
            imag_parts = [amp.imag for amp in statevector]
            
            scatter = ax4.scatter(real_parts, imag_parts, c=range(num_states), 
                                 cmap='viridis', s=100, alpha=0.8)
            ax4.set_xlabel('Real Part', fontweight='bold')
            ax4.set_ylabel('Imaginary Part', fontweight='bold')
            ax4.set_title('Complex Plane Representation', fontweight='bold', fontsize=14)
            ax4.grid(alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add unit circle
            circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            ax4.add_patch(circle)
            
            # Add colorbar for complex plane
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('State Index', fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Statevector plot generation failed: {str(e)}")
            return None
    
    def _generate_circuit_diagram(self, circuit: QuantumCircuit) -> Optional[str]:
        """Generate professional circuit diagram"""
        try:
            # Create circuit diagram with enhanced styling
            fig, ax = plt.subplots(figsize=(max(12, circuit.size() * 0.8), 
                                          max(6, circuit.num_qubits * 1.5)))
            
            # Use circuit_drawer to create the diagram
            circuit.draw(output='mpl', ax=ax, style='clifford', 
                        scale=1.2, plot_barriers=True)
            
            plt.title(f'Quantum Circuit ({circuit.num_qubits} qubits, {circuit.size()} gates)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Circuit diagram generation failed: {str(e)}")
            return None
    
    def _calculate_advanced_metrics(self, circuit: QuantumCircuit, 
                                  statevector: Optional[List], 
                                  counts: Dict[str, int], 
                                  shots: int, num_qubits: int) -> Dict[str, Any]:
        """Calculate comprehensive circuit and quantum state metrics"""
        try:
            metrics = {
                # Circuit metrics
                'circuit_depth': circuit.depth(),
                'total_gates': circuit.size(),
                'num_qubits': circuit.num_qubits,
                'num_classical_bits': circuit.num_clbits,
                
                # Gate type analysis
                'gate_counts': {},
                'two_qubit_gates': 0,
                'single_qubit_gates': 0
            }
            
            # Analyze gate composition
            for instruction in circuit.data:
                gate_name = instruction.operation.name.upper()
                metrics['gate_counts'][gate_name] = metrics['gate_counts'].get(gate_name, 0) + 1
                
                if len(instruction.qubits) == 1:
                    metrics['single_qubit_gates'] += 1
                elif len(instruction.qubits) == 2:
                    metrics['two_qubit_gates'] += 1
            
            # Measurement statistics
            if counts:
                total_counts = sum(counts.values())
                probabilities = [count/total_counts for count in counts.values()]
                
                # Shannon entropy
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                metrics['measurement_entropy'] = float(entropy)
                metrics['num_outcomes'] = len(counts)
                metrics['most_probable_outcome'] = max(counts, key=counts.get)
                metrics['outcome_uniformity'] = float(1 / len(counts) if counts else 0)
            
            # Quantum state metrics
            if statevector:
                amplitudes = [abs(amp) for amp in statevector]
                probabilities = [amp**2 for amp in amplitudes]
                
                # State purity (always 1 for pure states)
                metrics['state_purity'] = 1.0
                
                # Participation ratio
                metrics['participation_ratio'] = float(1 / sum(p**2 for p in probabilities if p > 0))
                
                # Entanglement measures (for multi-qubit systems)
                if num_qubits > 1:
                    metrics['entanglement_measures'] = self._calculate_entanglement_measures(
                        statevector, num_qubits
                    )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_entanglement_measures(self, statevector: List[complex], 
                                       num_qubits: int) -> Dict[str, float]:
        """Calculate entanglement measures for multi-qubit systems"""
        try:
            # Convert to numpy array and reshape
            psi = np.array(statevector).reshape([2] * num_qubits)
            
            measures = {}
            
            # For two-qubit systems, calculate concurrence
            if num_qubits == 2:
                # Simplified concurrence calculation
                psi_flat = np.array(statevector).flatten()
                rho = np.outer(psi_flat, np.conj(psi_flat))
                
                # Partial trace over second qubit
                rho_reduced = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                                       [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
                
                # Von Neumann entropy of reduced state
                eigenvals = np.linalg.eigvals(rho_reduced)
                eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero values
                entropy = -sum(ev * np.log2(ev) for ev in eigenvals)
                measures['entanglement_entropy'] = float(entropy)
                
                # Simple entanglement measure based on Schmidt decomposition
                measures['schmidt_number'] = float(len(eigenvals))
            
            return measures
            
        except Exception as e:
            logger.error(f"Entanglement measures calculation failed: {str(e)}")
            return {}
    
    # Gate implementation methods (enhanced versions)
    def _add_gate_to_circuit(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        """Enhanced gate addition with comprehensive error checking"""
        gate_type = gate_spec.get('type', '').upper()
        
        if gate_type not in self.supported_gates:
            raise ValueError(f"Unsupported gate type: {gate_type}")
        
        gate_info = self.supported_gates[gate_type]
        implementation = gate_info['implementation']
        implementation(circuit, gate_spec)
    
    def _add_hadamard(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.h(qubit)
    
    def _add_pauli_x(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.x(qubit)
    
    def _add_pauli_y(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.y(qubit)
    
    def _add_pauli_z(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.z(qubit)
    
    def _add_identity(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.id(qubit)
    
    def _add_cnot(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        control = gate_spec.get('control', 0)
        target = gate_spec.get('target', 1)
        self._validate_qubit_index(circuit, control)
        self._validate_qubit_index(circuit, target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        circuit.cx(control, target)
    
    def _add_cz(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        control = gate_spec.get('control', 0)
        target = gate_spec.get('target', 1)
        self._validate_qubit_index(circuit, control)
        self._validate_qubit_index(circuit, target)
        circuit.cz(control, target)
    
    def _add_swap(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit1 = gate_spec.get('qubit1', 0)
        qubit2 = gate_spec.get('qubit2', 1)
        self._validate_qubit_index(circuit, qubit1)
        self._validate_qubit_index(circuit, qubit2)
        circuit.swap(qubit1, qubit2)
    
    def _add_rotation_x(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        angle = gate_spec.get('angle', np.pi/2)
        self._validate_qubit_index(circuit, qubit)
        circuit.rx(angle, qubit)
    
    def _add_rotation_y(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        angle = gate_spec.get('angle', np.pi/2)
        self._validate_qubit_index(circuit, qubit)
        circuit.ry(angle, qubit)
    
    def _add_rotation_z(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        angle = gate_spec.get('angle', np.pi/2)
        self._validate_qubit_index(circuit, qubit)
        circuit.rz(angle, qubit)

    def _add_s_gate(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.s(qubit)
    
    def _add_t_gate(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.t(qubit)
    
    # BB84 preparation methods
    def _prepare_computational_0(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.reset(qubit)
    
    def _prepare_computational_1(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.reset(qubit)
        circuit.x(qubit)
    
    def _prepare_superposition_plus(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.reset(qubit)
        circuit.h(qubit)
    
    def _prepare_superposition_minus(self, circuit: QuantumCircuit, gate_spec: Dict[str, Any]) -> None:
        qubit = gate_spec.get('qubit', 0)
        self._validate_qubit_index(circuit, qubit)
        circuit.reset(qubit)
        circuit.x(qubit)
        circuit.h(qubit)
    
    def _add_measurement_to_circuit(self, circuit: QuantumCircuit, meas_spec: Dict[str, Any]) -> None:
        """Add measurements to circuit"""
        meas_type = meas_spec.get('type', '').upper()
        qubit = meas_spec.get('qubit', 0)
        classical_bit = meas_spec.get('classical_bit', qubit)
        
        if meas_type == 'MEASURE_Z':
            circuit.measure(qubit, classical_bit)
        elif meas_type == 'MEASURE_X':
            circuit.h(qubit)
            circuit.measure(qubit, classical_bit)
        else:
            circuit.measure(qubit, classical_bit)
    
    def _validate_qubit_index(self, circuit: QuantumCircuit, qubit: int) -> None:
        """Enhanced qubit index validation"""
        if not isinstance(qubit, int):
            raise ValueError(f"Qubit index must be an integer, got {type(qubit)}")
        if qubit < 0:
            raise ValueError(f"Qubit index must be non-negative, got {qubit}")
        if qubit >= circuit.num_qubits:
            raise ValueError(f"Qubit index {qubit} exceeds circuit size {circuit.num_qubits}")
    
    def validate_circuit(self, num_qubits: int, gates: List[Dict]) -> Tuple[bool, str]:
        """Enhanced circuit validation"""
        try:
            if num_qubits < 1 or num_qubits > self.max_qubits:
                return False, f"Number of qubits must be between 1 and {self.max_qubits}"
            
            for i, gate in enumerate(gates):
                if 'type' not in gate:
                    return False, f"Gate {i} missing type specification"
                
                gate_type = gate['type'].upper()
                if gate_type not in self.supported_gates:
                    return False, f"Unsupported gate type: {gate_type}"
                
                # Validate gate-specific requirements
                gate_info = self.supported_gates[gate_type]
                if gate_info['qubits_required'] > num_qubits:
                    return False, f"Gate {gate_type} requires {gate_info['qubits_required']} qubits, circuit has {num_qubits}"
                
                # Validate parameters
                validation_result = self._validate_gate_parameters(gate, num_qubits, i)
                if not validation_result[0]:
                    return validation_result
            
            return True, "Circuit validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_gate_parameters(self, gate: Dict, num_qubits: int, gate_index: int) -> Tuple[bool, str]:
        """Enhanced gate parameter validation"""
        gate_type = gate['type'].upper()
        
        # Validate qubit indices
        qubit_params = ['qubit', 'control', 'target', 'qubit1', 'qubit2']
        for param in qubit_params:
            if param in gate:
                qubit_idx = gate[param]
                if not isinstance(qubit_idx, int) or qubit_idx < 0 or qubit_idx >= num_qubits:
                    return False, f"Invalid {param} index {qubit_idx} in gate {gate_index}"
        
        # Two-qubit gate specific validations
        if gate_type in ['CNOT', 'CZ']:
            control = gate.get('control', 0)
            target = gate.get('target', 1)
            if control == target:
                return False, f"Control and target must be different in {gate_type} gate {gate_index}"
        
        elif gate_type == 'SWAP':
            qubit1 = gate.get('qubit1', 0)
            qubit2 = gate.get('qubit2', 1)
            if qubit1 == qubit2:
                return False, f"SWAP qubits must be different in gate {gate_index}"
        
        # Rotation gate angle validation
        elif gate_type in ['RX', 'RY', 'RZ']:
            angle = gate.get('angle', np.pi/2)
            if not isinstance(angle, (int, float)):
                return False, f"Angle must be numeric in {gate_type} gate {gate_index}"
        
        return True, "Parameters valid"
    
    # Algorithm template methods
    def _create_bell_state_template(self, bell_type: str = '00') -> List[Dict]:
        """Create Bell state circuit template"""
        if bell_type == '00':  # |Φ+⟩
            return [
                {'type': 'H', 'qubit': 0},
                {'type': 'CNOT', 'control': 0, 'target': 1}
            ]
        elif bell_type == '01':  # |Φ-⟩
            return [
                {'type': 'H', 'qubit': 0},
                {'type': 'Z', 'qubit': 1},
                {'type': 'CNOT', 'control': 0, 'target': 1}
            ]
        elif bell_type == '10':  # |Ψ+⟩
            return [
                {'type': 'H', 'qubit': 0},
                {'type': 'X', 'qubit': 1},
                {'type': 'CNOT', 'control': 0, 'target': 1}
            ]
        elif bell_type == '11':  # |Ψ-⟩
            return [
                {'type': 'H', 'qubit': 0},
                {'type': 'X', 'qubit': 1},
                {'type': 'Z', 'qubit': 1},
                {'type': 'CNOT', 'control': 0, 'target': 1}
            ]
        else:
            raise ValueError(f"Unknown Bell state type: {bell_type}")
    
    def _create_ghz_state_template(self, num_qubits: int = 3) -> List[Dict]:
        """Create GHZ state circuit template"""
        gates = [{'type': 'H', 'qubit': 0}]
        for i in range(1, num_qubits):
            gates.append({'type': 'CNOT', 'control': 0, 'target': i})
        return gates
    
    def get_supported_gates_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about supported gates"""
        return {
            gate_type: {
                'name': gate_info.get('name', gate_type),
                'description': gate_info.get('description', ''),
                'parameters': gate_info.get('parameters', []),
                'qubits_required': gate_info.get('qubits_required', 1),
                'matrix': gate_info.get('matrix', 'N/A'),
                'bb84_relevant': gate_info.get('bb84_relevant', False)
            }
            for gate_type, gate_info in self.supported_gates.items()
        }
    
    def create_algorithm_circuit(self, algorithm_name: str, **kwargs) -> Dict[str, Any]:
        """Create circuit for famous quantum algorithms"""
        if algorithm_name not in self.algorithm_templates:
            return {'error': f'Unknown algorithm: {algorithm_name}', 'success': False}
        
        try:
            template_func = self.algorithm_templates[algorithm_name]
            gates = template_func(**kwargs)
            
            # Determine number of qubits needed
            max_qubit = 0
            for gate in gates:
                for param in ['qubit', 'control', 'target', 'qubit1', 'qubit2']:
                    if param in gate:
                        max_qubit = max(max_qubit, gate[param])
            
            num_qubits = max_qubit + 1
            
            # Simulate the circuit
            result = self.simulate_circuit(num_qubits, gates)
            result['algorithm'] = algorithm_name
            result['template_gates'] = gates
            
            return result
            
        except Exception as e:
            logger.error(f"Algorithm circuit creation failed: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _generate_probability_chart(self, counts: Dict[str, int]) -> Optional[str]:
        """Generate a clean probability distribution chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            states = list(counts.keys())
            probs = [count/sum(counts.values()) for count in counts.values()]
            
            # Sort by probability for better visualization
            state_prob_pairs = list(zip(states, probs))
            state_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            states, probs = zip(*state_prob_pairs)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(states)))
            bars = ax.bar(range(len(states)), probs, color=colors, alpha=0.8)
            
            ax.set_xlabel('Quantum States', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title('Measurement Probability Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels([f'|{s}⟩' for s in states])
            ax.set_ylim(0, max(probs) * 1.1)
            
            # Add probability labels
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(probs)*0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Probability chart generation failed: {str(e)}")
            return None


# Example usage and testing functions
def example_usage():
    """Demonstrate usage of the enhanced simulator with comprehensive educational insights"""
    print("="*80)
    print("ENHANCED QUANTUM CIRCUIT SIMULATOR - EDUCATIONAL EDITION")
    print("="*80)
    
    # Initialize simulator
    simulator = EnhancedQuantumCircuitSimulator()
    
    # Example: Bell state with comprehensive educational context
    print("\n🔬 Creating Bell state with educational insights...")
    gates = [
        {'type': 'H', 'qubit': 0},
        {'type': 'CNOT', 'control': 0, 'target': 1},
    ]
    
    result = simulator.simulate_circuit(
        2, gates, shots=1024, 
        user_context={'level': 'beginner', 'interests': ['entanglement', 'cryptography']}
    )
    
    if result['success']:
        print(f"\n📊 Measurement counts: {result['counts']}")
        print(f"📈 Probabilities: {result['probabilities']}")
        
        print("\n" + "="*60)
        print("🎓 COMPREHENSIVE EDUCATIONAL INSIGHTS")
        print("="*60)
        
        insights = result['insights']
        
        print("\n1️⃣ Gate-by-Gate Analysis:")
        for i, gate_insight in enumerate(insights['gate_by_gate_insights'], 1):
            print(f"   Gate {i} ({gate_insight['gate']}):")
            print(f"      • Physics: {gate_insight.get('physics_explanation', 'N/A')}")
            print(f"      • Analogy: {gate_insight.get('visual_analogy', 'N/A')}")
            if 'practical_tips' in gate_insight:
                print(f"      • Tips: {', '.join(gate_insight['practical_tips'][:2])}")
        
        print(f"\n2️⃣ Quantum Phenomena Detected:")
        for phenomenon, details in insights['quantum_phenomena'].items():
            if details.get('detected'):
                print(f"   ✅ {phenomenon.title()}:")
                print(f"      • {details.get('explanation', '')}")
                if 'applications' in details:
                    print(f"      • Applications: {', '.join(details['applications'][:2])}")
        
        print(f"\n3️⃣ Circuit Story:")
        story = insights['circuit_story']
        print(f"   📖 {story['step_by_step']}")
        print(f"   🔬 Physics Summary: {story['physics_summary']}")
        
        print(f"\n4️⃣ Difficulty Assessment:")
        difficulty = insights['difficulty_assessment']
        print(f"   📊 Level: {difficulty['level'].title()}")
        print(f"   💡 Guidance: {difficulty['guidance']}")
        
        if insights['misconception_warnings']:
            print(f"\n5️⃣ Learning Alerts:")
            for warning in insights['misconception_warnings']:
                print(f"   ⚠️  {warning['explanation']}")
                print(f"      💡 Solution: {warning['correction']}")
        
        print(f"\n6️⃣ Next Challenges:")
        for challenge in insights['adaptive_challenges']:
            print(f"   🎯 {challenge['title']}: {challenge['description']}")
            print(f"      📝 Task: {challenge['task']}")
        
        print(f"\n7️⃣ Real-World Applications:")
        for app in insights['real_world_applications']:
            print(f"   🌍 {app['concept']} → {app['application']}")
            print(f"      📖 {app['description']}")
        
        print(f"\n8️⃣ Learning Path:")
        path = insights['learning_path']
        print(f"   ✅ Mastered: {', '.join(path['mastered_skills'])}")
        print(f"   📚 Next: {', '.join(path['next_concepts'])}")
        
        print(f"\n9️⃣ Visualization Guides:")
        for viz_type, guide in insights['visualization_guides'].items():
            if guide:
                print(f"   📊 {viz_type.title()}: {guide.get('how_to_read', 'N/A')}")
    
    else:
        print(f"❌ Simulation failed: {result['error']}")
    
    # Example 2: Advanced circuit
    print("\n" + "="*60)
    print("🔬 ADVANCED EXAMPLE: GHZ State")
    print("="*60)
    
    advanced_gates = [
        {'type': 'H', 'qubit': 0},
        {'type': 'CNOT', 'control': 0, 'target': 1},
        {'type': 'CNOT', 'control': 0, 'target': 2},
    ]
    
    advanced_result = simulator.simulate_circuit(
        3, advanced_gates, shots=1024,
        user_context={'level': 'intermediate', 'interests': ['algorithms']}
    )
    
    if advanced_result['success']:
        print(f"📊 Measurement counts: {advanced_result['counts']}")
        insights = advanced_result['insights']
        print(f"🎓 Difficulty: {insights['difficulty_assessment']['level']}")
        print(f"🔬 Physics: {insights['circuit_story']['physics_summary']}")


if __name__ == "__main__":
    example_usage()
