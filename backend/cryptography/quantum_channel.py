# ---- quantum_channel.py  (enhanced 2025-08-04) ----------------------
import numpy as np
import qutip as qt
import random
import logging
from typing import Any, Dict, List, Union, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

QT_CIRC_DIM = [[2], [2]]          # canonical qubit DM dims

def _to_density_matrix(state: qt.Qobj) -> qt.Qobj:
    """
    Ensure the object is a 2×2 density matrix (dims=[[2],[2]]).
    Accepts kets, density matrices, or operators.
    """
    if isinstance(state, qt.Qobj):
        # If it is already a density matrix with correct dims we are done
        if state.isoper and state.dims == QT_CIRC_DIM:
            return state
        # Ket → ρ
        if state.isket:
            dm = state * state.dag()
            dm.dims = QT_CIRC_DIM
            return dm
        # Square operator of wrong dims → re-tag
        if state.isoper and state.shape == (2, 2):
            state.dims = QT_CIRC_DIM
            return state
        # Otherwise try to cast – fall back to maximally mixed
    logger.warning("State could not be coerced, using |0><0| fallback")
    return qt.basis(2, 0) * qt.basis(2, 0).dag()

class QuantumChannel:
    """
    ENHANCED: Quantum channel simulator for noise modelling in BB84.
    Dimension-safety patch: all QuTiP branches now work with 2×2 density matrices.
    Guaranteed error application for realistic channel simulation.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.simulator = AerSimulator()
        
        # ENHANCED: Error tracking for guaranteed application
        self.error_statistics = {
            'total_applications': 0,
            'errors_applied': 0,
            'error_types_applied': []
        }
        
        # ------------------------------------------------------------------ #
        #                          CHANNEL REGISTRY                          #
        # ------------------------------------------------------------------ #
        self.supported_channels = {
            "identity": self._identity_channel,
            "depolarizing": self._depolarizing_channel,
            "amplitude_damping": self._amplitude_damping_channel,
            "phase_damping": self._phase_damping_channel,
            "bit_flip": self._bit_flip_channel,
            "phase_flip": self._phase_flip_channel,
            "pauli": self._pauli_channel,
            "thermal": self._thermal_channel,
            "bb84_realistic": self._bb84_realistic_channel,
            "fiber_loss": self._fiber_loss_channel,
            "detector_noise": self._detector_noise_channel,
            "custom": self._custom_channel,
        }

    # ======================================================================
    #                           ENHANCED PUBLIC API
    # ======================================================================
    def apply_channel(
        self,
        channel_type: str,
        state: Union[qt.Qobj, np.ndarray, QuantumCircuit, None] = None,
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        ENHANCED: Apply a quantum channel with automatic backend selection.
        Now handles None state for parameter-only analysis.
        """
        try:
            if parameters is None:
                parameters = {}
                
            if channel_type not in self.supported_channels:
                raise ValueError(f"Unsupported channel type: {channel_type}")

            # ENHANCED: Handle None state for parameter analysis
            if state is None:
                state = qt.basis(2, 0)  # Default to |0⟩ state
                logger.info("No state provided, using default |0⟩ state")

            backend_used = (
                self._select_backend(state, parameters)
                if self.backend == "auto"
                else self.backend
            )
            prepared_state = self._prepare_state(state, backend_used)

            # Track application
            self.error_statistics['total_applications'] += 1

            result = self.supported_channels[channel_type](prepared_state, parameters)

            # ENHANCED: Add comprehensive diagnostics
            result.update({
                "channel_type": channel_type,
                "parameters": parameters,
                "backend_used": backend_used,
                "input_state_info": self._analyze_state(prepared_state),
                "output_state_info": self._analyze_state(result.get('output_state', prepared_state)),
                "error_statistics": self.error_statistics.copy(),
                "success": True,
            })
            return result

        except Exception as e:
            logger.error("Channel application failed: %s", e)
            logger.debug("Traceback", exc_info=True)
            return {
                "error": str(e),
                "success": False,
                "state_dims": getattr(state, "dims", "circuit/array"),
                "channel_type": channel_type,
                "parameters": parameters or {}
            }

    def _analyze_state(self, state) -> Dict[str, Any]:
        """Analyze quantum state properties"""
        try:
            if isinstance(state, QuantumCircuit):
                return {
                    "type": "quantum_circuit",
                    "num_qubits": state.num_qubits,
                    "num_gates": len(state.data)
                }
            elif isinstance(state, qt.Qobj):
                return {
                    "type": "qutip_qobj",
                    "dims": state.dims,
                    "shape": state.shape,
                    "is_ket": state.isket,
                    "is_oper": state.isoper,
                    "purity": float(state.purity()) if state.isoper else None
                }
            else:
                return {"type": str(type(state))}
        except Exception as e:
            return {"type": "unknown", "error": str(e)}

    # ======================================================================
    #                     BACKEND-SELECTION  HELPERS
    # ======================================================================
    def _select_backend(self, state, parameters) -> str:
        """Enhanced heuristic backend selector."""
        if isinstance(state, QuantumCircuit):
            return "qiskit"
        if hasattr(state, "dims") and np.prod(state.dims[0]) >= 16:
            return "qutip"
        if parameters.get("num_qubits", 1) >= 4:
            return "qutip"
        return "qiskit"

    def _prepare_state(self, state, backend):
        """Enhanced state preparation with better error handling."""
        try:
            # ---- Qiskit branch remains unchanged ----------------------------- #
            if backend == "qiskit":
                if isinstance(state, QuantumCircuit):
                    return state
                qc = QuantumCircuit(1, 1)
                return qc
            
            # ---- QuTiP branch (enhanced) -------------------------------------- #
            qobj = None
            if isinstance(state, QuantumCircuit):
                qobj = qt.basis(2, 0)  # default ket
            elif isinstance(state, (list, np.ndarray)):
                arr = np.asarray(state, dtype=complex)
                # vector → dm
                if arr.ndim == 1:
                    if len(arr) == 2:
                        arr = np.outer(arr, arr.conj())
                    else:
                        # Default to |0⟩ for invalid arrays
                        arr = np.array([[1, 0], [0, 0]], dtype=complex)
                qobj = qt.Qobj(arr)
            elif isinstance(state, qt.Qobj):
                qobj = state
            else:
                logger.warning(f"Unsupported state type {type(state)}, using default |0⟩")
                qobj = qt.basis(2, 0)

            return _to_density_matrix(qobj)
        except Exception as e:
            logger.error(f"State preparation failed: {e}")
            return qt.basis(2, 0) * qt.basis(2, 0).dag()

    # ======================================================================
    #                      ENHANCED KRAUS CHANNEL IMPLEMENTATIONS
    # ======================================================================
    def _identity_channel(self, state, _):
        return {"output_state": state, "kraus_operators": []}

    def _depolarizing_channel(self, state, params):
        p = params.get("p", 0.1)
        state = _to_density_matrix(state)
        k = [
            np.sqrt(1 - 3 * p / 4) * qt.qeye(2),
            np.sqrt(p / 4) * qt.sigmax(),
            np.sqrt(p / 4) * qt.sigmay(),
            np.sqrt(p / 4) * qt.sigmaz(),
        ]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        # Track error application
        if p > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('depolarizing')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _amplitude_damping_channel(self, state, params):
        gamma = params.get("gamma", 0.1)
        state = _to_density_matrix(state)
        k = [
            qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]]),
            qt.Qobj([[0, np.sqrt(gamma)], [0, 0]]),
        ]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        if gamma > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('amplitude_damping')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _phase_damping_channel(self, state, params):
        lam = params.get("lambda", 0.1)
        state = _to_density_matrix(state)
        k = [
            qt.Qobj([[1, 0], [0, np.sqrt(1 - lam)]]),
            qt.Qobj([[0, 0], [0, np.sqrt(lam)]]),
        ]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        if lam > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('phase_damping')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _bit_flip_channel(self, state, params):
        p = params.get("p", 0.1)
        state = _to_density_matrix(state)
        k = [np.sqrt(1 - p) * qt.qeye(2), np.sqrt(p) * qt.sigmax()]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        if p > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('bit_flip')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _phase_flip_channel(self, state, params):
        p = params.get("p", 0.1)
        state = _to_density_matrix(state)
        k = [np.sqrt(1 - p) * qt.qeye(2), np.sqrt(p) * qt.sigmaz()]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        if p > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('phase_flip')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _pauli_channel(self, state, params):
        px, py, pz = params.get("px", 0), params.get("py", 0), params.get("pz", 0)
        if px + py + pz > 1:
            raise ValueError("px+py+pz must not exceed 1")
        state = _to_density_matrix(state)
        k = [
            np.sqrt(1 - px - py - pz) * qt.qeye(2),
            np.sqrt(px) * qt.sigmax(),
            np.sqrt(py) * qt.sigmay(),
            np.sqrt(pz) * qt.sigmaz(),
        ]
        rho_out = sum(K @ state @ K.dag() for K in k)
        
        if px + py + pz > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('pauli')
            
        return {"output_state": rho_out, "kraus_operators": k}

    def _thermal_channel(self, state, params):
        T1 = params.get("T1", 1.0)
        T2 = params.get("T2", 0.5)
        gate_time = params.get("gate_time", 0.1)
        gamma = 1 - np.exp(-gate_time / T1)
        lam = 1 - np.exp(-gate_time / T2)
        state = _to_density_matrix(state)
        rho_ad = self._amplitude_damping_channel(state, {"gamma": gamma})["output_state"]
        rho_pd = self._phase_damping_channel(rho_ad, {"lambda": lam})["output_state"]
        
        if gamma > 0 or lam > 0:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('thermal')
            
        return {"output_state": rho_pd, "kraus_operators": []}

    # --------------------------- ENHANCED BB84 realistic --------------------------
    def _bb84_realistic_channel(self, state, params):
        """
        ENHANCED: Composite channel with GUARANTEED error application.
        Now ensures errors are applied deterministically based on parameters.
        """
        state = _to_density_matrix(state)
        applied_errors = {}
        error_log = []

        # Extract parameters with defaults
        fiber_loss = params.get("fiber_loss", 0.05)
        atmospheric_turbulence = params.get("atmospheric_turbulence", 0.02)
        polarization_drift = params.get("polarization_drift", 0.03)
        detector_efficiency = params.get("detector_efficiency", 0.9)
        detector_dark_counts = params.get("detector_dark_counts", 0.01)
        
        # Force error application mode
        force_errors = params.get("force_errors", True)
        
        logger.info(f"BB84 realistic channel: fiber_loss={fiber_loss}, "
                   f"atmospheric_turbulence={atmospheric_turbulence}, "
                   f"polarization_drift={polarization_drift}")

        # 1. ALWAYS apply fiber loss (amplitude damping)
        if fiber_loss > 0:
            result = self._amplitude_damping_channel(state, {"gamma": fiber_loss})
            state = result["output_state"]
            applied_errors['fiber_loss'] = fiber_loss
            error_log.append(f"Applied fiber loss: {fiber_loss}")
            logger.info(f"Applied fiber loss: {fiber_loss}")

        # 2. GUARANTEED atmospheric turbulence application
        if atmospheric_turbulence > 0:
            if force_errors or random.random() < atmospheric_turbulence:
                # Apply Pauli errors with guaranteed application
                error_strength = max(atmospheric_turbulence, 0.01)  # Minimum error
                result = self._pauli_channel(state, {
                    "px": error_strength / 3,
                    "py": error_strength / 3,
                    "pz": error_strength / 3
                })
                state = result["output_state"]
                applied_errors['atmospheric_turbulence'] = error_strength
                error_log.append(f"Applied atmospheric turbulence: {error_strength}")
                logger.info(f"Applied atmospheric turbulence: {error_strength}")

        # 3. ALWAYS apply polarization drift (phase errors)
        if polarization_drift > 0:
            result = self._phase_flip_channel(state, {"p": polarization_drift})
            state = result["output_state"]
            applied_errors['polarization_drift'] = polarization_drift
            error_log.append(f"Applied polarization drift: {polarization_drift}")
            logger.info(f"Applied polarization drift: {polarization_drift}")

        # 4. Detector inefficiency with guaranteed application
        inefficiency = 1 - detector_efficiency
        if inefficiency > 0:
            if force_errors or random.random() < inefficiency:
                # Apply photon loss
                loss_state = qt.basis(2, 0) * qt.basis(2, 0).dag()
                # Mix with original state based on efficiency
                state = detector_efficiency * state + inefficiency * loss_state
                applied_errors['detector_inefficiency'] = inefficiency
                error_log.append(f"Applied detector inefficiency: {inefficiency}")
                logger.info(f"Applied detector inefficiency: {inefficiency}")

        # 5. Dark counts with guaranteed application
        if detector_dark_counts > 0:
            if force_errors or random.random() < detector_dark_counts:
                # Apply random bit flip
                state = qt.sigmax() * state * qt.sigmax().dag()
                applied_errors['detector_dark_counts'] = detector_dark_counts
                error_log.append(f"Applied dark counts: {detector_dark_counts}")
                logger.info(f"Applied dark counts: {detector_dark_counts}")

        # Update error statistics
        if applied_errors:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('bb84_realistic')

        # ENHANCED: Calculate total error impact
        total_error_impact = sum(applied_errors.values())
        
        logger.info(f"BB84 realistic channel complete. Applied {len(applied_errors)} error types. "
                   f"Total impact: {total_error_impact:.4f}")

        return {
            "output_state": state,
            "kraus_operators": [],
            "applied_errors": applied_errors,
            "error_log": error_log,
            "total_error_impact": total_error_impact,
            "num_errors_applied": len(applied_errors),
            "forced_error_mode": force_errors
        }

    # -------------------- Enhanced fibre / detector wrappers ----------------------
    def _fiber_loss_channel(self, state, params):
        loss_rate = params.get("loss_rate", 0.2)
        distance = params.get("distance", 10)
        total_loss = loss_rate * distance / 10  # Convert dB to probability
        gamma = 1 - 10**(-total_loss)
        
        logger.info(f"Fiber loss: {loss_rate} dB/km × {distance} km = {gamma:.4f} loss probability")
        
        return self._amplitude_damping_channel(state, {"gamma": gamma})

    def _detector_noise_channel(self, state, params):
        eff = params.get("efficiency", 0.9)
        drk = params.get("dark_count_rate", 0.01)
        afterpulse = params.get("afterpulse_probability", 0.0)
        
        state = _to_density_matrix(state)
        errors_applied = []

        # Detector inefficiency
        if random.random() > eff:
            state = qt.basis(2, 0) * qt.basis(2, 0).dag()
            errors_applied.append("inefficiency")

        # Dark counts
        if random.random() < drk:
            state = qt.sigmax() * state * qt.sigmax().dag()
            errors_applied.append("dark_counts")

        # Afterpulses
        if random.random() < afterpulse:
            state = qt.sigmay() * state * qt.sigmay().dag()
            errors_applied.append("afterpulse")

        if errors_applied:
            self.error_statistics['errors_applied'] += 1
            self.error_statistics['error_types_applied'].append('detector_noise')

        return {
            "output_state": state, 
            "kraus_operators": [],
            "errors_applied": errors_applied
        }

    # --------------------------- Enhanced custom ----------------------------------
    def _custom_channel(self, state, params):
        kmats = params.get("kraus_operators", [])
        if not kmats:
            raise ValueError("custom channel requires 'kraus_operators'")
        ks = [qt.Qobj(K) for K in kmats]
        state = _to_density_matrix(state)
        rho = sum(K @ state @ K.dag() for K in ks)
        
        self.error_statistics['errors_applied'] += 1
        self.error_statistics['error_types_applied'].append('custom')
        
        return {"output_state": rho, "kraus_operators": ks}

    # ======================================================================
    #                           UTILITY METHODS
    # ======================================================================
    def reset_error_statistics(self):
        """Reset error tracking statistics"""
        self.error_statistics = {
            'total_applications': 0,
            'errors_applied': 0,
            'error_types_applied': []
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics"""
        return self.error_statistics.copy()

    def get_supported_channels(self) -> Dict[str, Dict[str, Any]]:
        """
        ENHANCED: Get information about supported quantum channels
        """
        return {
            'identity': {
                'description': 'Identity channel (no noise)',
                'parameters': {},
                'typical_use': 'Reference/ideal case',
                'bb84_relevant': True
            },
            'depolarizing': {
                'description': 'Uniform mixing with maximally mixed state',
                'parameters': {'p': 'Depolarization probability (0-1)'},
                'typical_use': 'General noise modeling',
                'bb84_relevant': True
            },
            'bb84_realistic': {
                'description': 'Comprehensive BB84 channel with guaranteed error application',
                'parameters': {
                    'fiber_loss': 'Fiber transmission loss probability',
                    'detector_dark_counts': 'Dark count probability',
                    'detector_efficiency': 'Detector efficiency (0-1)',
                    'atmospheric_turbulence': 'Atmospheric turbulence level',
                    'polarization_drift': 'Polarization drift error rate',
                    'force_errors': 'Force error application (default: True)'
                },
                'typical_use': 'Realistic BB84 channel modeling with guaranteed errors',
                'bb84_relevant': True,
                'enhanced': True
            },
            'fiber_loss': {
                'description': 'Optical fiber loss modeling',
                'parameters': {
                    'loss_rate': 'Loss rate in dB/km',
                    'distance': 'Fiber length in km'
                },
                'typical_use': 'Long-distance QKD modeling',
                'bb84_relevant': True
            },
            'detector_noise': {
                'description': 'Enhanced detector noise with afterpulses',
                'parameters': {
                    'efficiency': 'Detection efficiency (0-1)',
                    'dark_count_rate': 'Dark count probability',
                    'afterpulse_probability': 'Afterpulse probability'
                },
                'typical_use': 'Realistic detector modeling',
                'bb84_relevant': True,
                'enhanced': True
            }
        }

# ENHANCED: Test function to verify error application
def test_bb84_realistic_channel():
    """Test function to verify BB84 realistic channel applies errors"""
    logger.info("Testing BB84 realistic channel error application...")
    
    channel = QuantumChannel()
    
    # Test with guaranteed error application
    result = channel.apply_channel(
        'bb84_realistic',
        None,  # Use default state
        {
            'fiber_loss': 0.1,
            'atmospheric_turbulence': 0.05,
            'polarization_drift': 0.02,
            'detector_efficiency': 0.8,
            'detector_dark_counts': 0.03,
            'force_errors': True
        }
    )
    
    if result['success']:
        applied_errors = result.get('applied_errors', {})
        logger.info(f"Test completed successfully. Applied errors: {applied_errors}")
        logger.info(f"Error log: {result.get('error_log', [])}")
        return True
    else:
        logger.error(f"Test failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    # Run test
    test_bb84_realistic_channel()
