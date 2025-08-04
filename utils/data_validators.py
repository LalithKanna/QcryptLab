"""
Data Validation Utilities
=========================

This module provides comprehensive data validation and sanitization utilities
for the quantum cryptography tutorial application.

Key Features:
- Input parameter validation
- Quantum state validation
- Circuit parameter sanitization
- Security bounds checking
- Educational input constraints
"""

import re
import math
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation and sanitization utilities
    """
    
    def __init__(self):
        """Initialize the data validator"""
        # Quantum computing constraints
        self.max_qubits = 10
        self.max_circuit_depth = 100
        self.max_gates_per_circuit = 200
        
        # BB84 protocol constraints
        self.max_bits_bb84 = 10000
        self.min_bits_bb84 = 10
        self.max_error_rate = 0.5
        
        # Numerical constraints
        self.max_float_value = 1e10
        self.min_probability = 0.0
        self.max_probability = 1.0
        
        # String constraints
        self.max_string_length = 1000
        self.allowed_gate_types = {'H', 'X', 'Y', 'Z', 'I', 'CNOT', 'CZ', 'T', 'S'}
        
    def validate_circuit_input(self, params: Dict[str, Any]) -> bool:
        """
        Validate circuit input parameters (alias for validate_quantum_circuit_params)
        
        Args:
            params: Circuit parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        validation_result = self.validate_quantum_circuit_params(params)
        return validation_result['valid']

    def validate_quantum_circuit_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quantum circuit parameters
        
        Args:
            params: Circuit parameters to validate
            
        Returns:
            dict: Validation result with sanitized parameters
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_params': {}
            }
            
            # Validate number of qubits
            num_qubits = params.get('num_qubits', 1)
            if not isinstance(num_qubits, int):
                try:
                    num_qubits = int(num_qubits)
                except (ValueError, TypeError):
                    result['errors'].append('num_qubits must be an integer')
                    result['valid'] = False
                    return result
            
            if num_qubits < 1:
                result['errors'].append('num_qubits must be at least 1')
                result['valid'] = False
            elif num_qubits > self.max_qubits:
                result['errors'].append(f'num_qubits cannot exceed {self.max_qubits}')
                result['valid'] = False
            else:
                result['sanitized_params']['num_qubits'] = num_qubits
            
            # Validate gates
            gates = params.get('gates', [])
            if not isinstance(gates, list):
                result['errors'].append('gates must be a list')
                result['valid'] = False
                return result
            
            if len(gates) > self.max_gates_per_circuit:
                result['errors'].append(f'Too many gates (max: {self.max_gates_per_circuit})')
                result['valid'] = False
                return result
            
            sanitized_gates = []
            for i, gate in enumerate(gates):
                gate_validation = self._validate_gate(gate, num_qubits)
                if not gate_validation['valid']:
                    result['errors'].extend([f"Gate {i}: {err}" for err in gate_validation['errors']])
                    result['valid'] = False
                else:
                    sanitized_gates.append(gate_validation['sanitized_gate'])
            
            result['sanitized_params']['gates'] = sanitized_gates
            
            # Calculate circuit depth
            circuit_depth = len(gates)
            if circuit_depth > self.max_circuit_depth:
                result['warnings'].append(f'Circuit depth ({circuit_depth}) is quite large')
            
            result['sanitized_params']['circuit_depth'] = circuit_depth
            
            return result
            
        except Exception as e:
            logger.error(f"Circuit parameter validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_params': {}
            }
    
    def _validate_gate(self, gate: Dict[str, Any], num_qubits: int) -> Dict[str, Any]:
        """Validate individual gate parameters"""
        try:
            result = {
                'valid': True,
                'errors': [],
                'sanitized_gate': {}
            }
            
            # Validate gate type
            gate_type = gate.get('type', '').upper()
            if not gate_type:
                result['errors'].append('Gate type is required')
                result['valid'] = False
                return result
            
            if gate_type not in self.allowed_gate_types:
                result['errors'].append(f'Unknown gate type: {gate_type}')
                result['valid'] = False
                return result
            
            result['sanitized_gate']['type'] = gate_type
            
            # Validate qubit indices
            if gate_type in ['CNOT', 'CZ']:
                # Two-qubit gates
                control = gate.get('control')
                target = gate.get('target')
                
                if control is None or target is None:
                    result['errors'].append(f'{gate_type} gate requires control and target qubits')
                    result['valid'] = False
                    return result
                
                # Validate control qubit
                if not isinstance(control, int) or control < 0 or control >= num_qubits:
                    result['errors'].append(f'Invalid control qubit: {control}')
                    result['valid'] = False
                
                # Validate target qubit
                if not isinstance(target, int) or target < 0 or target >= num_qubits:
                    result['errors'].append(f'Invalid target qubit: {target}')
                    result['valid'] = False
                
                # Check for same qubit
                if control == target:
                    result['errors'].append('Control and target qubits cannot be the same')
                    result['valid'] = False
                
                if result['valid']:
                    result['sanitized_gate']['control'] = control
                    result['sanitized_gate']['target'] = target
            else:
                # Single-qubit gates
                qubit = gate.get('qubit', 0)
                if not isinstance(qubit, int) or qubit < 0 or qubit >= num_qubits:
                    result['errors'].append(f'Invalid qubit index: {qubit}')
                    result['valid'] = False
                else:
                    result['sanitized_gate']['qubit'] = qubit
            
            # Validate rotation angle (if applicable)
            if 'angle' in gate:
                angle = gate['angle']
                if not isinstance(angle, (int, float)):
                    try:
                        angle = float(angle)
                    except (ValueError, TypeError):
                        result['errors'].append('Gate angle must be numeric')
                        result['valid'] = False
                        return result
                
                # Normalize angle to [0, 2π]
                angle = angle % (2 * math.pi)
                result['sanitized_gate']['angle'] = angle
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'sanitized_gate': {}
            }
    
    def validate_quantum_state(self, state_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quantum state parameters
        
        Args:
            state_params: State parameters including amplitudes or angles
            
        Returns:
            dict: Validation result with normalized state
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_state': {}
            }
            
            if 'amplitudes' in state_params:
                # Direct amplitude specification
                amplitudes = state_params['amplitudes']
                
                if not isinstance(amplitudes, (list, tuple, np.ndarray)):
                    result['errors'].append('Amplitudes must be a list or array')
                    result['valid'] = False
                    return result
                
                # Convert to numpy array
                try:
                    amp_array = np.array(amplitudes, dtype=complex)
                except (ValueError, TypeError):
                    result['errors'].append('Amplitudes must be numeric')
                    result['valid'] = False
                    return result
                
                # Check dimensions (must be power of 2)
                n_amps = len(amp_array)
                if n_amps == 0 or (n_amps & (n_amps - 1)) != 0:
                    result['errors'].append('Number of amplitudes must be a power of 2')
                    result['valid'] = False
                    return result
                
                # Check for reasonable values
                if np.any(np.abs(amp_array) > 10):
                    result['warnings'].append('Some amplitudes are very large')
                
                # Normalize the state
                norm = np.sqrt(np.sum(np.abs(amp_array)**2))
                if norm < 1e-10:
                    result['errors'].append('State vector has zero norm')
                    result['valid'] = False
                    return result
                
                normalized_state = amp_array / norm
                result['sanitized_state']['amplitudes'] = normalized_state.tolist()
                result['sanitized_state']['norm'] = float(norm)
                
            elif 'theta' in state_params or 'phi' in state_params:
                # Spherical coordinate specification
                theta = state_params.get('theta', 0)
                phi = state_params.get('phi', 0)
                
                # Validate theta
                if not isinstance(theta, (int, float)):
                    try:
                        theta = float(theta)
                    except (ValueError, TypeError):
                        result['errors'].append('Theta must be numeric')
                        result['valid'] = False
                        return result
                
                # Validate phi
                if not isinstance(phi, (int, float)):
                    try:
                        phi = float(phi)
                    except (ValueError, TypeError):
                        result['errors'].append('Phi must be numeric')
                        result['valid'] = False
                        return result
                
                # Normalize angles
                theta = theta % (2 * math.pi)
                phi = phi % (2 * math.pi)
                
                # Convert to amplitudes
                alpha = math.cos(theta / 2)
                beta = math.sin(theta / 2) * complex(math.cos(phi), math.sin(phi))
                
                result['sanitized_state']['theta'] = theta
                result['sanitized_state']['phi'] = phi
                result['sanitized_state']['amplitudes'] = [alpha, beta]
                
            else:
                result['errors'].append('State must specify either amplitudes or theta/phi angles')
                result['valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Quantum state validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_state': {}
            }
    
    def validate_bb84_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate BB84 protocol parameters
        
        Args:
            params: BB84 protocol parameters
            
        Returns:
            dict: Validation result with sanitized parameters
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_params': {}
            }
            
            # Validate number of bits
            num_bits = params.get('num_bits', 100)
            if not isinstance(num_bits, int):
                try:
                    num_bits = int(num_bits)
                except (ValueError, TypeError):
                    result['errors'].append('num_bits must be an integer')
                    result['valid'] = False
                    return result
            
            if num_bits < self.min_bits_bb84:
                result['errors'].append(f'num_bits must be at least {self.min_bits_bb84}')
                result['valid'] = False
            elif num_bits > self.max_bits_bb84:
                result['errors'].append(f'num_bits cannot exceed {self.max_bits_bb84}')
                result['valid'] = False
            else:
                result['sanitized_params']['num_bits'] = num_bits
            
            # Validate error rate
            error_rate = params.get('error_rate', 0.0)
            if not isinstance(error_rate, (int, float)):
                try:
                    error_rate = float(error_rate)
                except (ValueError, TypeError):
                    result['errors'].append('error_rate must be numeric')
                    result['valid'] = False
                    return result
            
            if error_rate < 0 or error_rate > self.max_error_rate:
                result['errors'].append(f'error_rate must be between 0 and {self.max_error_rate}')
                result['valid'] = False
            else:
                result['sanitized_params']['error_rate'] = error_rate
                
                if error_rate > 0.11:
                    result['warnings'].append('Error rate exceeds BB84 security threshold (11%)')
            
            # Validate eavesdropper presence
            eavesdropper = params.get('eavesdropper_present', False)
            if not isinstance(eavesdropper, bool):
                result['warnings'].append('eavesdropper_present should be boolean, converting')
                eavesdropper = bool(eavesdropper)
            
            result['sanitized_params']['eavesdropper_present'] = eavesdropper
            
            # Validate measurement probability (if specified)
            if 'measurement_prob' in params:
                meas_prob = params['measurement_prob']
                if not isinstance(meas_prob, (int, float)):
                    try:
                        meas_prob = float(meas_prob)
                    except (ValueError, TypeError):
                        result['errors'].append('measurement_prob must be numeric')
                        result['valid'] = False
                        return result
                
                if meas_prob < 0 or meas_prob > 1:
                    result['errors'].append('measurement_prob must be between 0 and 1')
                    result['valid'] = False
                else:
                    result['sanitized_params']['measurement_prob'] = meas_prob
            
            return result
            
        except Exception as e:
            logger.error(f"BB84 parameter validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_params': {}
            }
    
    def validate_bloch_sphere_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Bloch sphere visualization parameters
        
        Args:
            params: Bloch sphere parameters
            
        Returns:
            dict: Validation result with sanitized parameters
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_params': {}
            }
            
            # Validate state vector
            if 'statevector' in params:
                state_validation = self.validate_quantum_state({'amplitudes': params['statevector']})
                if not state_validation['valid']:
                    result['errors'].extend(state_validation['errors'])
                    result['valid'] = False
                else:
                    result['sanitized_params']['statevector'] = state_validation['sanitized_state']['amplitudes']
            
            # Validate title
            title = params.get('title', 'Quantum State')
            if not isinstance(title, str):
                title = str(title)
            
            # Sanitize title (remove potentially harmful characters)
            title = re.sub(r'[<>"\']', '', title)
            if len(title) > self.max_string_length:
                title = title[:self.max_string_length]
                result['warnings'].append('Title was truncated')
            
            result['sanitized_params']['title'] = title
            
            # Validate visualization options
            if 'show_axes' in params:
                result['sanitized_params']['show_axes'] = bool(params['show_axes'])
            
            if 'show_equator' in params:
                result['sanitized_params']['show_equator'] = bool(params['show_equator'])
            
            return result
            
        except Exception as e:
            logger.error(f"Bloch sphere parameter validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_params': {}
            }
    
    def sanitize_user_input(self, user_input: str, input_type: str = 'general') -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            user_input: Raw user input string
            input_type: Type of input ('general', 'equation', 'filename')
            
        Returns:
            str: Sanitized input string
        """
        try:
            if not isinstance(user_input, str):
                user_input = str(user_input)
            
            # Basic sanitization
            sanitized = user_input.strip()
            
            if input_type == 'general':
                # Remove potentially dangerous characters
                sanitized = re.sub(r'[<>&"\']', '', sanitized)
                # Remove script tags
                sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
                
            elif input_type == 'equation':
                # Allow mathematical symbols but remove dangerous ones
                allowed_chars = r'[a-zA-Z0-9+\-*/()^{}\[\].,=\s|><πθφαβγ∑∏∫√±∞]'
                sanitized = ''.join(re.findall(allowed_chars, sanitized))
                
            elif input_type == 'filename':
                # Only allow safe filename characters
                allowed_chars = r'[a-zA-Z0-9._-]'
                sanitized = ''.join(re.findall(allowed_chars, sanitized))
                if not sanitized:
                    sanitized = 'default_name'
            
            # Truncate if too long
            if len(sanitized) > self.max_string_length:
                sanitized = sanitized[:self.max_string_length]
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Input sanitization failed: {str(e)}")
            return 'sanitization_failed'
    
    def validate_numerical_range(self, value: Union[int, float], min_val: float = None,
                                max_val: float = None, param_name: str = 'parameter') -> Dict[str, Any]:
        """
        Validate numerical parameter within specified range
        
        Args:
            value: Numerical value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_name: Parameter name for error messages
            
        Returns:
            dict: Validation result
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_value': value
            }
            
            # Convert to appropriate numeric type
            if not isinstance(value, (int, float)):
                try:
                    if '.' in str(value):
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    result['errors'].append(f'{param_name} must be numeric')
                    result['valid'] = False
                    return result
            
            # Check for reasonable magnitude
            if abs(value) > self.max_float_value:
                result['errors'].append(f'{param_name} magnitude too large')
                result['valid'] = False
                return result
            
            # Check range constraints
            if min_val is not None and value < min_val:
                result['errors'].append(f'{param_name} must be at least {min_val}')
                result['valid'] = False
            
            if max_val is not None and value > max_val:
                result['errors'].append(f'{param_name} cannot exceed {max_val}')
                result['valid'] = False
            
            # Check for special values
            if math.isnan(value):
                result['errors'].append(f'{param_name} cannot be NaN')
                result['valid'] = False
            elif math.isinf(value):
                result['errors'].append(f'{param_name} cannot be infinite')
                result['valid'] = False
            
            result['sanitized_value'] = value
            return result
            
        except Exception as e:
            logger.error(f"Numerical validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_value': None
            }
    
    def validate_probability_distribution(self, probabilities: List[float]) -> Dict[str, Any]:
        """
        Validate probability distribution
        
        Args:
            probabilities: List of probability values
            
        Returns:
            dict: Validation result with normalized probabilities
        """
        try:
            result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'sanitized_probabilities': []
            }
            
            if not isinstance(probabilities, (list, tuple, np.ndarray)):
                result['errors'].append('Probabilities must be a list or array')
                result['valid'] = False
                return result
            
            # Convert to numpy array
            try:
                prob_array = np.array(probabilities, dtype=float)
            except (ValueError, TypeError):
                result['errors'].append('All probabilities must be numeric')
                result['valid'] = False
                return result
            
            # Check individual probability constraints
            if np.any(prob_array < 0):
                result['errors'].append('Probabilities cannot be negative')
                result['valid'] = False
            
            if np.any(prob_array > 1):
                result['errors'].append('Individual probabilities cannot exceed 1')
                result['valid'] = False
            
            # Check normalization
            total_prob = np.sum(prob_array)
            if abs(total_prob - 1.0) > 1e-10:
                if abs(total_prob) < 1e-10:
                    result['errors'].append('Probability distribution has zero sum')
                    result['valid'] = False
                else:
                    # Auto-normalize
                    prob_array = prob_array / total_prob
                    result['warnings'].append('Probability distribution was auto-normalized')
            
            result['sanitized_probabilities'] = prob_array.tolist()
            return result
            
        except Exception as e:
            logger.error(f"Probability distribution validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'sanitized_probabilities': []
            }

def require_valid_params(param_validator_func):
    """
    Decorator to ensure parameters are validated before function execution
    
    Args:
        param_validator_func: Function that validates and returns sanitized parameters
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract parameters from kwargs or first argument
            if 'parameters' in kwargs:
                params = kwargs['parameters']
            elif len(args) > 0 and isinstance(args[0], dict):
                params = args[0]
            else:
                raise ValueError("No parameters found to validate")
            
            # Validate parameters
            validation_result = param_validator_func(params)
            
            if not validation_result['valid']:
                return {
                    'error': 'Parameter validation failed',
                    'validation_errors': validation_result['errors'],
                    'success': False
                }
            
            # Replace parameters with sanitized version
            if 'parameters' in kwargs:
                kwargs['parameters'] = validation_result['sanitized_params']
            else:
                args = (validation_result['sanitized_params'],) + args[1:]
            
            # Execute original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator