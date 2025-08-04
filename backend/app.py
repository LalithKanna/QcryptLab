"""
Enhanced Flask Backend Application for Quantum Cryptography Tutorial
===================================================================

Updated main application module with comprehensive error fixes and enhanced
integration support for the bulletproof frontend JavaScript implementation.

Key Updates:
- ADDED: /api/simulate-circuit endpoint for quantum circuit simulation
- ADDED: /api/security/analyze endpoint for advanced security analysis
- ADDED: /api/channel/simulate endpoint for quantum channel simulation
- ADDED: /api/docs endpoint for API documentation
- ADDED: Input validation decorators
- ADDED: Configuration management via environment variables
- Fixed JSON response structure for frontend compatibility  
- Enhanced error handling and validation
- Bulletproof API endpoint responses
- Comprehensive logging and debugging support
- Improved CORS configuration
- Enhanced component initialization with fallbacks
- FIXED: Unicode encoding errors (removed emojis)
- FIXED: JSON serialization errors (complex numbers)
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import logging
import traceback
import os
import datetime
import sys
import json
import numpy as np
from typing import Dict, Any, Optional, List, Union
from functools import wraps

# Import enhanced quantum simulation modules with error handling
try:
    from backend.quantum_simulator.enhanced_quantum_circuit_simulator import EnhancedQuantumCircuitSimulator
    CIRCUIT_SIMULATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EnhancedQuantumCircuitSimulator not available: {e}")
    CIRCUIT_SIMULATOR_AVAILABLE = False

try:
    from backend.quantum_simulator.state_visualizer import BlochSphereVisualizer
    BLOCH_VISUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BlochSphereVisualizer not available: {e}")
    BLOCH_VISUALIZER_AVAILABLE = False

try:
    from backend.quantum_simulator.advanced_simulator import AdvancedQuantumSimulator
    ADVANCED_SIMULATOR_AVAILABLE = True
except ImportError as e:
    ADVANCED_SIMULATOR_AVAILABLE = False

# Import enhanced cryptography modules with error handling
try:
    from backend.cryptography.bb84_protocol import BB84Protocol
    BB84_PROTOCOL_AVAILABLE = True
except ImportError as e:
    BB84_PROTOCOL_AVAILABLE = False

try:
    from backend.cryptography.quantum_channel import QuantumChannel
    QUANTUM_CHANNEL_AVAILABLE = True
except ImportError as e:
    QUANTUM_CHANNEL_AVAILABLE = False

try:
    from backend.cryptography.security_analysis import QKDSecurityAnalyzer
    SECURITY_ANALYZER_AVAILABLE = True
except ImportError as e:
    SECURITY_ANALYZER_AVAILABLE = False

# Import utility modules with error handling
try:
    from utils.image_generator import ImageGenerator
    from utils.data_validators import DataValidator
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False

# FIXED: Configure UTF-8 logging on Windows to prevent emoji encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7 doesn't have reconfigure

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Configuration management via environment variables
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    """Application configuration from environment variables"""
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
    MAX_QUBITS = int(os.getenv("MAX_QUBITS", 10))
    MAX_BB84_BITS = int(os.getenv("MAX_BB84_BITS", 1000))
    ENABLE_ADVANCED_ANALYSIS = os.getenv("ENABLE_ADVANCED_ANALYSIS", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-change-in-production")
    DEBUG_MODE = os.getenv("FLASK_DEBUG", "True").lower() == "true"

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_tutorial.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application with enhanced configuration
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# ENHANCED CORS configuration for bulletproof frontend integration
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # Allow all origins for development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False  # Simplified for debugging
    }
})

# Application configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON key order
app.config.from_object(Config)

# ENHANCED: Initialize components with comprehensive error handling
def safe_initialize_component(component_class, component_name, *args, **kwargs):
    """Safely initialize a component with error handling"""
    try:
        component = component_class(*args, **kwargs)
        logger.info(f"Successfully initialized {component_name}")
        return component
    except Exception as e:
        logger.error(f"Failed to initialize {component_name}: {str(e)}")
        return None

# Initialize enhanced quantum simulation components
circuit_simulator = None
if CIRCUIT_SIMULATOR_AVAILABLE:
    circuit_simulator = safe_initialize_component(
        EnhancedQuantumCircuitSimulator, "EnhancedQuantumCircuitSimulator", backend='qiskit'
    )

bloch_visualizer = None
if BLOCH_VISUALIZER_AVAILABLE:
    bloch_visualizer = safe_initialize_component(
        BlochSphereVisualizer, "BlochSphereVisualizer", style_theme='educational'
    )

advanced_simulator = None
if ADVANCED_SIMULATOR_AVAILABLE:
    advanced_simulator = safe_initialize_component(
        AdvancedQuantumSimulator, "AdvancedQuantumSimulator"
    )

# Initialize enhanced cryptography components
bb84_protocol = None
if BB84_PROTOCOL_AVAILABLE:
    bb84_protocol = safe_initialize_component(
        BB84Protocol, "BB84Protocol", backend='auto'
    )

quantum_channel = None
if QUANTUM_CHANNEL_AVAILABLE:
    quantum_channel = safe_initialize_component(
        QuantumChannel, "QuantumChannel"
    )

security_analyzer = None
if SECURITY_ANALYZER_AVAILABLE:
    security_analyzer = safe_initialize_component(
        QKDSecurityAnalyzer, "QKDSecurityAnalyzer", backend='auto'
    )

# Initialize utilities
image_generator = None
data_validator = None
if UTILS_AVAILABLE:
    image_generator = safe_initialize_component(ImageGenerator, "ImageGenerator")
    data_validator = safe_initialize_component(DataValidator, "DataValidator")

# FIXED: Deep serialization function to handle all complex numbers and NumPy objects
# ENHANCED: Deep serialization function to handle all complex numbers and NumPy objects
def deep_serialize_for_json(obj):
    """Recursively convert any object to JSON-serializable format"""
    if isinstance(obj, complex):
        return {
            'real': float(obj.real), 
            'imag': float(obj.imag), 
            'magnitude': float(abs(obj)), 
            'phase': float(np.angle(obj))
        }
    elif isinstance(obj, np.ndarray):
        return [deep_serialize_for_json(item) for item in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [deep_serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deep_serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):  # FIX: Handle NumPy booleans
        return bool(obj)
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        return str(obj)
    else:
        return str(obj)  # Convert unknown types to string


# ENHANCED: Error response helper
def create_error_response(error_message: str, details: str = None, status_code: int = 500) -> tuple:
    """Create a standardized error response"""
    response = {
        'success': False,
        'error': error_message,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    if details:
        response['details'] = details
    
    logger.error(f"API Error ({status_code}): {error_message}")
    if details:
        logger.error(f"Error details: {details}")
    
    return jsonify(response), status_code

# ENHANCED: Success response helper
def create_success_response(data: Dict[str, Any], message: str = None) -> Dict[str, Any]:
    """Create a standardized success response"""
    response = {
        'success': True,
        'timestamp': datetime.datetime.now().isoformat(),
        **data
    }
    
    if message:
        response['message'] = message
    
    return response

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Input validation decorators
# ─────────────────────────────────────────────────────────────────────────────
def validate_bb84_params(func):
    """Decorator for BB84 parameter validation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        
        # Validate num_bits
        num_bits = data.get('num_bits', 100)
        if not isinstance(num_bits, int) or not (10 <= num_bits <= Config.MAX_BB84_BITS):
            return create_error_response(
                "Invalid num_bits parameter",
                f"num_bits must be an integer between 10 and {Config.MAX_BB84_BITS}",
                400
            )
        
        # Validate noise_level
        noise_level = data.get('noise_level', 0.05)
        if not isinstance(noise_level, (int, float)) or not (0 <= noise_level <= 1):
            return create_error_response(
                "Invalid noise_level parameter",
                "noise_level must be a number between 0 and 1",
                400
            )
        
        # Validate backend choice
        backend = data.get('backend', 'auto')
        if backend not in ['auto', 'qiskit', 'qutip']:
            return create_error_response(
                "Invalid backend parameter",
                "backend must be 'auto', 'qiskit', or 'qutip'",
                400
            )
        
        return func(*args, **kwargs)
    return wrapper

def validate_circuit_params(func):
    """Decorator for circuit simulation parameter validation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        
        # Validate num_qubits
        num_qubits = data.get('num_qubits', 2)
        if not isinstance(num_qubits, int) or not (1 <= num_qubits <= Config.MAX_QUBITS):
            return create_error_response(
                "Invalid num_qubits parameter",
                f"num_qubits must be an integer between 1 and {Config.MAX_QUBITS}",
                400
            )
        
        # Validate gates
        gates = data.get('gates', [])
        if not isinstance(gates, list):
            return create_error_response(
                "Invalid gates parameter",
                "gates must be a list of gate specifications",
                400
            )
        
        # Validate shots
        shots = data.get('shots', 1024)
        if not isinstance(shots, int) or not (100 <= shots <= 10000):
            # Clamp to valid range instead of erroring
            data['shots'] = max(100, min(10000, shots))
            logger.warning(f"Shots value clamped to {data['shots']}")
        
        return func(*args, **kwargs)
    return wrapper

# Enhanced error handlers
@app.errorhandler(404)
def not_found(error):
    """Enhanced 404 error handler"""
    logger.warning(f"404 error: {request.url}")
    return create_error_response(
        "Resource not found",
        f"The requested endpoint {request.url} does not exist",
        404
    )

@app.errorhandler(500)
def internal_error(error):
    """Enhanced 500 error handler"""
    logger.error(f"Internal server error: {str(error)}")
    logger.error(traceback.format_exc())
    return create_error_response(
        "Internal server error",
        "An unexpected error occurred",
        500
    )

@app.errorhandler(413)
def payload_too_large(error):
    """Handle payload too large errors"""
    return create_error_response(
        "Payload too large",
        "Request payload exceeds maximum size limit",
        413
    )

# Frontend routes (enhanced with error handling)
@app.route('/')
def index():
    """Serve the main index page"""
    try:
        logger.info("Serving main index page")
        return send_from_directory('../frontend', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return "Error loading page", 500

@app.route('/quantum-basics/<path:filename>')
def quantum_basics(filename):
    """Serve quantum basics tutorial pages"""
    try:
        logger.info(f"Serving quantum basics page: {filename}")
        return send_from_directory('../frontend/quantum_basics', filename)
    except Exception as e:
        logger.error(f"Error serving quantum basics page {filename}: {str(e)}")
        return "Error loading page", 500

@app.route('/bb84-protocol/<path:filename>')
def bb84_protocol_pages(filename):
    """Serve BB84 protocol tutorial pages"""
    try:
        logger.info(f"Serving BB84 protocol page: {filename}")
        return send_from_directory('../frontend/bb84-protocol', filename)
    except Exception as e:
        logger.error(f"Error serving BB84 page {filename}: {str(e)}")
        return "Error loading page", 500

@app.route('/css/<path:filename>')
def css_files(filename):
    """Serve CSS files"""
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    """Serve JavaScript files"""
    return send_from_directory('../frontend/js', filename)

# CRITICAL: The missing /api/simulate-circuit endpoint that was causing the error
@app.route('/api/simulate-circuit', methods=['POST', 'OPTIONS'])
@validate_circuit_params
def simulate_circuit():
    """
    CRITICAL: Enhanced quantum circuit simulation endpoint - THIS WAS MISSING!
    This endpoint handles all circuit simulation requests from the frontend.
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        logger.info("Starting quantum circuit simulation request")
        
        # Check if circuit simulator is available
        if not circuit_simulator:
            logger.error("Circuit simulator component not available")
            return create_error_response(
                "Circuit simulator not available",
                "The quantum circuit simulator component is not initialized. Please restart the server.",
                503
            )
        
        # Extract and validate request data
        try:
            data = request.get_json()
            logger.info(f"Received circuit simulation request: {data}")
        except Exception as e:
            logger.error(f"Invalid JSON in circuit simulation request: {str(e)}")
            return create_error_response(
                "Invalid JSON data",
                "The request body must contain valid JSON",
                400
            )
        
        if not data:
            logger.error("No JSON data provided for circuit simulation")
            return create_error_response(
                "No JSON data provided",
                "Request body is empty or not valid JSON",
                400
            )
        
        # Extract parameters (validation already done by decorator)
        num_qubits = data.get('num_qubits', 2)
        gates = data.get('gates', [])
        shots = data.get('shots', 1024)
        include_statevector = data.get('include_statevector', True)
        include_bloch_sphere = data.get('include_bloch_sphere', True)
        
        logger.info(f"Simulating circuit: {num_qubits} qubits, {len(gates)} gates, {shots} shots")
        
        # Run the circuit simulation
        try:
            simulation_result = circuit_simulator.simulate_circuit(
                num_qubits=num_qubits,
                gates=gates,
                shots=shots,
                include_statevector=include_statevector,
                include_bloch_sphere=include_bloch_sphere
            )
            
            if not simulation_result.get('success', False):
                raise Exception(simulation_result.get('error', 'Unknown simulation error'))
            
            logger.info("Circuit simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Circuit simulation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(
                "Circuit simulation failed",
                f"Simulation error: {str(e)}",
                500
            )
        
        # Serialize the result for JSON response
        try:
            serialized_result = deep_serialize_for_json(simulation_result)
            
            # Test serialization
            json.dumps(serialized_result)
            logger.info("Circuit simulation result serialization test passed")
            
        except Exception as serialize_error:
            logger.error(f"Circuit simulation result serialization failed: {str(serialize_error)}")
            return create_error_response(
                "Result serialization failed",
                f"Could not serialize simulation result: {str(serialize_error)}",
                500
            )
        
        # Return successful response
        logger.info("Successfully completed circuit simulation request")
        return jsonify(serialized_result)
        
    except Exception as e:
        logger.error(f"Critical error in circuit simulation endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(
            "Circuit simulation endpoint failed",
            f"Critical error: {str(e)}",
            500
        )

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Missing quantum channel simulation endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/channel/simulate', methods=['POST', 'OPTIONS'])
def simulate_channel():
    """Quantum channel simulation endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
        
    if not quantum_channel:
        return create_error_response(
            "Quantum channel not available",
            "The quantum channel component is not initialized",
            503
        )
    
    try:
        data = request.get_json(silent=True) or {}
        channel_type = data.get('channel_type', 'bb84_realistic')
        parameters = data.get('parameters', {})
        
        logger.info(f"Simulating quantum channel: {channel_type}")
        
        result = quantum_channel.apply_channel(
            channel_type=channel_type,
            state=None,
            parameters=parameters
        )
        
        if not result.get('success', False):
            raise Exception(result.get('error', 'Channel simulation failed'))
        
        return jsonify(create_success_response({
            'channel_result': deep_serialize_for_json(result)
        }))
        
    except Exception as e:
        logger.error(f"Channel simulation failed: {str(e)}")
        return create_error_response("Channel simulation failed", str(e), 500)

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Missing advanced security analysis endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/security/analyze', methods=['POST', 'OPTIONS'])
def analyze_security():
    """Advanced security analysis endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if not security_analyzer:
        return create_error_response(
            "Security analyzer not available",
            "The security analyzer component is not initialized",
            503
        )
    
    try:
        data = request.get_json(silent=True) or {}
        protocol_result = data.get('protocol_result', {})
        advanced_analysis = data.get('advanced_analysis', True)
        
        logger.info("Performing advanced security analysis...")
        
        analysis = security_analyzer.analyze_protocol_security(
            protocol_result, 
            advanced_analysis=advanced_analysis
        )
        
        if not analysis.get('success', False):
            raise Exception(analysis.get('error', 'Security analysis failed'))
        
        return jsonify(create_success_response({
            'security_analysis': deep_serialize_for_json(analysis)
        }))
        
    except Exception as e:
        logger.error(f"Security analysis failed: {str(e)}")
        return create_error_response("Security analysis failed", str(e), 500)

# ENHANCED: Health check endpoint with detailed component status
@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with detailed component status"""
    try:
        component_status = {
            'circuit_simulator': circuit_simulator is not None,
            'bloch_visualizer': bloch_visualizer is not None,
            'advanced_simulator': advanced_simulator is not None,
            'bb84_protocol': bb84_protocol is not None,
            'quantum_channel': quantum_channel is not None,
            'security_analyzer': security_analyzer is not None,
            'image_generator': image_generator is not None,
            'data_validator': data_validator is not None
        }
        
        all_healthy = all(component_status.values())
        
        health_data = {
            'status': 'healthy' if all_healthy else 'degraded',
            'message': 'Enhanced Quantum Cryptography Tutorial API',
            'version': '2.2.0',
            'components': component_status,
            'features': {
                'bb84_protocol': bb84_protocol is not None,
                'advanced_simulations': advanced_simulator is not None,
                'enhanced_visualizations': bloch_visualizer is not None,
                'security_analysis': security_analyzer is not None,
                'circuit_simulation': circuit_simulator is not None,
                'quantum_channel_simulation': quantum_channel is not None
            },
            'configuration': {
                'max_qubits': Config.MAX_QUBITS,
                'max_bb84_bits': Config.MAX_BB84_BITS,
                'advanced_analysis_enabled': Config.ENABLE_ADVANCED_ANALYSIS
            },
            'debug_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'flask_debug': Config.DEBUG_MODE,
                'component_counts': sum(component_status.values())
            }
        }
        
        return jsonify(create_success_response(health_data))
        
    except Exception as e:
        return create_error_response("Health check failed", str(e), 500)

# BULLETPROOF: Enhanced Bloch Sphere Visualization Endpoint
@app.route('/api/visualize-bloch', methods=['POST', 'OPTIONS'])
def visualize_bloch_sphere():
    """
    BULLETPROOF Bloch sphere visualization endpoint with comprehensive error handling
    and frontend compatibility - FIXED Unicode and JSON serialization issues
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        logger.info("Starting Bloch sphere visualization request")
        
        # ENHANCED: Validate component availability
        if not bloch_visualizer:
            logger.error("Bloch visualizer component not available")
            return create_error_response(
                "Bloch visualizer not available",
                "The Bloch sphere visualization component is not initialized",
                503
            )
        
        # ENHANCED: Extract and validate request data
        try:
            data = request.get_json()
            logger.info(f"Received request data: {data}")
        except Exception as e:
            logger.error(f"Invalid JSON in request: {str(e)}")
            return create_error_response(
                "Invalid JSON data",
                "The request body must contain valid JSON",
                400
            )
        
        if not data:
            logger.error("No JSON data provided")
            return create_error_response(
                "No JSON data provided",
                "Request body is empty or not valid JSON",
                400
            )
        
        # ENHANCED: Extract parameters with comprehensive fallbacks
        title = data.get('title', 'Quantum State')
        show_state_info = data.get('show_state_info', True)
        bb84_context = data.get('bb84_context', None)
        style_theme = data.get('style_theme', 'educational')
        
        logger.info(f"Extracted parameters: title='{title}', show_state_info={show_state_info}")
        
        # ENHANCED: Handle different input formats with validation
        statevector = None
        
        if 'statevector' in data:
            statevector = data['statevector']
            logger.info(f"Using provided statevector: {statevector}")
            
        elif 'theta' in data and 'phi' in data:
            try:
                theta = float(data['theta'])
                phi = float(data['phi'])
                
                # ENHANCED: Validate angles
                if not (0 <= theta <= np.pi + 0.1):  # Small tolerance for floating point
                    return create_error_response(
                        "Invalid theta value",
                        f"theta must be between 0 and π, got {theta}",
                        400
                    )
                    
                if not (0 <= phi <= 2 * np.pi + 0.1):  # Small tolerance
                    return create_error_response(
                        "Invalid phi value", 
                        f"phi must be between 0 and 2π, got {phi}",
                        400
                    )
                
                # Convert to statevector
                statevector = bloch_visualizer.spherical_to_statevector(theta, phi)
                logger.info(f"Converted spherical coordinates (theta={theta:.3f}, phi={phi:.3f}) to statevector")
                
            except ValueError as e:
                return create_error_response(
                    "Invalid angle values",
                    f"theta and phi must be valid numbers: {str(e)}",
                    400
                )
        else:
            # ENHANCED: Default to |0⟩ state with warning
            statevector = [1, 0]
            logger.warning("No statevector or angles provided, using default |0> state")
        
        # ENHANCED: Apply style theme with error handling
        if style_theme:
            try:
                bloch_visualizer.style_theme = style_theme
                bloch_visualizer._initialize_theme_styles()
                logger.info(f"Applied style theme: {style_theme}")
            except Exception as e:
                logger.warning(f"Failed to apply style theme {style_theme}: {str(e)}")
        
        # ENHANCED: Generate Bloch sphere with comprehensive error handling
        try:
            logger.info("Generating Bloch sphere visualization...")
            bloch_image = bloch_visualizer.generate_bloch_sphere(
                statevector=statevector,
                title=title,
                show_state_info=show_state_info,
                bb84_context=bb84_context
            )
            
            if not bloch_image:
                raise Exception("Bloch sphere generation returned empty result")
                
            logger.info("Successfully generated Bloch sphere image")
            
        except Exception as e:
            logger.error(f"Bloch sphere generation failed: {str(e)}")
            return create_error_response(
                "Bloch sphere generation failed",
                f"Error during visualization: {str(e)}",
                500
            )
        
        # ENHANCED: Generate state analysis with comprehensive error handling
        try:
            logger.info("Analyzing quantum state properties...")
            state_analysis = bloch_visualizer.analyze_state_properties(statevector)
            
            if not state_analysis or 'error' in state_analysis:
                raise Exception(f"State analysis failed: {state_analysis.get('error', 'Unknown error')}")
                
            logger.info("Successfully analyzed state properties")
            
        except Exception as e:
            logger.error(f"State analysis failed: {str(e)}")
            # ENHANCED: Provide fallback analysis
            state_analysis = {
                'error': str(e),
                'fallback': True,
                'message': 'Basic analysis unavailable, using fallback'
            }
        
        # BULLETPROOF: Create comprehensive response structure with FIXED serialization
        try:
            # FIXED: Deep serialize everything to handle complex numbers
            statevector_serializable = deep_serialize_for_json(statevector)
            state_analysis_serializable = deep_serialize_for_json(state_analysis)
            
            response_data = {
                'bloch_image': str(bloch_image) if bloch_image else '',
                'state_analysis': state_analysis_serializable,
                'statevector': statevector_serializable,
                'state_vector': statevector_serializable,  # CRITICAL: Backward compatibility
                'request_parameters': {
                    'title': title,
                    'show_state_info': show_state_info,
                    'bb84_context': bb84_context,
                    'style_theme': style_theme
                },
                'generation_info': {
                    'backend_used': 'BlochSphereVisualizer',
                    'image_format': 'base64_png',
                    'analysis_included': 'error' not in state_analysis_serializable
                }
            }
            
            # CRITICAL: Test JSON serialization before sending
            try:
                json.dumps(response_data)  # This will raise an error if serialization fails
                logger.info("Response data serialization test passed")
            except Exception as serialize_error:
                logger.error(f"Response serialization test failed: {str(serialize_error)}")
                raise serialize_error
            
            logger.info("Successfully created response data structure")
            
            # BULLETPROOF: Final response with success flag
            final_response = create_success_response(
                response_data,
                "Bloch sphere visualization generated successfully"
            )
            
            logger.info("Successfully completed Bloch sphere visualization request")
            return jsonify(final_response)
            
        except Exception as response_error:
            logger.error(f"Response creation failed: {str(response_error)}")
            
            # ULTIMATE FALLBACK: Minimal safe response
            return jsonify({
                'success': False,
                'error': 'Response creation failed',
                'details': str(response_error),
                'fallback_response': {
                    'bloch_image': str(bloch_image) if 'bloch_image' in locals() else '',
                    'statevector': [1.0, 0.0],
                    'state_vector': [1.0, 0.0]
                },
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
        
    except Exception as e:
        logger.error(f"Critical error in Bloch visualization endpoint: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return create_error_response(
            "Bloch sphere visualization failed",
            f"Critical error: {str(e)}",
            500
        )

# ENHANCED: BB84 Protocol Simulation Endpoint
@app.route('/api/bb84/simulate', methods=['POST', 'OPTIONS'])
@validate_bb84_params
def simulate_bb84_protocol():
    """Enhanced BB84 quantum key distribution protocol simulation"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        logger.info("Starting BB84 protocol simulation")
        
        if not bb84_protocol:
            return create_error_response(
                "BB84 protocol not available",
                "The BB84 protocol component is not initialized",
                503
            )
        
        data = request.get_json() or {}
        logger.info(f"BB84 simulation request: {data}")
        
        # ENHANCED: Parameter extraction (validation already done by decorator)
        num_bits = data.get('num_bits', 100)
        eavesdropper_present = data.get('eavesdropper_present', data.get('eavesdropper', False))
        noise_level = data.get('noise_level', 0.05)
        error_correction = data.get('error_correction', True)
        backend = data.get('backend', 'auto')
        advanced_analysis = data.get('advanced_analysis', Config.ENABLE_ADVANCED_ANALYSIS)
        
        # ENHANCED: Protocol execution with detailed logging
        logger.info(f"Running BB84 protocol: {num_bits} bits, eavesdropper={eavesdropper_present}")
        
        protocol_result = bb84_protocol.run_full_protocol(
            num_bits=num_bits,
            eavesdropper_present=eavesdropper_present,
            noise_level=noise_level,
            error_correction=error_correction,
            backend=backend,
            advanced_analysis=advanced_analysis
        )
        
        if not protocol_result.get('success', False):
            return create_error_response(
                "BB84 protocol execution failed",
                protocol_result.get('error', 'Unknown protocol error'),
                500
            )
        
        # ENHANCED: Automatic security analysis if available
        if security_analyzer and Config.ENABLE_ADVANCED_ANALYSIS:
            try:
                logger.info("Running automatic security analysis...")
                security_result = security_analyzer.analyze_protocol_security(
                    protocol_result, advanced_analysis=advanced_analysis
                )
                if security_result.get('success'):
                    protocol_result['automatic_security_analysis'] = security_result
                else:
                    logger.warning(f"Automatic security analysis failed: {security_result.get('error')}")
            except Exception as e:
                logger.warning(f"Security analysis failed: {str(e)}")
                protocol_result['security_analysis_error'] = str(e)
        
        # FIXED: Deep serialize the protocol result
        protocol_result_serializable = deep_serialize_for_json(protocol_result)
        
        logger.info("BB84 protocol simulation completed successfully")
        return jsonify(create_success_response(protocol_result_serializable))
        
    except Exception as e:
        logger.error(f"BB84 simulation error: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(
            "BB84 protocol simulation failed",
            str(e),
            500
        )

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: API documentation endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """Auto-generated API documentation"""
    docs = {
        'title': 'Quantum Cryptography Tutorial API',
        'version': '2.2.0',
        'description': 'Enhanced Flask backend for quantum cryptography education',
        'base_url': Config.BACKEND_URL,
        'endpoints': {
            '/api/simulate-circuit': {
                'method': 'POST',
                'description': 'Simulate quantum circuits with Qiskit backend',
                'parameters': {
                    'num_qubits': 'int (1-' + str(Config.MAX_QUBITS) + ') - Number of qubits',
                    'gates': 'list - Gate specifications',
                    'shots': 'int (100-10000) - Number of measurement shots',
                    'include_statevector': 'bool - Include quantum state vector',
                    'include_bloch_sphere': 'bool - Generate Bloch sphere visualization'
                },
                'example': {
                    'num_qubits': 2,
                    'gates': [{'gate': 'h', 'qubit': 0}, {'gate': 'cx', 'control': 0, 'target': 1}],
                    'shots': 1024
                }
            },
            '/api/visualize-bloch': {
                'method': 'POST',
                'description': 'Generate Bloch sphere visualizations',
                'parameters': {
                    'statevector': 'list - Quantum state vector [a, b]',
                    'theta': 'float - Polar angle (alternative to statevector)',
                    'phi': 'float - Azimuthal angle (alternative to statevector)',
                    'title': 'str - Visualization title',
                    'show_state_info': 'bool - Show state information'
                }
            },
            '/api/bb84/simulate': {
                'method': 'POST',
                'description': 'Run BB84 quantum key distribution protocol',
                'parameters': {
                    'num_bits': 'int (10-' + str(Config.MAX_BB84_BITS) + ') - Number of bits to exchange',
                    'eavesdropper_present': 'bool - Include eavesdropper (Eve)',
                    'noise_level': 'float (0-1) - Channel noise level',
                    'error_correction': 'bool - Apply error correction',
                    'backend': 'str (auto/qiskit/qutip) - Quantum backend',
                    'advanced_analysis': 'bool - Enable advanced security analysis'
                }
            },
            '/api/channel/simulate': {
                'method': 'POST',
                'description': 'Simulate quantum channel effects',
                'parameters': {
                    'channel_type': 'str - Type of quantum channel',
                    'parameters': 'dict - Channel-specific parameters'
                }
            },
            '/api/security/analyze': {
                'method': 'POST',
                'description': 'Advanced security analysis of QKD protocols',
                'parameters': {
                    'protocol_result': 'dict - Results from protocol execution',
                    'advanced_analysis': 'bool - Enable finite-key analysis'
                }
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Health check and component status'
            },
            '/api/system-status': {
                'method': 'GET',
                'description': 'Detailed system status and capabilities'
            }
        },
        'configuration': {
            'max_qubits': Config.MAX_QUBITS,
            'max_bb84_bits': Config.MAX_BB84_BITS,
            'advanced_analysis_enabled': Config.ENABLE_ADVANCED_ANALYSIS,
            'debug_mode': Config.DEBUG_MODE
        },
        'supported_features': {
            'quantum_circuit_simulation': circuit_simulator is not None,
            'bloch_sphere_visualization': bloch_visualizer is not None,
            'bb84_protocol': bb84_protocol is not None,
            'quantum_channel_simulation': quantum_channel is not None,
            'security_analysis': security_analyzer is not None
        }
    }
    
    return jsonify(docs)

# ENHANCED: Test endpoint for debugging
@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for debugging frontend-backend communication"""
    try:
        method = request.method
        data = request.get_json() if request.method == 'POST' else None
        
        test_response = {
            'method': method,
            'received_data': data,
            'server_time': datetime.datetime.now().isoformat(),
            'configuration': {
                'max_qubits': Config.MAX_QUBITS,
                'max_bb84_bits': Config.MAX_BB84_BITS,
                'debug_mode': Config.DEBUG_MODE
            },
            'test_statevector': [1, 0],
            'test_state_analysis': {
                'amplitudes': {
                    'alpha': {'real': 1.0, 'imag': 0.0},
                    'beta': {'real': 0.0, 'imag': 0.0}
                },
                'probabilities': {
                    'P(|0⟩)': 1.0,
                    'P(|1⟩)': 0.0
                }
            }
        }
        
        logger.info(f"Test endpoint called with method {method}")
        return jsonify(create_success_response(test_response, "Test endpoint working"))
        
    except Exception as e:
        return create_error_response("Test endpoint failed", str(e), 500)

# ENHANCED: System status endpoint
@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status and capabilities"""
    try:
        system_status = {
            'components': {
                'quantum_simulators': {
                    'circuit_simulator': {
                        'available': circuit_simulator is not None,
                        'backend': getattr(circuit_simulator, 'backend', 'unknown') if circuit_simulator else None
                    },
                    'advanced_simulator': {
                        'available': advanced_simulator is not None,
                        'backend': 'QuTiP' if advanced_simulator else None
                    },
                    'bloch_visualizer': {
                        'available': bloch_visualizer is not None,
                        'style_theme': getattr(bloch_visualizer, 'style_theme', 'unknown') if bloch_visualizer else None
                    }
                },
                'cryptography': {
                    'bb84_protocol': {
                        'available': bb84_protocol is not None,
                        'backend': getattr(bb84_protocol, 'backend', 'unknown') if bb84_protocol else None
                    },
                    'quantum_channel': {
                        'available': quantum_channel is not None
                    },
                    'security_analyzer': {
                        'available': security_analyzer is not None
                    }
                }
            },
            'capabilities': {
                'circuit_simulation': circuit_simulator is not None,
                'advanced_simulations': advanced_simulator is not None,
                'bb84_protocol': bb84_protocol is not None,
                'security_analysis': security_analyzer is not None,
                'bloch_visualizations': bloch_visualizer is not None,
                'quantum_channel_simulation': quantum_channel is not None
            },
            'configuration': {
                'max_qubits': Config.MAX_QUBITS,
                'max_bb84_bits': Config.MAX_BB84_BITS,
                'advanced_analysis_enabled': Config.ENABLE_ADVANCED_ANALYSIS,
                'debug_mode': Config.DEBUG_MODE,
                'log_level': Config.LOG_LEVEL
            },
            'debug_info': {
                'total_components': sum([
                    circuit_simulator is not None,
                    bloch_visualizer is not None,
                    advanced_simulator is not None,
                    bb84_protocol is not None,
                    quantum_channel is not None,
                    security_analyzer is not None
                ]),
                'python_path': sys.path[0],
                'working_directory': os.getcwd(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }
        
        return jsonify(create_success_response(system_status))
        
    except Exception as e:
        return create_error_response("System status retrieval failed", str(e), 500)

if __name__ == '__main__':
    logger.info("Starting Enhanced Quantum Cryptography Tutorial API v2.2.0")
    logger.info(f"Configuration - Max Qubits: {Config.MAX_QUBITS}, "
               f"Max BB84 Bits: {Config.MAX_BB84_BITS}, "
               f"Advanced Analysis: {Config.ENABLE_ADVANCED_ANALYSIS}")
    logger.info(f"Component Status - Circuit: {circuit_simulator is not None}, "
               f"Bloch: {bloch_visualizer is not None}, "
               f"Advanced: {advanced_simulator is not None}, "
               f"BB84: {bb84_protocol is not None}, "
               f"Channel: {quantum_channel is not None}, "
               f"Security: {security_analyzer is not None}")
    
    # ENHANCED: Development server configuration
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=Config.DEBUG_MODE,
            threaded=True,
            use_reloader=False  # Disable reloader for stability
        )
    except Exception as e:
        logger.error(f"Failed to start Flask server: {str(e)}")
        raise
