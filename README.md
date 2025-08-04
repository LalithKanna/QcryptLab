# Quantum Cryptography Tutorial Website

A comprehensive educational platform for learning quantum computing fundamentals and quantum key distribution protocols using interactive visualizations.

## 🎯 Project Overview

This project provides an interactive web-based tutorial for quantum cryptography, featuring:

1. **Quantum Computing Basics** - Interactive tutorials with Bloch sphere visualizations
2. **BB84 Protocol Simulation** - Step-by-step quantum key distribution implementation
3. **Real-time Quantum Simulations** - Using Qiskit and QuTiP for backend computations
4. **Educational Interface** - Drag-and-drop circuit builder and interactive demonstrations

## 🛠️ Technology Stack

### Frontend
- **HTML5/CSS3/Vanilla JavaScript** - Pure web technologies for maximum compatibility
- **Interactive Visualizations** - Canvas-based quantum state representations
- **Responsive Design** - Mobile-friendly educational interface

### Backend
- **Python Flask** - Lightweight web framework
- **Qiskit** - IBM's quantum computing framework for circuit simulation
- **QuTiP** - Quantum Toolbox in Python for advanced simulations
- **Matplotlib** - Server-side visualization generation with base64 encoding

### Key Features
- **CORS-enabled API** - Seamless frontend-backend communication
- **Real-time Bloch Sphere Generation** - Dynamic quantum state visualization
- **Complete BB84 Implementation** - Alice, Bob, and Eve simulation
- **Educational Error Handling** - User-friendly quantum computing education

## 📁 Project Structure

```
quantum-crypto-tutorial/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── run.py                            # Application entry point
├── frontend/                         # Client-side application
│   ├── index.html                    # Main landing page
│   ├── quantum-basics/               # Quantum computing tutorials
│   │   ├── quantum-intro.html        # Introduction to quantum computing
│   │   ├── qubits-tutorial.html      # Qubits and superposition
│   │   ├── quantum-gates.html        # Quantum gate operations
│   │   └── quantum-circuits.html     # Interactive circuit builder
│   ├── bb84-protocol/                # Quantum key distribution
│   │   ├── bb84-tutorial.html        # BB84 protocol theory
│   │   └── bb84-simulator.html       # Interactive BB84 simulation
│   ├── css/                          # Stylesheets
│   │   ├── styles.css                # Global styles and layout
│   │   ├── quantum-visualizations.css # Quantum-specific styling
│   │   └── bb84-interface.css        # BB84 protocol interface
│   ├── js/                           # JavaScript modules
│   │   ├── main.js                   # Core application logic
│   │   ├── quantum-visualizer.js     # Circuit builder and visualizations
│   │   ├── bloch-sphere.js           # Bloch sphere rendering
│   │   └── bb84-simulator.js         # BB84 protocol interface
│   └── assets/                       # Static assets
│       └── images/                   # Educational images and icons
├── backend/                          # Server-side application
│   ├── app.py                        # Main Flask application
│   ├── quantum_simulator/            # Quantum simulation modules
│   │   ├── __init__.py
│   │   ├── circuit_simulator.py      # Qiskit-based circuit simulation
│   │   ├── state_visualizer.py       # Bloch sphere generation
│   │   └── advanced_simulator.py     # QuTiP integration
│   ├── cryptography/                 # Quantum cryptography implementations
│   │   ├── __init__.py
│   │   ├── bb84_protocol.py          # Complete BB84 implementation
│   │   ├── quantum_channel.py        # Quantum channel simulation
│   │   └── security_analysis.py      # Eavesdropping detection
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── image_generator.py        # Base64 image encoding
│       └── data_validators.py        # Input validation
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Modern web browser with JavaScript enabled
- Internet connection for initial setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd quantum-crypto-tutorial
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the Backend Server
```bash
python run.py
```
The Flask server will start on `http://localhost:5000`

### Step 4: Serve the Frontend
You have several options:

**Option A: Python HTTP Server**
```bash
cd frontend
python -m http.server 3000
```
Access the application at `http://localhost:3000`

**Option B: Direct File Access**
Open `frontend/index.html` directly in your web browser

**Option C: Live Server (if using VS Code)**
Use the Live Server extension to serve the frontend directory

## 🎓 Learning Modules

### 1. Quantum Computing Fundamentals
- **Introduction to Quantum Computing** - Basic concepts and principles
- **Qubits and Superposition** - Understanding quantum states with Bloch sphere
- **Quantum Gates and Operations** - Interactive gate manipulation
- **Interactive Circuit Builder** - Drag-and-drop quantum circuit construction

### 2. Quantum Cryptography
- **BB84 Protocol Theory** - Understanding quantum key distribution
- **Interactive BB84 Simulation** - Step-by-step protocol implementation

## 🔧 API Endpoints

### Quantum Computing Endpoints
- `GET /api/quantum-basics/<lesson>` - Retrieve tutorial content
- `POST /api/simulate-circuit` - Execute quantum circuit simulation
- `POST /api/visualize-bloch` - Generate Bloch sphere visualization
- `POST /api/advanced-simulation` - Complex quantum dynamics with QuTiP

### BB84 Protocol Endpoints
- `POST /api/bb84/simulate` - Complete BB84 protocol execution
- `GET /api/bb84/tutorial` - BB84 educational content
- `POST /api/bb84/step-simulation` - Individual protocol step simulation

### Utility Endpoints
- `GET /api/health` - Service health check
- `GET /api/quantum-states` - Common quantum state information

## 💡 Key Features

### Interactive Quantum Circuit Builder
- Drag-and-drop interface for quantum gates (H, X, Y, Z, CNOT)
- Real-time statevector calculation and visualization
- Circuit diagram generation with Qiskit
- Educational error messages and validation

### Bloch Sphere Visualizations
- Dynamic 3D Bloch sphere rendering using Matplotlib
- Interactive parameter adjustment (θ, φ angles)
- Real-time quantum state probability calculations
- Base64 encoded images for web display

### Complete BB84 Implementation
- Alice's random bit and basis generation
- Quantum state preparation and encoding
- Quantum channel simulation with noise
- Bob's random measurement basis selection
- Key sifting and eavesdropping detection
- Security analysis with QBER calculation

## 🧪 Educational Features

### Progressive Learning Path
1. Start with classical vs quantum computing concepts
2. Learn about qubits and superposition
3. Explore quantum gates and their effects
4. Build quantum circuits interactively
5. Understand quantum cryptography principles
6. Simulate the BB84 protocol step-by-step

### Interactive Demonstrations
- Classical vs quantum search algorithm comparison
- Qubit measurement simulation with probability visualization
- Custom quantum state creation and analysis
- Real-time protocol security analysis

## 🔒 Security Considerations

- Input validation for all quantum parameters
- Rate limiting on simulation endpoints
- Resource constraints for circuit size (max 10 qubits)
- Educational-focused error messages
- No persistent data storage (stateless design)

## 🎨 Customization Guide

### Adding New Quantum Gates
1. Define gate function in `circuit_simulator.py`
2. Add gate to `supported_gates` dictionary
3. Update frontend gate palette in `quantum-visualizer.js`
4. Add corresponding CSS styling

### Extending BB84 Protocol
1. Implement new protocol variations in `bb84_protocol.py`
2. Add corresponding API endpoints in `app.py`
3. Create frontend interface components
4. Update educational content

### Adding New Visualizations
1. Create visualization functions in `state_visualizer.py`
2. Add API endpoints for new visualizations
3. Implement frontend rendering logic
4. Update educational tutorials

## 🐛 Troubleshooting

### Common Issues

**Backend fails to start:**
- Verify Python version (3.8+)
- Check all dependencies are installed: `pip install -r requirements.txt`
- Ensure port 5000 is not in use

**Frontend cannot connect to backend:**
- Verify CORS settings in `app.py`
- Check that backend is running on `http://localhost:5000`
- Confirm frontend is making requests to correct URL

**Quantum simulations fail:**
- Check circuit parameters are within limits (max 10 qubits)
- Verify gate specifications are properly formatted
- Review server logs for detailed error messages

**Visualizations not displaying:**
- Ensure matplotlib is properly installed with Agg backend
- Check browser developer console for JavaScript errors
- Verify base64 image encoding is working correctly

## 🔮 Future Development

### Planned Features
- Additional quantum algorithms (Shor's, Grover's)
- More quantum cryptography protocols (E91, SARG04)
- Advanced error correction demonstrations
- Quantum machine learning tutorials
- Multi-language support

### Technical Improvements
- WebAssembly integration for client-side simulations
- Real quantum hardware integration (IBM Quantum, IonQ)
- Progressive Web App (PWA) capabilities
- Advanced 3D visualizations with WebGL

## 📚 Educational Resources

### Prerequisites
- Basic linear algebra (vectors, matrices)
- Elementary probability theory
- High school level mathematics
- No prior quantum physics knowledge required

### Learning Outcomes
After completing this tutorial, students will:
- Understand fundamental quantum computing principles
- Be able to construct and analyze quantum circuits
- Comprehend quantum cryptography protocols
- Recognize quantum advantage in specific applications

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow existing code style and documentation standards
4. Add tests for new functionality
5. Submit a pull request with detailed description

### Code Guidelines
- Use descriptive variable names and comments
- Follow PEP 8 for Python code
- Use modern JavaScript (ES6+) features
- Maintain responsive design principles
- Include educational explanations in code comments

## 📄 License

This project is created for educational purposes. Please refer to the license file for detailed terms and conditions.

## 🙏 Acknowledgments

- **IBM Qiskit Team** - For the excellent quantum computing framework
- **QuTiP Community** - For advanced quantum simulation capabilities
- **Quantum Computing Education Community** - For inspiration and resources
- **Open Source Contributors** - For various libraries and tools used

## 📧 Support

For questions, issues, or contributions:
- Open an issue on the project repository
- Review the troubleshooting section
- Check existing documentation and code comments
- Follow educational best practices for quantum computing

---

**Built with ❤️ for quantum computing education**