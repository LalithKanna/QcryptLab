"""
Quantum Cryptography Tutorial Website
=====================================

A comprehensive educational platform for learning quantum computing fundamentals
and quantum key distribution protocols using interactive visualizations.

Author: Generated for Educational Purpose
Tech Stack: HTML/CSS/JavaScript + Python Flask + Qiskit + QuTiP
License: Educational Use Only
"""

from backend.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)