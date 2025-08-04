/**
 * Enhanced Quantum Visualizer JavaScript
 * ======================================
 * 
 * Provides interactive quantum state visualizations with full backend integration:
 * - Bloch sphere rendering with API integration
 * - Quantum circuit diagrams for BB84 protocol
 * - State vector animations with real-time updates
 * - Enhanced error handling and defensive programming
 * - Fixed JSON response structure handling
 * - Resolved "Cannot read properties of undefined" errors
 */

class QuantumVisualizer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.currentState = null;
        this.animationId = null;
        this.isInitialized = false;
        this.apiBaseUrl = '/api';
        this.loadingStates = new Set();
        
        // Enhanced configuration
        this.config = {
            apiTimeout: 15000,
            animationDuration: 1000,
            retryAttempts: 3,
            debugMode: false
        };
        
        this.init();
    }
    
    init() {
        this.canvas = document.getElementById('bloch-canvas');
        if (!this.canvas) {
            console.warn('Bloch sphere canvas not found');
            this.createFallbackCanvas();
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        this.drawBlochSphere();
        this.isInitialized = true;
        
        // Initialize enhanced controls
        this.setupStateSelector();
        this.setupInteractiveControls();
        this.setupErrorHandling();
        
        // Initialize with default state
        this.loadQuantumStateFromAPI(0, 0);
    }
    
    createFallbackCanvas() {
        // Create canvas if it doesn't exist
        const container = document.getElementById('bloch-container') || 
                         document.getElementById('visualization-container') ||
                         document.body;
        
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'bloch-canvas';
        this.canvas.width = 400;
        this.canvas.height = 400;
        this.canvas.style.border = '1px solid #ccc';
        container.appendChild(this.canvas);
        
        console.log('Created fallback canvas for Bloch sphere visualization');
    }
    
    setupCanvas() {
        // Set canvas size for high DPI displays
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        
        // Set canvas display size
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }
    
    /**
     * **FIXED: Enhanced API integration with proper error handling**
     */
    async loadQuantumStateFromAPI(theta, phi, bb84Context = null) {
        const loadingId = `bloch-${Date.now()}`;
        this.addLoadingState(loadingId);
        
        try {
            const response = await this.fetchWithRetry('/api/visualize-bloch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    theta: theta,
                    phi: phi,
                    show_state_info: true,
                    bb84_context: bb84Context,
                    style_theme: 'educational'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Raw API Response:', data); // Debug logging
            
            // **FIXED: Comprehensive response validation**
            if (!this.validateAPIResponse(data)) {
                throw new Error('Invalid API response structure');
            }

            // **FIXED: Safe property access with multiple fallback paths**
            const stateAnalysis = this.safeGetProperty(data, 'state_analysis', {});
            const blochCoords = this.safeGetProperty(stateAnalysis, 'bloch_coordinates', {});
            const sphericalCoords = this.safeGetProperty(stateAnalysis, 'spherical_coordinates', {});
            const amplitudes = this.safeGetProperty(stateAnalysis, 'amplitudes', null);
            const probabilities = this.safeGetProperty(stateAnalysis, 'probabilities', null);
            
            // **FIXED: Build current state with defensive programming**
            this.currentState = {
                theta: this.safeGetNumber(sphericalCoords, 'theta', theta),
                phi: this.safeGetNumber(sphericalCoords, 'phi', phi),
                x: this.safeGetNumber(blochCoords, 'x', 0),
                y: this.safeGetNumber(blochCoords, 'y', 0),
                z: this.safeGetNumber(blochCoords, 'z', 1),
                amplitudes: amplitudes,
                probabilities: probabilities,
                name: this.getStateName(theta, phi),
                apiData: data // Store complete API response for debugging
            };

            // **FIXED: Display visualization with error handling**
            if (data.bloch_image) {
                await this.displayServerBlochSphere(data.bloch_image);
            } else {
                this.drawBlochSphere();
            }
            
            // Update state information panel
            this.updateStateInfoPanel(stateAnalysis, bb84Context);
            
            this.log('State loaded successfully:', this.currentState);
            
        } catch (error) {
            console.error('Network error loading visualization:', error.message);
            this.handleVisualizationError(error);
            
            // **ENHANCED: Graceful fallback to local rendering**
            this.setQuantumStateLocal(theta, phi);
            
        } finally {
            this.removeLoadingState(loadingId);
        }
    }
    
    /**
     * **NEW: Comprehensive API response validation**
     */
    validateAPIResponse(data) {
        if (!data || typeof data !== 'object') {
            console.error('API response is not a valid object:', data);
            return false;
        }
        
        // Check for success flag (new response format)
        if (data.hasOwnProperty('success') && !data.success) {
            console.error('API request failed:', data.error || 'Unknown error');
            return false;
        }
        
        // Validate that we have either bloch_image or state_analysis
        if (!data.bloch_image && !data.state_analysis) {
            console.warn('API response missing both bloch_image and state_analysis');
            return false;
        }
        
        return true;
    }
    
    /**
     * **NEW: Safe property getter with fallback**
     */
    safeGetProperty(obj, prop, fallback = null) {
        try {
            return obj && obj.hasOwnProperty(prop) ? obj[prop] : fallback;
        } catch (error) {
            console.warn(`Failed to access property '${prop}':`, error);
            return fallback;
        }
    }
    
    /**
     * **NEW: Safe number getter with validation**
     */
    safeGetNumber(obj, prop, fallback = 0) {
        try {
            const value = this.safeGetProperty(obj, prop, fallback);
            return typeof value === 'number' && !isNaN(value) ? value : fallback;
        } catch (error) {
            console.warn(`Failed to get number from '${prop}':`, error);
            return fallback;
        }
    }
    
    /**
     * Enhanced fetch with retry mechanism and better error handling
     */
    async fetchWithRetry(url, options, attempt = 1) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.apiTimeout);
            
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            return response;
            
        } catch (error) {
            if (attempt < this.config.retryAttempts && error.name !== 'AbortError') {
                this.log(`Retry attempt ${attempt} for ${url}`);
                await this.delay(1000 * attempt);
                return this.fetchWithRetry(url, options, attempt + 1);
            }
            
            // Enhanced error information
            if (error.name === 'AbortError') {
                throw new Error(`Request timeout (${this.config.apiTimeout}ms) for ${url}`);
            }
            throw error;
        }
    }
    
    /**
     * **ENHANCED: Display server-generated Bloch sphere image**
     */
    async displayServerBlochSphere(base64Image) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => {
                try {
                    // Clear canvas and draw server image
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    
                    // Calculate scaling to fit canvas
                    const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
                    const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);
                    
                    const scale = Math.min(
                        canvasWidth / img.width,
                        canvasHeight / img.height
                    ) * 0.9;
                    
                    const x = (canvasWidth - img.width * scale) / 2;
                    const y = (canvasHeight - img.height * scale) / 2;
                    
                    this.ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                    
                    // Add interactive overlay
                    this.drawInteractiveOverlay();
                    
                    resolve();
                    
                } catch (drawError) {
                    console.error('Failed to draw server image:', drawError);
                    reject(drawError);
                }
            };
            
            img.onerror = (error) => {
                console.warn('Failed to load server Bloch sphere image, using local rendering');
                this.drawBlochSphere();
                reject(error);
            };
            
            // Enhanced base64 validation
            if (!base64Image || typeof base64Image !== 'string') {
                console.error('Invalid base64 image data');
                this.drawBlochSphere();
                reject(new Error('Invalid base64 image data'));
                return;
            }
            
            img.src = `data:image/png;base64,${base64Image}`;
        });
    }
    
    /**
     * Local Bloch sphere rendering (enhanced fallback)
     */
    drawBlochSphere() {
        try {
            const dpr = window.devicePixelRatio || 1;
            const centerX = this.canvas.width / (2 * dpr);
            const centerY = this.canvas.height / (2 * dpr);
            const radius = Math.min(centerX, centerY) * 0.8;
            
            // Clear canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Set canvas context for high DPI
            this.ctx.save();
            this.ctx.scale(1/dpr, 1/dpr);
            
            // Draw enhanced sphere
            this.drawSphere(centerX, centerY, radius);
            this.drawCoordinateAxes(centerX, centerY, radius);
            this.drawGridLines(centerX, centerY, radius);
            this.drawQuantumLabels(centerX, centerY, radius);
            
            // Draw state vector if available
            if (this.currentState) {
                this.drawStateVector(centerX, centerY, radius);
            }
            
            this.ctx.restore();
            
        } catch (error) {
            console.error('Failed to draw local Bloch sphere:', error);
            this.drawErrorVisualization();
        }
    }
    
    /**
     * **NEW: Draw error visualization when everything fails**
     */
    drawErrorVisualization() {
        try {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw error message
            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.font = 'bold 16px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
                'Visualization Error',
                this.canvas.width / 2,
                this.canvas.height / 2 - 20
            );
            
            this.ctx.fillStyle = '#666';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(
                'Unable to render Bloch sphere',
                this.canvas.width / 2,
                this.canvas.height / 2 + 10
            );
            
        } catch (finalError) {
            console.error('Even error visualization failed:', finalError);
        }
    }
    
    /**
     * Enhanced sphere drawing with better styling
     */
    drawSphere(centerX, centerY, radius) {
        // Main sphere outline
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        this.ctx.strokeStyle = '#4a90e2';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        
        // Add subtle sphere shading
        const gradient = this.ctx.createRadialGradient(
            centerX - radius * 0.3, centerY - radius * 0.3, 0,
            centerX, centerY, radius
        );
        gradient.addColorStop(0, 'rgba(74, 144, 226, 0.1)');
        gradient.addColorStop(1, 'rgba(74, 144, 226, 0.05)');
        
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
    }
    
    drawCoordinateAxes(centerX, centerY, radius) {
        const axisLength = radius * 1.3;
        const axisWidth = 2;
        
        // X-axis (red) - |+⟩/|−⟩ basis
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - axisLength, centerY);
        this.ctx.lineTo(centerX + axisLength, centerY);
        this.ctx.strokeStyle = '#e74c3c';
        this.ctx.lineWidth = axisWidth;
        this.ctx.stroke();
        
        // Y-axis (green) - |+i⟩/|−i⟩ basis
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY - axisLength);
        this.ctx.lineTo(centerX, centerY + axisLength);
        this.ctx.strokeStyle = '#2ecc71';
        this.ctx.lineWidth = axisWidth;
        this.ctx.stroke();
        
        // Z-axis indicator (blue) - |0⟩/|1⟩ basis
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY - radius * 0.9, 8, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#3498db';
        this.ctx.fill();
        
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY + radius * 0.9, 8, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#9b59b6';
        this.ctx.fill();
    }
    
    /**
     * Enhanced quantum state labels
     */
    drawQuantumLabels(centerX, centerY, radius) {
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // BB84 basis labels
        this.ctx.fillStyle = '#e74c3c';
        this.ctx.fillText('|+⟩', centerX + radius * 1.4, centerY);
        this.ctx.fillText('|−⟩', centerX - radius * 1.4, centerY);
        
        this.ctx.fillStyle = '#2ecc71';
        this.ctx.fillText('|+i⟩', centerX, centerY - radius * 1.4);
        this.ctx.fillText('|−i⟩', centerX, centerY + radius * 1.4);
        
        this.ctx.fillStyle = '#3498db';
        this.ctx.fillText('|0⟩', centerX + 25, centerY - radius * 1.1);
        
        this.ctx.fillStyle = '#9b59b6';
        this.ctx.fillText('|1⟩', centerX + 25, centerY + radius * 1.1);
        
        // Basis annotations
        this.ctx.font = '12px Arial';
        this.ctx.fillStyle = '#7f8c8d';
        this.ctx.fillText('Diagonal Basis', centerX + radius * 1.6, centerY - 20);
        this.ctx.fillText('Computational Basis', centerX + 40, centerY - radius * 1.3);
    }
    
    drawGridLines(centerX, centerY, radius) {
        this.ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        this.ctx.lineWidth = 1;
        
        // Enhanced latitude lines
        for (let i = 1; i <= 3; i++) {
            const r = radius * (i / 4);
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
            this.ctx.stroke();
        }
        
        // Enhanced longitude lines
        for (let i = 0; i < 12; i++) {
            const angle = (i * Math.PI) / 6;
            const x1 = centerX + radius * Math.cos(angle);
            const y1 = centerY + radius * Math.sin(angle);
            const x2 = centerX - radius * Math.cos(angle);
            const y2 = centerY - radius * Math.sin(angle);
            
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();
        }
    }
    
    /**
     * **ENHANCED: State vector drawing with improved error handling**
     */
    drawStateVector(centerX, centerY, radius) {
        if (!this.currentState) return;
        
        try {
            // Use Bloch coordinates from API if available
            let x, y, z;
            if (this.currentState.x !== undefined && this.currentState.z !== undefined) {
                x = centerX + radius * this.currentState.x;
                y = centerY; // Y coordinate in 2D projection
                z = centerY - radius * this.currentState.z;
            } else {
                // Fallback to spherical coordinates
                const { theta, phi } = this.currentState;
                x = centerX + radius * Math.sin(theta) * Math.cos(phi);
                y = centerY - radius * Math.sin(theta) * Math.sin(phi);
                z = centerY - radius * Math.cos(theta);
            }
            
            // Draw state vector arrow with enhanced styling
            const gradient = this.ctx.createLinearGradient(centerX, centerY, x, z);
            gradient.addColorStop(0, '#f39c12');
            gradient.addColorStop(1, '#e67e22');
            
            this.ctx.beginPath();
            this.ctx.moveTo(centerX, centerY);
            this.ctx.lineTo(x, z);
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 4;
            this.ctx.stroke();
            
            // Enhanced arrowhead
            const angle = Math.atan2(z - centerY, x - centerX);
            const arrowLength = 20;
            const arrowAngle = Math.PI / 6;
            
            this.ctx.beginPath();
            this.ctx.moveTo(x, z);
            this.ctx.lineTo(
                x - arrowLength * Math.cos(angle - arrowAngle),
                z - arrowLength * Math.sin(angle - arrowAngle)
            );
            this.ctx.moveTo(x, z);
            this.ctx.lineTo(
                x - arrowLength * Math.cos(angle + arrowAngle),
                z - arrowLength * Math.sin(angle + arrowAngle)
            );
            this.ctx.strokeStyle = '#e67e22';
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
            
            // Enhanced state point
            this.ctx.beginPath();
            this.ctx.arc(x, z, 8, 0, 2 * Math.PI);
            this.ctx.fillStyle = '#f39c12';
            this.ctx.fill();
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Enhanced state label with probabilities
            this.drawStateLabel(x, z);
            
        } catch (error) {
            console.error('Failed to draw state vector:', error);
        }
    }
    
    /**
     * **ENHANCED: State label with improved probability display**
     */
    drawStateLabel(x, z) {
        if (!this.currentState) return;
        
        try {
            const { theta, phi, probabilities, name } = this.currentState;
            
            // Background for label
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.fillRect(x - 80, z - 50, 160, 40);
            
            // State name and angles
            this.ctx.font = 'bold 14px Arial';
            this.ctx.fillStyle = '#ffffff';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
                name || `θ=${(theta * 180 / Math.PI).toFixed(1)}°, φ=${(phi * 180 / Math.PI).toFixed(1)}°`,
                x, z - 35
            );
            
            // Measurement probabilities if available
            if (probabilities && probabilities['P(|0⟩)'] !== undefined) {
                this.ctx.font = '12px Arial';
                this.ctx.fillText(
                    `P(|0⟩)=${probabilities['P(|0⟩)'].toFixed(3)}, P(|1⟩)=${probabilities['P(|1⟩)'].toFixed(3)}`,
                    x, z - 20
                );
            }
            
        } catch (error) {
            console.error('Failed to draw state label:', error);
        }
    }
    
    /**
     * **ENHANCED: State setting with better error handling**
     */
    async setQuantumState(stateKey, animate = true) {
        try {
            const state = this.getStateFromSelector(stateKey);
            
            if (animate && this.currentState) {
                await this.animateStateTransition(state);
            } else {
                await this.loadQuantumStateFromAPI(state.theta, state.phi);
            }
            
        } catch (error) {
            console.error('Failed to set quantum state:', error);
            this.handleVisualizationError(error);
        }
    }
    
    /**
     * Local state setting (enhanced fallback)
     */
    setQuantumStateLocal(theta, phi) {
        try {
            this.currentState = {
                theta: theta,
                phi: phi,
                x: Math.sin(theta) * Math.cos(phi),
                y: Math.sin(theta) * Math.sin(phi),
                z: Math.cos(theta),
                name: this.getStateName(theta, phi),
                isLocal: true
            };
            this.drawBlochSphere();
            
        } catch (error) {
            console.error('Failed to set local quantum state:', error);
            this.drawErrorVisualization();
        }
    }
    
    /**
     * **ENHANCED: State selector setup with error handling**
     */
    setupStateSelector() {
        try {
            const stateSelector = document.getElementById('state-selector');
            if (stateSelector) {
                stateSelector.addEventListener('change', async (e) => {
                    try {
                        await this.setQuantumState(e.target.value, true);
                    } catch (error) {
                        console.error('State selector error:', error);
                    }
                });
            }
            
            // Add custom state input
            this.setupCustomStateInput();
            
        } catch (error) {
            console.error('Failed to setup state selector:', error);
        }
    }
    
    /**
     * **ENHANCED: Custom state input controls**
     */
    setupCustomStateInput() {
        try {
            const thetaSlider = document.getElementById('theta-slider');
            const phiSlider = document.getElementById('phi-slider');
            
            if (thetaSlider && phiSlider) {
                const updateState = async () => {
                    try {
                        const theta = parseFloat(thetaSlider.value);
                        const phi = parseFloat(phiSlider.value);
                        
                        if (!isNaN(theta) && !isNaN(phi)) {
                            await this.loadQuantumStateFromAPI(theta, phi);
                        }
                    } catch (error) {
                        console.error('Slider update error:', error);
                    }
                };
                
                thetaSlider.addEventListener('input', updateState);
                phiSlider.addEventListener('input', updateState);
            }
            
        } catch (error) {
            console.error('Failed to setup custom state input:', error);
        }
    }
    
    /**
     * Interactive controls setup
     */
    setupInteractiveControls() {
        try {
            // Mouse interaction for sphere rotation
            let isDragging = false;
            let lastX, lastY;
            
            this.canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
            });
            
            this.canvas.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                try {
                    const deltaX = e.clientX - lastX;
                    const deltaY = e.clientY - lastY;
                    
                    // Convert mouse movement to spherical coordinates
                    if (this.currentState) {
                        const newPhi = this.currentState.phi + deltaX * 0.01;
                        const newTheta = Math.max(0, Math.min(Math.PI, this.currentState.theta + deltaY * 0.01));
                        
                        this.loadQuantumStateFromAPI(newTheta, newPhi);
                    }
                    
                    lastX = e.clientX;
                    lastY = e.clientY;
                    
                } catch (error) {
                    console.error('Mouse move error:', error);
                }
            });
            
            this.canvas.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
        } catch (error) {
            console.error('Failed to setup interactive controls:', error);
        }
    }
    
    /**
     * Enhanced error handling setup
     */
    setupErrorHandling() {
        window.addEventListener('error', (e) => {
            if (e.message.includes('quantum') || e.message.includes('visualization')) {
                this.handleVisualizationError(e.error);
            }
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.message && 
                (event.reason.message.includes('quantum') || 
                 event.reason.message.includes('visualization'))) {
                console.error('Unhandled quantum visualization error:', event.reason);
                this.handleVisualizationError(event.reason);
                event.preventDefault();
            }
        });
    }
    
    /**
     * **ENHANCED: State information panel with safe property access**
     */
    updateStateInfoPanel(stateAnalysis, bb84Context) {
        try {
            const infoPanel = document.getElementById('state-info-panel');
            if (!infoPanel || !stateAnalysis) return;
            
            const amplitudes = this.safeGetProperty(stateAnalysis, 'amplitudes', null);
            const probabilities = this.safeGetProperty(stateAnalysis, 'probabilities', null);
            const blochCoords = this.safeGetProperty(stateAnalysis, 'bloch_coordinates', null);
            const bb84Analysis = this.safeGetProperty(stateAnalysis, 'bb84_analysis', null);
            
            let infoHTML = '<div class="state-info">';
            
            // Amplitudes
            if (amplitudes && amplitudes.alpha && amplitudes.beta) {
                infoHTML += `
                    <div class="info-section">
                        <h4>State Amplitudes</h4>
                        <p>α = ${amplitudes.alpha.real.toFixed(3)} + ${amplitudes.alpha.imag.toFixed(3)}i</p>
                        <p>β = ${amplitudes.beta.real.toFixed(3)} + ${amplitudes.beta.imag.toFixed(3)}i</p>
                    </div>
                `;
            }
            
            // Probabilities
            if (probabilities) {
                const p0 = this.safeGetNumber(probabilities, 'P(|0⟩)', 0);
                const p1 = this.safeGetNumber(probabilities, 'P(|1⟩)', 0);
                infoHTML += `
                    <div class="info-section">
                        <h4>Measurement Probabilities</h4>
                        <p>P(|0⟩) = ${p0.toFixed(3)}</p>
                        <p>P(|1⟩) = ${p1.toFixed(3)}</p>
                    </div>
                `;
            }
            
            // Bloch coordinates
            if (blochCoords) {
                const x = this.safeGetNumber(blochCoords, 'x', 0);
                const y = this.safeGetNumber(blochCoords, 'y', 0);
                const z = this.safeGetNumber(blochCoords, 'z', 0);
                infoHTML += `
                    <div class="info-section">
                        <h4>Bloch Coordinates</h4>
                        <p>X = ${x.toFixed(3)}</p>
                        <p>Y = ${y.toFixed(3)}</p>
                        <p>Z = ${z.toFixed(3)}</p>
                    </div>
                `;
            }
            
            // BB84 analysis
            if (bb84Analysis && bb84Analysis.preferred_basis) {
                infoHTML += `
                    <div class="info-section">
                        <h4>BB84 Analysis</h4>
                        <p>Preferred Basis: ${bb84Analysis.preferred_basis}</p>
                        <p>Closest State: ${bb84Analysis.closest_bb84_state || 'Unknown'}</p>
                    </div>
                `;
            }
            
            infoHTML += '</div>';
            infoPanel.innerHTML = infoHTML;
            
        } catch (error) {
            console.error('Failed to update state info panel:', error);
        }
    }
    
    /**
     * Enhanced state definitions
     */
    getStateFromSelector(value) {
        const states = {
            '0': { theta: 0, phi: 0, name: '|0⟩' },
            '1': { theta: Math.PI, phi: 0, name: '|1⟩' },
            'plus': { theta: Math.PI / 2, phi: 0, name: '|+⟩' },
            'minus': { theta: Math.PI / 2, phi: Math.PI, name: '|−⟩' },
            'plus-i': { theta: Math.PI / 2, phi: Math.PI / 2, name: '|+i⟩' },
            'minus-i': { theta: Math.PI / 2, phi: 3 * Math.PI / 2, name: '|−i⟩' },
            // BB84 specific states
            'bb84-comp-0': { theta: 0, phi: 0, name: 'BB84 |0⟩' },
            'bb84-comp-1': { theta: Math.PI, phi: 0, name: 'BB84 |1⟩' },
            'bb84-diag-0': { theta: Math.PI / 2, phi: 0, name: 'BB84 |+⟩' },
            'bb84-diag-1': { theta: Math.PI / 2, phi: Math.PI, name: 'BB84 |−⟩' }
        };
        
        return states[value] || states['0'];
    }
    
    getStateName(theta, phi) {
        const tolerance = 0.1;
        
        if (Math.abs(theta) < tolerance) return '|0⟩';
        if (Math.abs(theta - Math.PI) < tolerance) return '|1⟩';
        
        if (Math.abs(theta - Math.PI/2) < tolerance) {
            if (Math.abs(phi) < tolerance) return '|+⟩';
            if (Math.abs(phi - Math.PI) < tolerance) return '|−⟩';
            if (Math.abs(phi - Math.PI/2) < tolerance) return '|+i⟩';
            if (Math.abs(phi - 3*Math.PI/2) < tolerance) return '|−i⟩';
        }
        
        return `|ψ(${(theta*180/Math.PI).toFixed(0)}°,${(phi*180/Math.PI).toFixed(0)}°)⟩`;
    }
    
    /**
     * Enhanced animation with API integration
     */
    async animateStateTransition(targetState, duration = 1000) {
        if (!this.currentState) {
            await this.loadQuantumStateFromAPI(targetState.theta, targetState.phi);
            return;
        }
        
        const startState = { ...this.currentState };
        const startTime = Date.now();
        
        const animate = async () => {
            try {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                const easeProgress = this.easeInOutCubic(progress);
                
                const currentTheta = startState.theta + (targetState.theta - startState.theta) * easeProgress;
                const currentPhi = startState.phi + (targetState.phi - startState.phi) * easeProgress;
                
                // Update visualization locally during animation
                this.currentState = {
                    ...this.currentState,
                    theta: currentTheta,
                    phi: currentPhi,
                    name: targetState.name
                };
                
                this.drawBlochSphere();
                
                if (progress < 1) {
                    this.animationId = requestAnimationFrame(animate);
                } else {
                    // Final API call for accurate state
                    await this.loadQuantumStateFromAPI(targetState.theta, targetState.phi);
                }
                
            } catch (error) {
                console.error('Animation error:', error);
            }
        };
        
        animate();
    }
    
    /**
     * Loading state management
     */
    addLoadingState(id) {
        this.loadingStates.add(id);
        this.showLoadingIndicator();
    }
    
    removeLoadingState(id) {
        this.loadingStates.delete(id);
        if (this.loadingStates.size === 0) {
            this.hideLoadingIndicator();
        }
    }
    
    showLoadingIndicator() {
        const indicator = document.getElementById('loading-indicator');
        if (indicator) {
            indicator.style.display = 'block';
        } else {
            // Create loading indicator if it doesn't exist
            this.createLoadingIndicator();
        }
    }
    
    hideLoadingIndicator() {
        const indicator = document.getElementById('loading-indicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }
    
    createLoadingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'loading-indicator';
        indicator.innerHTML = '<div class="spinner"></div><p>Loading quantum state...</p>';
        indicator.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            z-index: 9999;
            text-align: center;
        `;
        document.body.appendChild(indicator);
    }
    
    /**
     * **ENHANCED: Error handling with user feedback**
     */
    handleVisualizationError(error) {
        console.error('Quantum visualization error:', error);
        
        // Show user-friendly error message
        let errorContainer = document.getElementById('visualization-error');
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'visualization-error';
            errorContainer.style.cssText = `
                background: #ff6b6b;
                color: white;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                display: none;
            `;
            
            const container = document.getElementById('bloch-container') || 
                             document.getElementById('visualization-container') ||
                             this.canvas.parentNode;
            if (container) {
                container.appendChild(errorContainer);
            }
        }
        
        errorContainer.innerHTML = `
            <div class="error-message">
                <h4>⚠️ Visualization Error</h4>
                <p>Unable to load quantum state visualization from server.</p>
                <p>Using local rendering fallback.</p>
                <button onclick="this.parentElement.parentElement.style.display='none'">Dismiss</button>
            </div>
        `;
        errorContainer.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorContainer) {
                errorContainer.style.display = 'none';
            }
        }, 5000);
    }
    
    drawInteractiveOverlay() {
        // Add interactive elements over server-generated image
        // This can include click targets, hover effects, etc.
        try {
            // Add state indicator overlay
            const dpr = window.devicePixelRatio || 1;
            const centerX = this.canvas.width / (2 * dpr);
            const centerY = this.canvas.height / (2 * dpr);
            
            this.ctx.save();
            this.ctx.scale(1/dpr, 1/dpr);
            
            // Add small indicator that this is server-generated
            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            this.ctx.fillRect(10, 10, 120, 25);
            this.ctx.fillStyle = 'black';
            this.ctx.font = '12px Arial';
            this.ctx.fillText('Server Generated', 15, 27);
            
            this.ctx.restore();
            
        } catch (error) {
            console.error('Failed to draw interactive overlay:', error);
        }
    }
    
    // Utility functions
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    log(...args) {
        if (this.config.debugMode) {
            console.log('[QuantumVisualizer]', ...args);
        }
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.loadingStates.clear();
        
        // Clean up event listeners
        try {
            this.canvas.removeEventListener('mousedown');
            this.canvas.removeEventListener('mousemove');
            this.canvas.removeEventListener('mouseup');
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }
}

// Enhanced BB84 Circuit Visualizer (keeping existing functionality)
class QuantumCircuitVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.circuit = [];
        this.qubits = 2;
        this.apiBaseUrl = '/api';
        
        this.init();
    }
    
    init() {
        if (!this.container) {
            console.warn('BB84 circuit container not found');
            return;
        }
        
        this.drawBB84Circuit();
        this.setupCircuitControls();
    }
    
    /**
     * Enhanced BB84 circuit diagram with step-by-step breakdown
     */
    drawBB84Circuit() {
        const circuitHTML = `
            <div class="bb84-circuit-container">
                <div class="circuit-header">
                    <h3>BB84 Quantum Key Distribution Protocol</h3>
                    <div class="circuit-controls">
                        <button id="step-btn" class="btn-primary">Step Through Protocol</button>
                        <button id="simulate-btn" class="btn-secondary">Simulate Full Protocol</button>
                    </div>
                </div>
                
                <div class="bb84-circuit-diagram">
                    <div class="circuit-step" id="step-1">
                        <h4>Step 1: Alice's Preparation</h4>
                        <div class="circuit-row">
                            <span class="qubit-label">Alice's Qubit:</span>
                            <div class="gate-cell prep">PREP</div>
                            <div class="wire-cell">───</div>
                            <div class="gate-cell" id="basis-gate">?</div>
                            <div class="wire-cell">───</div>
                            <div class="transmission-cell">→ Channel</div>
                        </div>
                    </div>
                    
                    <div class="circuit-step" id="step-2">
                        <h4>Step 2: Quantum Transmission</h4>
                        <div class="circuit-row">
                            <span class="qubit-label">Quantum Channel:</span>
                            <div class="wire-cell">~~~</div>
                            <div class="noise-cell">NOISE</div>
                            <div class="wire-cell">~~~</div>
                            <div class="eve-cell" id="eve-indicator" style="display:none">EVE</div>
                            <div class="wire-cell">~~~</div>
                        </div>
                    </div>
                    
                    <div class="circuit-step" id="step-3">
                        <h4>Step 3: Bob's Measurement</h4>
                        <div class="circuit-row">
                            <span class="qubit-label">Bob's Measurement:</span>
                            <div class="wire-cell">───</div>
                            <div class="gate-cell" id="bob-basis-gate">?</div>
                            <div class="wire-cell">───</div>
                            <div class="measure-cell">M</div>
                            <div class="result-cell" id="measurement-result">?</div>
                        </div>
                    </div>
                </div>
                
                <div class="protocol-legend">
                    <div class="legend-item">
                        <span class="gate-cell">H</span>
                        <span>Hadamard Gate (Diagonal Basis)</span>
                    </div>
                    <div class="legend-item">
                        <span class="gate-cell">I</span>
                        <span>Identity (Computational Basis)</span>
                    </div>
                    <div class="legend-item">
                        <span class="measure-cell">M</span>
                        <span>Measurement</span>
                    </div>
                    <div class="legend-item">
                        <span class="eve-cell">EVE</span>
                        <span>Eavesdropper Interference</span>
                    </div>
                </div>
            </div>
        `;
        
        this.container.innerHTML = circuitHTML;
    }
    
    /**
     * Setup interactive circuit controls
     */
    setupCircuitControls() {
        const stepBtn = document.getElementById('step-btn');
        const simulateBtn = document.getElementById('simulate-btn');
        
        if (stepBtn) {
            stepBtn.addEventListener('click', () => this.stepThroughProtocol());
        }
        
        if (simulateBtn) {
            simulateBtn.addEventListener('click', () => this.simulateFullProtocol());
        }
    }
    
    /**
     * Step-by-step protocol demonstration
     */
    async stepThroughProtocol() {
        const steps = [
            { name: 'Alice prepares random bit and basis', delay: 1000 },
            { name: 'Alice encodes qubit', delay: 1000 },
            { name: 'Quantum transmission', delay: 1500 },
            { name: 'Bob selects measurement basis', delay: 1000 },
            { name: 'Bob measures qubit', delay: 1000 },
            { name: 'Basis comparison and key sifting', delay: 1500 }
        ];
        
        for (let i = 0; i < steps.length; i++) {
            await this.highlightStep(i + 1, steps[i]);
            await this.delay(steps[i].delay);
        }
    }
    
    /**
     * **ENHANCED: Simulate full BB84 protocol via API with error handling**
     */
    async simulateFullProtocol() {
        try {
            console.log('Starting BB84 protocol simulation...');
            
            const response = await fetch(`${this.apiBaseUrl}/bb84/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_bits: 10,
                    eavesdropper_present: document.getElementById('eve-toggle')?.checked || false,
                    noise_level: 0.05,
                    error_correction: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('BB84 simulation result:', result);
            
            if (result.success) {
                this.displayProtocolResults(result);
            } else {
                console.error('BB84 simulation failed:', result.error);
                this.displayProtocolError(result.error);
            }
            
        } catch (error) {
            console.error('Failed to simulate BB84 protocol:', error);
            this.displayProtocolError(error.message);
        }
    }
    
    /**
     * **ENHANCED: Display BB84 protocol simulation results**
     */
    displayProtocolResults(result) {
        let resultsContainer = document.getElementById('protocol-results');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'protocol-results';
            this.container.appendChild(resultsContainer);
        }
        
        const efficiency = result.protocol_efficiency || 0;
        const errorRate = result.error_rate || 0;
        const keyLength = result.final_key?.length || 0;
        const isSecure = result.security_analysis?.secure || false;
        
        resultsContainer.innerHTML = `
            <div class="protocol-results">
                <h4>🔐 BB84 Protocol Results</h4>
                <div class="result-metrics">
                    <div class="metric">
                        <span class="metric-label">Protocol Efficiency:</span>
                        <span class="metric-value">${(efficiency * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Error Rate (QBER):</span>
                        <span class="metric-value">${(errorRate * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Final Key Length:</span>
                        <span class="metric-value">${keyLength} bits</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Security Status:</span>
                        <span class="metric-value ${isSecure ? 'secure' : 'insecure'}">
                            ${isSecure ? '✅ SECURE' : '❌ COMPROMISED'}
                        </span>
                    </div>
                </div>
                
                <div class="key-display">
                    <h5>Generated Key (first 20 bits):</h5>
                    <code>${result.final_key?.slice(0, 20).join('') || 'N/A'}</code>
                </div>
                
                <div class="backend-info">
                    <small>Backend: ${result.simulation_backend || 'Unknown'}</small>
                </div>
            </div>
        `;
    }
    
    /**
     * **NEW: Display protocol error**
     */
    displayProtocolError(errorMessage) {
        let resultsContainer = document.getElementById('protocol-results');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'protocol-results';
            this.container.appendChild(resultsContainer);
        }
        
        resultsContainer.innerHTML = `
            <div class="protocol-error">
                <h4>❌ BB84 Protocol Simulation Failed</h4>
                <p>Error: ${errorMessage}</p>
                <button onclick="location.reload()">Retry</button>
            </div>
        `;
    }
    
    async highlightStep(stepNum, stepInfo) {
        const stepElement = document.getElementById(`step-${stepNum}`);
        if (stepElement) {
            stepElement.classList.add('active-step');
        }
        
        await this.updateStepVisualization(stepNum, stepInfo);
    }
    
    async updateStepVisualization(stepNum, stepInfo) {
        switch (stepNum) {
            case 1:
                const basisGate = document.getElementById('basis-gate');
                if (basisGate) {
                    basisGate.textContent = Math.random() > 0.5 ? 'H' : 'I';
                }
                break;
                
            case 3:
                const bobBasisGate = document.getElementById('bob-basis-gate');
                const measurementResult = document.getElementById('measurement-result');
                if (bobBasisGate && measurementResult) {
                    bobBasisGate.textContent = Math.random() > 0.5 ? 'H' : 'I';
                    measurementResult.textContent = Math.random() > 0.5 ? '1' : '0';
                }
                break;
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Enhanced State Vector Calculator (keeping existing functionality)
class StateVectorCalculator {
    static sphericalToCartesian(theta, phi) {
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);
        return { x, y, z };
    }
    
    static cartesianToSpherical(x, y, z) {
        const r = Math.sqrt(x * x + y * y + z * z);
        const theta = Math.acos(z / r);
        const phi = Math.atan2(y, x);
        return { r, theta, phi };
    }
    
    static sphericalToStateVector(theta, phi) {
        const alpha = Math.cos(theta / 2);
        const beta_real = Math.sin(theta / 2) * Math.cos(phi);
        const beta_imag = Math.sin(theta / 2) * Math.sin(phi);
        
        return {
            alpha: { real: alpha, imag: 0 },
            beta: { real: beta_real, imag: beta_imag }
        };
    }
    
    static calculateBlochCoordinates(stateVector) {
        const { alpha, beta } = stateVector;
        
        const a = { real: alpha.real, imag: alpha.imag };
        const b = { real: beta.real, imag: beta.imag };
        
        const x = 2 * (a.real * b.real + a.imag * b.imag);
        const y = 2 * (a.real * b.imag - a.imag * b.real);
        const z = a.real * a.real + a.imag * a.imag - b.real * b.real - b.imag * b.imag;
        
        return { x, y, z };
    }
    
    static calculateMeasurementProbabilities(stateVector) {
        const { alpha, beta } = stateVector;
        
        const prob_0 = alpha.real * alpha.real + alpha.imag * alpha.imag;
        const prob_1 = beta.real * beta.real + beta.imag * beta.imag;
        
        return { 
            'P(|0⟩)': prob_0, 
            'P(|1⟩)': prob_1 
        };
    }
}

// **ENHANCED: Initialization with comprehensive error handling**
document.addEventListener('DOMContentLoaded', function() {
    try {
        console.log('Initializing Enhanced Quantum Visualizer...');
        
        // Initialize quantum visualizer with error handling
        let visualizer = null;
        try {
            visualizer = new QuantumVisualizer();
            console.log('✅ QuantumVisualizer initialized successfully');
        } catch (visualizerError) {
            console.error('❌ Failed to initialize QuantumVisualizer:', visualizerError);
        }
        
        // Initialize circuit visualizer with error handling
        let circuitVisualizer = null;
        try {
            circuitVisualizer = new QuantumCircuitVisualizer('bb84-circuit');
            console.log('✅ QuantumCircuitVisualizer initialized successfully');
        } catch (circuitError) {
            console.error('❌ Failed to initialize QuantumCircuitVisualizer:', circuitError);
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            try {
                if (visualizer && visualizer.isInitialized) {
                    visualizer.setupCanvas();
                    visualizer.drawBlochSphere();
                }
            } catch (resizeError) {
                console.error('Resize error:', resizeError);
            }
        });
        
        // Global error handler for quantum operations
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.message && 
                (event.reason.message.includes('quantum') || 
                 event.reason.message.includes('visualization'))) {
                console.error('Unhandled quantum visualization error:', event.reason);
                if (visualizer) {
                    visualizer.handleVisualizationError(event.reason);
                }
                event.preventDefault();
            }
        });
        
        // Export for global access
        window.QuantumVisualizer = QuantumVisualizer;
        window.QuantumCircuitVisualizer = QuantumCircuitVisualizer;
        window.StateVectorCalculator = StateVectorCalculator;
        
        // Export instances for debugging
        window.quantumVisualizerInstance = visualizer;
        window.circuitVisualizerInstance = circuitVisualizer;
        
        console.log('🚀 Enhanced Quantum Visualizer system initialized successfully');
        
        // **DEBUG: Test API connectivity**
        if (visualizer) {
            setTimeout(() => {
                console.log('Testing API connectivity...');
                visualizer.loadQuantumStateFromAPI(0, 0)
                    .then(() => console.log('✅ API connectivity test passed'))
                    .catch(error => console.warn('⚠️ API connectivity test failed:', error));
            }, 1000);
        }
        
    } catch (error) {
        console.error('💥 Critical error during Quantum Visualizer initialization:', error);
        
        // Create fallback error display
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `
            <div style="background: #ff6b6b; color: white; padding: 20px; margin: 20px; border-radius: 10px;">
                <h3>🔧 Quantum Visualizer Initialization Failed</h3>
                <p>There was an error initializing the quantum visualization system.</p>
                <details>
                    <summary>Technical Details</summary>
                    <pre>${error.stack || error.message}</pre>
                </details>
                <button onclick="location.reload()">🔄 Reload Page</button>
            </div>
        `;
        document.body.appendChild(errorDiv);
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        QuantumVisualizer, 
        QuantumCircuitVisualizer, 
        StateVectorCalculator 
    };
}
