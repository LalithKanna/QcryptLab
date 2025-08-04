/**
 * Enhanced Bloch Sphere Visualization JavaScript with Backend Integration
 * =====================================================================
 * 
 * Provides interactive Bloch sphere visualizations for quantum states with:
 * - Full Flask backend API integration
 * - Error handling for undefined response properties
 * - Canvas-based local rendering as fallback
 * - Server-generated image display capability
 * - Defensive programming against network errors
 * - BB84 protocol integration
 */

class BlochSphereVisualizer {
    constructor(containerId, canvasId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById(canvasId);
        
        // **ENHANCED: Create canvas if it doesn't exist**
        if (!this.canvas && this.container) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = canvasId || 'bloch-sphere';
            this.canvas.width = 400;
            this.canvas.height = 400;
            this.container.appendChild(this.canvas);
        }
        
        if (!this.canvas) {
            console.error('Canvas not found and could not be created for Bloch sphere visualization');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.stateVector = [1, 0]; // Default |0⟩ state
        this.theta = 0;
        this.phi = 0;
        
        // **NEW: Backend integration properties**
        this.apiBaseUrl = '/api';
        this.loadingStates = new Set();
        this.currentApiState = null;
        this.useServerImages = true; // Prefer server-generated images
        
        // **NEW: Enhanced configuration**
        this.config = {
            apiTimeout: 15000,
            retryAttempts: 3,
            debugMode: false
        };
        
        this.init();
    }
    
    init() {
        // Set canvas size
        this.resize();
        
        // Add event listeners
        window.addEventListener('resize', () => this.resize());
        
        // **NEW: Setup error handling**
        this.setupErrorHandling();
        
        // Initial render
        this.render();
        
        // **NEW: Initialize with API call**
        this.loadQuantumStateFromAPI(0, 0);
    }
    
    /**
     * **NEW: Enhanced API integration for loading quantum states**
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
            
            // **FIXED: Update current state with API data**
            this.currentApiState = {
                theta: this.safeGetNumber(sphericalCoords, 'theta', theta),
                phi: this.safeGetNumber(sphericalCoords, 'phi', phi),
                x: this.safeGetNumber(blochCoords, 'x', Math.sin(theta) * Math.cos(phi)),
                y: this.safeGetNumber(blochCoords, 'y', Math.sin(theta) * Math.sin(phi)),
                z: this.safeGetNumber(blochCoords, 'z', Math.cos(theta)),
                amplitudes: amplitudes,
                probabilities: probabilities,
                apiData: data
            };
            
            // Update local state
            this.theta = this.currentApiState.theta;
            this.phi = this.currentApiState.phi;
            this.updateStateVector();

            // **FIXED: Display visualization with error handling**
            if (data.bloch_image && this.useServerImages) {
                await this.displayServerBlochSphere(data.bloch_image);
            } else {
                this.render();
            }
            
            // **NEW: Update state information panel**
            this.updateStateInfoPanel(stateAnalysis, bb84Context);
            
            this.log('State loaded successfully from API:', this.currentApiState);
            
        } catch (error) {
            console.error('Network error loading visualization:', error.message);
            this.handleVisualizationError(error);
            
            // **ENHANCED: Graceful fallback to local rendering**
            this.setQuantumState(theta, phi);
            
        } finally {
            this.removeLoadingState(loadingId);
        }
    }
    
    /**
     * **NEW: API response validation**
     */
    validateAPIResponse(data) {
        if (!data || typeof data !== 'object') {
            console.error('API response is not a valid object:', data);
            return false;
        }
        
        if (data.hasOwnProperty('success') && !data.success) {
            console.error('API request failed:', data.error || 'Unknown error');
            return false;
        }
        
        return true;
    }
    
    /**
     * **NEW: Safe property getter**
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
     * **NEW: Safe number getter**
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
     * **NEW: Fetch with retry mechanism**
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
            
            if (error.name === 'AbortError') {
                throw new Error(`Request timeout (${this.config.apiTimeout}ms) for ${url}`);
            }
            throw error;
        }
    }
    
    /**
     * **NEW: Display server-generated Bloch sphere**
     */
    async displayServerBlochSphere(base64Image) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => {
                try {
                    // Clear canvas and draw server image
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    
                    // Calculate scaling to fit canvas
                    const scale = Math.min(
                        this.canvas.width / img.width,
                        this.canvas.height / img.height
                    ) * 0.9;
                    
                    const x = (this.canvas.width - img.width * scale) / 2;
                    const y = (this.canvas.height - img.height * scale) / 2;
                    
                    this.ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                    
                    // **NEW: Add server indicator**
                    this.drawServerIndicator();
                    
                    resolve();
                    
                } catch (drawError) {
                    console.error('Failed to draw server image:', drawError);
                    reject(drawError);
                }
            };
            
            img.onerror = (error) => {
                console.warn('Failed to load server Bloch sphere image, using local rendering');
                this.render();
                reject(error);
            };
            
            if (!base64Image || typeof base64Image !== 'string') {
                console.error('Invalid base64 image data');
                this.render();
                reject(new Error('Invalid base64 image data'));
                return;
            }
            
            img.src = `data:image/png;base64,${base64Image}`;
        });
    }
    
    /**
     * **NEW: Draw server indicator**
     */
    drawServerIndicator() {
        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        this.ctx.fillRect(10, 10, 100, 20);
        this.ctx.fillStyle = 'black';
        this.ctx.font = '12px Arial';
        this.ctx.fillText('Server Generated', 15, 22);
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        // Re-render after resize
        if (this.currentApiState && this.currentApiState.apiData && this.currentApiState.apiData.bloch_image) {
            this.displayServerBlochSphere(this.currentApiState.apiData.bloch_image).catch(() => {
                this.render();
            });
        } else {
            this.render();
        }
    }
    
    /**
     * **ENHANCED: Set quantum state with API integration**
     */
    async setQuantumState(theta, phi) {
        this.theta = theta;
        this.phi = phi;
        this.updateStateVector();
        
        // **NEW: Try to load from API first**
        try {
            await this.loadQuantumStateFromAPI(theta, phi);
        } catch (error) {
            console.warn('API call failed, using local rendering:', error);
            this.render();
        }
        
        return this.getStateInfo();
    }
    
    /**
     * **ENHANCED: Set state vector with API integration**
     */
    async setStateVector(alpha, beta) {
        this.stateVector = [alpha, beta];
        
        // Calculate spherical coordinates
        const r = Math.sqrt(Math.abs(alpha) ** 2 + this.getMagnitude(beta) ** 2);
        if (r > 0) {
            this.theta = 2 * Math.acos(Math.abs(alpha) / r);
            this.phi = this.getPhase(beta);
        }
        
        // **NEW: Try to load from API**
        try {
            await this.loadQuantumStateFromAPI(this.theta, this.phi);
        } catch (error) {
            console.warn('API call failed, using local rendering:', error);
            this.render();
        }
        
        return this.getStateInfo();
    }
    
    /**
     * Update state vector from spherical coordinates
     */
    updateStateVector() {
        const alpha = Math.cos(this.theta / 2);
        const betaReal = Math.sin(this.theta / 2) * Math.cos(this.phi);
        const betaImag = Math.sin(this.theta / 2) * Math.sin(this.phi);
        
        this.stateVector = [alpha, this.complex(betaReal, betaImag)];
    }
    
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.35;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // **ENHANCED: Draw background**
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw Bloch sphere components
        this.drawBlochSphere(centerX, centerY, radius);
        this.drawCoordinateAxes(centerX, centerY, radius);
        this.drawQuantumState(centerX, centerY, radius);
        this.drawLabels(centerX, centerY, radius);
        
        // **NEW: Draw local rendering indicator**
        this.drawLocalIndicator();
    }
    
    /**
     * **NEW: Draw local rendering indicator**
     */
    drawLocalIndicator() {
        this.ctx.fillStyle = 'rgba(255, 165, 0, 0.8)';
        this.ctx.fillRect(10, 10, 90, 20);
        this.ctx.fillStyle = 'black';
        this.ctx.font = '12px Arial';
        this.ctx.fillText('Local Render', 15, 22);
    }
    
    drawBlochSphere(centerX, centerY, radius) {
        const ctx = this.ctx;
        
        // **ENHANCED: Draw sphere with gradient**
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, 'rgba(74, 144, 226, 0.1)');
        gradient.addColorStop(1, 'rgba(74, 144, 226, 0.05)');
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.strokeStyle = '#4a90e2';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // **ENHANCED: Draw equator and meridians**
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        
        // Equator
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        
        // Meridians
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI) / 3;
            const ellipseRadius = radius * Math.abs(Math.cos(angle));
            if (ellipseRadius > 10) {
                ctx.beginPath();
                ctx.ellipse(centerX, centerY, ellipseRadius, radius, angle, 0, 2 * Math.PI);
                ctx.stroke();
            }
        }
        ctx.setLineDash([]);
    }
    
    drawCoordinateAxes(centerX, centerY, radius) {
        const ctx = this.ctx;
        const axisLength = radius * 1.3;
        
        // X-axis (red)
        ctx.beginPath();
        ctx.moveTo(centerX - axisLength, centerY);
        ctx.lineTo(centerX + axisLength, centerY);
        ctx.strokeStyle = '#ff4444';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Y-axis (green) - vertical line for 2D representation
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - axisLength);
        ctx.lineTo(centerX, centerY + axisLength);
        ctx.strokeStyle = '#44ff44';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Z-axis indicators (blue)
        ctx.fillStyle = '#4444ff';
        ctx.beginPath();
        ctx.arc(centerX, centerY - radius * 0.9, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.fillStyle = '#9944ff';
        ctx.beginPath();
        ctx.arc(centerX, centerY + radius * 0.9, 6, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    drawQuantumState(centerX, centerY, radius) {
        const ctx = this.ctx;
        
        // **ENHANCED: Use API coordinates if available**
        let stateX, stateY;
        
        if (this.currentApiState && this.currentApiState.x !== undefined && this.currentApiState.z !== undefined) {
            stateX = centerX + radius * this.currentApiState.x;
            stateY = centerY - radius * this.currentApiState.z; // Z maps to vertical
        } else {
            // Fallback to local calculation
            const x = Math.sin(this.theta) * Math.cos(this.phi);
            const z = Math.cos(this.theta);
            stateX = centerX + radius * x;
            stateY = centerY - radius * z;
        }
        
        // **ENHANCED: Draw state vector with gradient**
        const gradient = ctx.createLinearGradient(centerX, centerY, stateX, stateY);
        gradient.addColorStop(0, '#ffaa00');
        gradient.addColorStop(1, '#ff6600');
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(stateX, stateY);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.stroke();
        
        // **ENHANCED: Draw arrowhead**
        const angle = Math.atan2(stateY - centerY, stateX - centerX);
        const arrowLength = 15;
        const arrowAngle = Math.PI / 6;
        
        ctx.beginPath();
        ctx.moveTo(stateX, stateY);
        ctx.lineTo(
            stateX - arrowLength * Math.cos(angle - arrowAngle),
            stateY - arrowLength * Math.sin(angle - arrowAngle)
        );
        ctx.moveTo(stateX, stateY);
        ctx.lineTo(
            stateX - arrowLength * Math.cos(angle + arrowAngle),
            stateY - arrowLength * Math.sin(angle + arrowAngle)
        );
        ctx.strokeStyle = '#ff6600';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // **ENHANCED: Draw state point**
        ctx.beginPath();
        ctx.arc(stateX, stateY, 10, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffaa00';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // **ENHANCED: Draw state label with probabilities**
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('|ψ⟩', stateX, stateY - 20);
        
        // **NEW: Show measurement probabilities if available**
        if (this.currentApiState && this.currentApiState.probabilities) {
            ctx.font = '12px Arial';
            ctx.fillText(
                `P(0)=${this.currentApiState.probabilities['P(|0⟩)'].toFixed(3)}`,
                stateX, stateY - 35
            );
        }
    }
    
    drawLabels(centerX, centerY, radius) {
        const ctx = this.ctx;
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        
        // **ENHANCED: Quantum state labels**
        ctx.fillText('|0⟩', centerX, centerY - radius - 25);
        ctx.fillText('|1⟩', centerX, centerY + radius + 35);
        ctx.fillText('|+⟩', centerX + radius + 25, centerY + 5);
        ctx.fillText('|−⟩', centerX - radius - 25, centerY + 5);
        
        // **ENHANCED: Coordinate labels**
        ctx.font = '14px Arial';
        ctx.fillStyle = '#ff4444';
        ctx.fillText('X', centerX + radius * 1.4, centerY + 5);
        ctx.fillStyle = '#44ff44';
        ctx.fillText('Y', centerX + 5, centerY - radius * 1.4);
        ctx.fillStyle = '#4444ff';
        ctx.fillText('Z', centerX + 15, centerY - radius * 1.2);
    }
    
    /**
     * **NEW: Update state information panel**
     */
    updateStateInfoPanel(stateAnalysis, bb84Context) {
        const infoPanel = document.getElementById('state-info-panel');
        if (!infoPanel || !stateAnalysis) return;
        
        const amplitudes = this.safeGetProperty(stateAnalysis, 'amplitudes', null);
        const probabilities = this.safeGetProperty(stateAnalysis, 'probabilities', null);
        
        let infoHTML = '<div class="state-info">';
        
        if (amplitudes && amplitudes.alpha && amplitudes.beta) {
            infoHTML += `
                <div class="info-section">
                    <h4>State Amplitudes</h4>
                    <p>α = ${amplitudes.alpha.real.toFixed(3)} + ${amplitudes.alpha.imag.toFixed(3)}i</p>
                    <p>β = ${amplitudes.beta.real.toFixed(3)} + ${amplitudes.beta.imag.toFixed(3)}i</p>
                </div>
            `;
        }
        
        if (probabilities) {
            const p0 = this.safeGetNumber(probabilities, 'P(|0⟩)', 0);
            const p1 = this.safeGetNumber(probabilities, 'P(|1⟩)', 0);
            infoHTML += `
                <div class="info-section">
                    <h4>Measurement Probabilities</h4>
                    <p>P(|0⟩) = ${p0.toFixed(3)} (${(p0*100).toFixed(1)}%)</p>
                    <p>P(|1⟩) = ${p1.toFixed(3)} (${(p1*100).toFixed(1)}%)</p>
                </div>
            `;
        }
        
        infoHTML += '</div>';
        infoPanel.innerHTML = infoHTML;
    }
    
    /**
     * **NEW: Error handling setup**
     */
    setupErrorHandling() {
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.message && 
                event.reason.message.includes('visualization')) {
                this.handleVisualizationError(event.reason);
                event.preventDefault();
            }
        });
    }
    
    /**
     * **NEW: Handle visualization errors**
     */
    handleVisualizationError(error) {
        console.error('Quantum visualization error:', error);
        
        // Show user-friendly error message
        this.showErrorMessage(`Visualization error: ${error.message}. Using local rendering.`);
        
        // Force local rendering
        this.useServerImages = false;
        this.render();
        
        // Re-enable server images after a delay
        setTimeout(() => {
            this.useServerImages = true;
        }, 30000);
    }
    
    /**
     * **NEW: Show error message**
     */
    showErrorMessage(message) {
        let errorDiv = document.getElementById('bloch-error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'bloch-error';
            errorDiv.style.cssText = `
                background: #ff6b6b;
                color: white;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                display: none;
            `;
            if (this.container) {
                this.container.appendChild(errorDiv);
            }
        }
        
        errorDiv.innerHTML = `
            <p>${message}</p>
            <button onclick="this.parentElement.style.display='none'">Dismiss</button>
        `;
        errorDiv.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 10000);
    }
    
    /**
     * **NEW: Loading state management**
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
        let indicator = document.getElementById('bloch-loading');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'bloch-loading';
            indicator.innerHTML = '<p>Loading quantum state...</p>';
            indicator.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 15px;
                border-radius: 10px;
                z-index: 1000;
            `;
            if (this.container) {
                this.container.style.position = 'relative';
                this.container.appendChild(indicator);
            }
        }
        indicator.style.display = 'block';
    }
    
    hideLoadingIndicator() {
        const indicator = document.getElementById('bloch-loading');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }
    
    // **ENHANCED: Utility functions**
    complex(real, imag) {
        return { real: real || 0, imag: imag || 0 };
    }
    
    getMagnitude(complexNum) {
        if (typeof complexNum === 'number') return Math.abs(complexNum);
        if (complexNum && typeof complexNum === 'object') {
            return Math.sqrt((complexNum.real || 0) ** 2 + (complexNum.imag || 0) ** 2);
        }
        return 0;
    }
    
    getPhase(complexNum) {
        if (typeof complexNum === 'number') return complexNum >= 0 ? 0 : Math.PI;
        if (complexNum && typeof complexNum === 'object') {
            return Math.atan2(complexNum.imag || 0, complexNum.real || 0);
        }
        return 0;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    log(...args) {
        if (this.config.debugMode) {
            console.log('[BlochSphereVisualizer]', ...args);
        }
    }
    
    /**
     * **ENHANCED: Get current state information**
     */
    getStateInfo() {
        const alpha = this.stateVector[0];
        const beta = this.stateVector[1];
        const betaMagnitude = this.getMagnitude(beta);
        
        let stateInfo = {
            stateVector: [alpha, beta],
            theta: this.theta,
            phi: this.phi,
            probabilities: {
                zero: Math.abs(alpha) ** 2,
                one: betaMagnitude ** 2
            },
            sphericalCoords: {
                theta: this.theta * 180 / Math.PI,
                phi: this.phi * 180 / Math.PI
            }
        };
        
        // **NEW: Include API state if available**
        if (this.currentApiState) {
            stateInfo.apiState = this.currentApiState;
            stateInfo.enhanced = true;
        }
        
        return stateInfo;
    }
    
    /**
     * **NEW: Toggle between server and local rendering**
     */
    toggleRenderingMode() {
        this.useServerImages = !this.useServerImages;
        console.log(`Switched to ${this.useServerImages ? 'server' : 'local'} rendering mode`);
        
        if (this.useServerImages && this.currentApiState) {
            this.loadQuantumStateFromAPI(this.theta, this.phi);
        } else {
            this.render();
        }
    }
    
    /**
     * **NEW: Cleanup method**
     */
    destroy() {
        // Remove event listeners
        window.removeEventListener('resize', () => this.resize());
        
        // Clear loading states
        this.loadingStates.clear();
        
        // Remove error and loading indicators
        const errorDiv = document.getElementById('bloch-error');
        const loadingDiv = document.getElementById('bloch-loading');
        if (errorDiv) errorDiv.remove();
        if (loadingDiv) loadingDiv.remove();
    }
}

// **ENHANCED: Global functions for easy access**
function initializeBlochSphere(containerId = 'bloch-sphere-container', canvasId = 'bloch-sphere') {
    console.log('Initializing enhanced Bloch sphere visualizer...');
    return new BlochSphereVisualizer(containerId, canvasId);
}

async function updateBlochVisualization(theta, phi, visualizer) {
    if (visualizer) {
        try {
            return await visualizer.setQuantumState(theta, phi);
        } catch (error) {
            console.error('Failed to update Bloch visualization:', error);
            return null;
        }
    }
    return null;
}

/**
 * **NEW: Create visualizer with enhanced options**
 */
function createEnhancedBlochVisualizer(options = {}) {
    const {
        containerId = 'bloch-sphere-container',
        canvasId = 'bloch-sphere',
        useServerImages = true,
        debugMode = false
    } = options;
    
    const visualizer = new BlochSphereVisualizer(containerId, canvasId);
    visualizer.useServerImages = useServerImages;
    visualizer.config.debugMode = debugMode;
    
    return visualizer;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        BlochSphereVisualizer,
        initializeBlochSphere,
        updateBlochVisualization,
        createEnhancedBlochVisualizer
    };
}

// **NEW: Auto-initialization if DOM is ready**
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        // Auto-initialize if elements exist
        const container = document.getElementById('bloch-sphere-container');
        const canvas = document.getElementById('bloch-sphere');
        
        if (container && canvas) {
            console.log('Auto-initializing Bloch sphere visualizer...');
            window.blochVisualizerInstance = initializeBlochSphere();
        }
    });
}
