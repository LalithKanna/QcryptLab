/**
 * BB84 Quantum Key Distribution Simulator
 * =======================================
 * 
 * Advanced interactive simulation of the BB84 protocol featuring:
 * - Real-time protocol visualization with quantum state animations
 * - Step-by-step educational simulation mode
 * - Advanced eavesdropping detection and security analysis
 * - Comprehensive error correction and privacy amplification
 * - Interactive Bloch sphere and quantum circuit visualizations
 * - Performance metrics and efficiency analysis
 * 
 * @author QuantumCrypto Tutorial Team
 * @version 2.0.0
 */

class BB84Simulator {
    constructor() {
        this.simulationState = {
            numBits: 100,
            eavesdropper: false,
            noiseLevel: 0.05,
            animationSpeed: 1.0,
            errorCorrection: true,
            privacyAmplification: true,
            currentStep: 0,
            isRunning: false,
            stepByStepMode: false,
            pauseBetweenSteps: 1000
        };
        
        this.protocolData = {
            aliceBits: [],
            aliceBases: [],
            aliceStates: [],
            bobBases: [],
            bobMeasurements: [],
            eveInterceptions: [],
            siftedBits: [],
            testBits: [],
            errorRate: 0,
            finalKey: [],
            securityAnalysis: {},
            performanceMetrics: {},
            transmissionLog: []
        };
        
        this.visualizers = {};
        this.animationQueue = [];
        this.notificationSystem = null;
        
        this.init();
    }
    
    /**
     * Initialize the simulator with all necessary components
     */
    init() {
        this.setupEventListeners();
        this.initializeVisualizers();
        this.initializeNotificationSystem();
        this.updateDisplays();
        this.preloadAnimations();
        
        console.log('BB84 Simulator initialized successfully');
    }
    
    /**
     * Setup all event listeners for interactive controls
     */
    setupEventListeners() {
        // Simulation parameter controls
        const controls = [
            { id: 'num-bits', prop: 'numBits', type: 'int', callback: this.validateBitCount.bind(this) },
            { id: 'eavesdropper', prop: 'eavesdropper', type: 'bool', callback: this.updateEveVisualization.bind(this) },
            { id: 'noise-level', prop: 'noiseLevel', type: 'float', callback: this.updateNoiseDisplay.bind(this) },
            { id: 'animation-speed', prop: 'animationSpeed', type: 'float', callback: this.updateSpeedDisplay.bind(this) },
            { id: 'error-correction', prop: 'errorCorrection', type: 'bool' },
            { id: 'privacy-amplification', prop: 'privacyAmplification', type: 'bool' }
        ];
        
        controls.forEach(control => {
            const element = document.getElementById(control.id);
            if (element) {
                const eventType = control.type === 'bool' ? 'change' : 'input';
                element.addEventListener(eventType, (e) => {
                    let value = e.target.value;
                    if (control.type === 'int') value = parseInt(value);
                    else if (control.type === 'float') value = parseFloat(value);
                    else if (control.type === 'bool') value = e.target.checked;
                    
                    this.simulationState[control.prop] = value;
                    if (control.callback) control.callback(value);
                });
            }
        });
        
        // Bloch sphere state selector
        const stateSelector = document.getElementById('state-selector');
        if (stateSelector) {
            stateSelector.addEventListener('change', (e) => {
                this.updateBlochSphereState(e.target.value);
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    }
    
    /**
     * Handle keyboard shortcuts for simulator control
     */
    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'Enter':
                    event.preventDefault();
                    this.runBB84Simulation();
                    break;
                case 's':
                    event.preventDefault();
                    this.runStepByStep();
                    break;
                case 'r':
                    event.preventDefault();
                    this.resetSimulation();
                    break;
            }
        }
    }
    
    /**
     * Initialize visualization components
     */
    initializeVisualizers() {
        try {
            // Initialize Bloch sphere visualizer
            if (window.BlochSphereVisualizer) {
                this.visualizers.bloch = new BlochSphereVisualizer('bloch-sphere-container', 'bloch-canvas');
                this.updateBlochSphereState('0'); // Default to |0⟩ state
            }
            
            // Initialize quantum circuit visualizer
            if (window.QuantumCircuitVisualizer) {
                this.visualizers.circuit = new QuantumCircuitVisualizer('bb84-circuit');
            }
            
            // Initialize particle animation system
            this.visualizers.particles = new QuantumParticleSystem('qubit-stream');
            
            console.log('Visualizers initialized successfully');
        } catch (error) {
            console.warn('Some visualizers failed to initialize:', error);
        }
    }
    
    /**
     * Initialize notification system for user feedback
     */
    initializeNotificationSystem() {
        this.notificationSystem = {
            show: (message, type = 'info', duration = 3000) => {
                this.showNotification(message, type, duration);
            }
        };
    }
    
    /**
     * Validate bit count and provide feedback
     */
    validateBitCount(value) {
        const min = 10, max = 1000;
        if (value < min || value > max) {
            this.showNotification(`Bit count must be between ${min} and ${max}`, 'warning');
            return false;
        }
        return true;
    }
    
    /**
     * Update noise level display
     */
    updateNoiseDisplay() {
        const noiseValue = document.getElementById('noise-value');
        if (noiseValue) {
            const percentage = (this.simulationState.noiseLevel * 100).toFixed(1);
            noiseValue.textContent = percentage + '%';
        }
    }
    
    /**
     * Update animation speed display
     */
    updateSpeedDisplay() {
        const speedValue = document.getElementById('speed-value');
        if (speedValue) {
            speedValue.textContent = this.simulationState.animationSpeed.toFixed(1) + 'x';
        }
        
        // Update animation timings
        this.updateAnimationTimings();
    }
    
    /**
     * Update Eve visualization based on eavesdropper setting
     */
    updateEveVisualization() {
        const eveInterception = document.getElementById('eve-interception');
        if (eveInterception) {
            if (this.simulationState.eavesdropper) {
                eveInterception.style.display = 'block';
                eveInterception.classList.add('eve-active');
                this.showNotification('Eve (Eavesdropper) enabled - Security will be compromised!', 'warning');
            } else {
                eveInterception.style.display = 'none';
                eveInterception.classList.remove('eve-active');
            }
        }
    }
    
    /**
     * Update all display elements
     */
    updateDisplays() {
        this.updateNoiseDisplay();
        this.updateSpeedDisplay();
        this.updateEveVisualization();
    }
    
    /**
     * Main BB84 simulation runner with comprehensive error handling
     */
    async runBB84Simulation() {
        if (this.simulationState.isRunning) {
            this.showNotification('Simulation already in progress', 'warning');
            return;
        }
        
        if (!this.validateBitCount(this.simulationState.numBits)) {
            return;
        }
        
        this.simulationState.isRunning = true;
        this.simulationState.stepByStepMode = false;
        this.resetSimulation();
        this.showLoadingState();
        
        try {
            this.showNotification('Starting BB84 simulation...', 'info');
            
            // Try API first, fallback to local simulation
            let result;
            try {
                result = await this.callSimulationAPI();
            } catch (apiError) {
                console.warn('API call failed, using local simulation:', apiError);
                result = await this.runLocalSimulation();
            }
            
            if (result) {
                await this.processAndDisplayResults(result);
                this.showNotification('BB84 simulation completed successfully!', 'success');
            } else {
                throw new Error('Simulation failed to produce results');
            }
            
        } catch (error) {
            console.error('Simulation error:', error);
            this.showError(`Simulation failed: ${error.message}`);
            this.showNotification('Simulation failed. Please try again.', 'error');
        } finally {
            this.simulationState.isRunning = false;
            this.hideLoadingState();
        }
    }
    
    /**
     * Call external simulation API
     */
    async callSimulationAPI() {
        const response = await fetch('/api/bb84/simulate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_bits: this.simulationState.numBits,
                eavesdropper: this.simulationState.eavesdropper,
                noise_level: this.simulationState.noiseLevel,
                error_correction: this.simulationState.errorCorrection,
                privacy_amplification: this.simulationState.privacyAmplification
            }),
            timeout: 30000 // 30 second timeout
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Run complete local simulation as fallback
     */
    async runLocalSimulation() {
        const startTime = performance.now();
        
        // Step 1: Alice's preparation
        this.updateSimulationProgress('Preparing quantum states...', 10);
        await this.simulateAlicePreparation();
        
        // Step 2: Quantum transmission
        this.updateSimulationProgress('Transmitting qubits...', 30);
        await this.simulateQuantumTransmission();
        
        // Step 3: Bob's measurements
        this.updateSimulationProgress('Performing measurements...', 50);
        await this.simulateBobMeasurement();
        
        // Step 4: Key sifting
        this.updateSimulationProgress('Sifting key bits...', 70);
        await this.simulateKeySifting();
        
        // Step 5: Error estimation and correction
        this.updateSimulationProgress('Analyzing security...', 85);
        await this.simulateSecurityAnalysis();
        
        // Step 6: Final key generation
        this.updateSimulationProgress('Generating final key...', 95);
        await this.generateFinalKey();
        
        const endTime = performance.now();
        const simulationTime = endTime - startTime;
        
        // Compile results
        return this.compileSimulationResults(simulationTime);
    }
    
    /**
     * Run step-by-step simulation for educational purposes
     */
    async runStepByStep() {
        if (this.simulationState.isRunning) {
            this.showNotification('Simulation already in progress', 'warning');
            return;
        }
        
        this.simulationState.isRunning = true;
        this.simulationState.stepByStepMode = true;
        this.resetSimulation();
        
        try {
            this.showNotification('Starting step-by-step BB84 simulation...', 'info');
            
            const steps = [
                { name: 'Alice\'s Preparation', method: this.simulateAlicePreparation.bind(this) },
                { name: 'Quantum Transmission', method: this.simulateQuantumTransmission.bind(this) },
                { name: 'Bob\'s Measurement', method: this.simulateBobMeasurement.bind(this) },
                { name: 'Key Sifting', method: this.simulateKeySifting.bind(this) },
                { name: 'Error Estimation', method: this.simulateErrorEstimation.bind(this) },
                { name: 'Error Correction', method: this.simulateErrorCorrection.bind(this) },
                { name: 'Privacy Amplification', method: this.simulatePrivacyAmplification.bind(this) }
            ];
            
            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];
                this.highlightCurrentStep(i);
                this.showNotification(`Step ${i + 1}: ${step.name}`, 'info');
                
                await step.method();
                await this.delay(this.simulationState.pauseBetweenSteps / this.simulationState.animationSpeed);
                
                this.completeStep(i);
            }
            
            // Final results compilation
            const result = this.compileSimulationResults();
            await this.displayResults(result);
            
            this.showNotification('Step-by-step simulation completed!', 'success');
            
        } catch (error) {
            console.error('Step-by-step simulation error:', error);
            this.showError(`Step-by-step simulation failed: ${error.message}`);
        } finally {
            this.simulationState.isRunning = false;
        }
    }
    
    /**
     * Simulate Alice's quantum state preparation
     */
    async simulateAlicePreparation() {
        this.updateStepContent('sifting-content', 'Alice is generating random bits and selecting measurement bases...');
        this.updateStepProgress('sifting-step', 0);
        
        // Generate random bits
        this.protocolData.aliceBits = this.generateRandomBits(this.simulationState.numBits);
        
        // Generate random bases
        this.protocolData.aliceBases = this.generateRandomBases(this.simulationState.numBits);
        
        // Prepare quantum states
        this.protocolData.aliceStates = this.prepareQuantumStates(
            this.protocolData.aliceBits,
            this.protocolData.aliceBases
        );
        
        // Update visualizations with animation
        await this.animateAlicePreparation();
        
        this.updateStepProgress('sifting-step', 100);
        this.updateStepContent('sifting-content', 
            `Alice prepared ${this.simulationState.numBits} quantum states: ` +
            `${this.protocolData.aliceBases.filter(b => b === 'rectilinear').length} rectilinear, ` +
            `${this.protocolData.aliceBases.filter(b => b === 'diagonal').length} diagonal`
        );
    }
    
    /**
     * Simulate quantum transmission through the channel
     */
    async simulateQuantumTransmission() {
        this.updateStepContent('error-estimation-content', 'Transmitting qubits through quantum channel...');
        this.updateStepProgress('error-estimation-step', 0);
        
        // Simulate transmission with potential interference
        const transmissionResults = [];
        
        for (let i = 0; i < this.protocolData.aliceStates.length; i++) {
            let state = { ...this.protocolData.aliceStates[i] };
            
            // Apply noise if enabled
            if (this.simulationState.noiseLevel > 0) {
                state = this.applyChannelNoise(state);
            }
            
            // Apply Eve's interference if enabled
            if (this.simulationState.eavesdropper) {
                const eveResult = this.simulateEveInterference(state, i);
                state = eveResult.modifiedState;
                this.protocolData.eveInterceptions.push(eveResult.interceptionData);
            }
            
            transmissionResults.push(state);
            
            // Update progress and visualization
            if (i % 10 === 0 || i === this.protocolData.aliceStates.length - 1) {
                const progress = ((i + 1) / this.protocolData.aliceStates.length) * 100;
                this.updateStepProgress('error-estimation-step', progress);
                await this.animateQubitTransmission(i, state);
            }
        }
        
        this.protocolData.transmittedStates = transmissionResults;
        this.protocolData.transmissionLog = this.generateTransmissionLog();
        
        this.updateStepContent('error-estimation-content', 
            `Transmitted ${this.simulationState.numBits} qubits. ` +
            `${this.simulationState.eavesdropper ? 'Eve intercepted and re-transmitted all qubits.' : 'Channel secure.'} ` +
            `Noise level: ${(this.simulationState.noiseLevel * 100).toFixed(1)}%`
        );
    }
    
    /**
     * Simulate Bob's quantum measurements
     */
    async simulateBobMeasurement() {
        this.updateStepContent('error-correction-content', 'Bob is randomly selecting measurement bases and measuring qubits...');
        this.updateStepProgress('error-correction-step', 0);
        
        // Bob generates random measurement bases
        this.protocolData.bobBases = this.generateRandomBases(this.simulationState.numBits);
        
        // Bob performs measurements
        this.protocolData.bobMeasurements = [];
        
        for (let i = 0; i < this.protocolData.transmittedStates.length; i++) {
            const state = this.protocolData.transmittedStates[i];
            const basis = this.protocolData.bobBases[i];
            
            // Perform quantum measurement
            const measurement = this.performQuantumMeasurement(state, basis);
            this.protocolData.bobMeasurements.push(measurement);
            
            // Update progress
            if (i % 10 === 0 || i === this.protocolData.transmittedStates.length - 1) {
                const progress = ((i + 1) / this.protocolData.transmittedStates.length) * 100;
                this.updateStepProgress('error-correction-step', progress);
                await this.animateBobMeasurement(i, measurement);
            }
        }
        
        // Calculate basis matching statistics
        const matchingBases = this.protocolData.aliceBases.filter(
            (basis, i) => basis === this.protocolData.bobBases[i]
        ).length;
        
        this.updateStepContent('error-correction-content',
            `Bob completed measurements using randomly chosen bases. ` +
            `${matchingBases}/${this.simulationState.numBits} (${(matchingBases/this.simulationState.numBits*100).toFixed(1)}%) bases matched Alice's choices.`
        );
    }
    
    /**
     * Simulate key sifting process
     */
    async simulateKeySifting() {
        this.updateStepContent('privacy-amplification-content', 'Comparing measurement bases and sifting matching bits...');
        this.updateStepProgress('privacy-amplification-step', 0);
        
        this.protocolData.siftedBits = [];
        const siftedIndices = [];
        
        for (let i = 0; i < this.simulationState.numBits; i++) {
            if (this.protocolData.aliceBases[i] === this.protocolData.bobBases[i]) {
                this.protocolData.siftedBits.push({
                    aliceBit: this.protocolData.aliceBits[i],
                    bobBit: this.protocolData.bobMeasurements[i],
                    index: i
                });
                siftedIndices.push(i);
            }
            
            // Update progress
            const progress = ((i + 1) / this.simulationState.numBits) * 100;
            this.updateStepProgress('privacy-amplification-step', progress);
        }
        
        await this.animateKeySifting(siftedIndices);
        
        this.updateStepContent('privacy-amplification-content',
            `Key sifting completed. Kept ${this.protocolData.siftedBits.length} bits where bases matched. ` +
            `Efficiency: ${(this.protocolData.siftedBits.length/this.simulationState.numBits*100).toFixed(1)}%`
        );
    }
    
    /**
     * Simulate error estimation process
     */
    async simulateErrorEstimation() {
        // Sample bits for error estimation
        const sampleSize = Math.min(
            Math.max(Math.floor(this.protocolData.siftedBits.length * 0.1), 1),
            this.protocolData.siftedBits.length
        );
        
        this.protocolData.testBits = this.sampleRandomBits(this.protocolData.siftedBits, sampleSize);
        
        // Calculate error rate
        let errors = 0;
        this.protocolData.testBits.forEach(bit => {
            if (bit.aliceBit !== bit.bobBit) {
                errors++;
            }
        });
        
        this.protocolData.errorRate = this.protocolData.testBits.length > 0 ? 
            errors / this.protocolData.testBits.length : 0;
        
        // Remove test bits from sifted bits
        const testIndices = new Set(this.protocolData.testBits.map(bit => bit.index));
        this.protocolData.siftedBits = this.protocolData.siftedBits.filter(
            bit => !testIndices.has(bit.index)
        );
    }
    
    /**
     * Simulate error correction process
     */
    async simulateErrorCorrection() {
        if (!this.simulationState.errorCorrection) {
            return;
        }
        
        // Simplified CASCADE error correction simulation
        const errorCorrectionEfficiency = 0.85; // Typical efficiency
        const informationLeak = Math.ceil(this.protocolData.errorRate * this.protocolData.siftedBits.length * 1.2);
        
        // Remove bits due to error correction overhead
        const correctedLength = Math.max(
            0,
            Math.floor(this.protocolData.siftedBits.length * errorCorrectionEfficiency) - informationLeak
        );
        
        this.protocolData.siftedBits = this.protocolData.siftedBits.slice(0, correctedLength);
    }
    
    /**
     * Simulate privacy amplification process
     */
    async simulatePrivacyAmplification() {
        if (!this.simulationState.privacyAmplification) {
            return;
        }
        
        // Privacy amplification reduces key length to eliminate Eve's information
        const eveInformation = this.calculateEveInformation();
        const amplificationRatio = Math.max(0.5, 1 - eveInformation);
        
        const amplifiedLength = Math.floor(this.protocolData.siftedBits.length * amplificationRatio);
        
        // Apply universal hashing (simplified)
        this.protocolData.amplifiedBits = this.applyUniversalHashing(
            this.protocolData.siftedBits,
            amplifiedLength
        );
    }
    
    /**
     * Perform comprehensive security analysis
     */
    async simulateSecurityAnalysis() {
        const securityThreshold = 0.11; // Standard BB84 threshold
        const isSecure = this.protocolData.errorRate < securityThreshold;
        
        this.protocolData.securityAnalysis = {
            errorRate: this.protocolData.errorRate,
            securityThreshold: securityThreshold,
            isSecure: isSecure,
            eavesdroppingDetected: this.simulationState.eavesdropper && this.protocolData.errorRate > securityThreshold,
            securityLevel: this.determineSecurityLevel(this.protocolData.errorRate),
            confidence: Math.max(0, 1 - (this.protocolData.errorRate / securityThreshold)),
            eveInformation: this.calculateEveInformation(),
            recommendations: this.generateSecurityRecommendations()
        };
    }
    
    /**
     * Generate final cryptographic key
     */
    async generateFinalKey() {
        if (this.simulationState.privacyAmplification && this.protocolData.amplifiedBits) {
            this.protocolData.finalKey = this.protocolData.amplifiedBits.map(bit => bit.bobBit);
        } else {
            this.protocolData.finalKey = this.protocolData.siftedBits.map(bit => bit.bobBit);
        }
        
        // Calculate performance metrics
        this.protocolData.performanceMetrics = {
            totalBits: this.simulationState.numBits,
            siftedBits: this.protocolData.siftedBits.length,
            finalKeyLength: this.protocolData.finalKey.length,
            siftingEfficiency: this.protocolData.siftedBits.length / this.simulationState.numBits,
            overallEfficiency: this.protocolData.finalKey.length / this.simulationState.numBits,
            informationLeakage: this.calculateInformationLeakage()
        };
    }
    
    /**
     * Process and display comprehensive simulation results
     */
    async processAndDisplayResults(result) {
        // Process external API results or use local results
        if (result.alice_bits) {
            this.processExternalResults(result);
        }
        
        // Display all results with animations
        await this.displayResults(result);
        await this.updateAllVisualizations();
        await this.generateDetailedReport();
    }
    
    /**
     * Process results from external API
     */
    processExternalResults(result) {
        this.protocolData = {
            aliceBits: result.alice_bits || [],
            aliceBases: result.alice_bases || [],
            bobBases: result.bob_bases || [],
            bobMeasurements: result.bob_measurements || [],
            siftedBits: (result.sifted_bits || []).map((bit, i) => ({
                aliceBit: bit,
                bobBit: bit,
                index: i
            })),
            errorRate: result.error_rate || 0,
            finalKey: result.shared_key || [],
            securityAnalysis: result.security_analysis || {},
            performanceMetrics: result.performance_metrics || {}
        };
    }
    
    /**
     * Display comprehensive results
     */
    async displayResults(result) {
        this.updateProtocolStats();
        this.updateSecurityMetrics();
        this.updateFinalKeyDisplay();
        
        // Animate results display
        await this.animateResultsDisplay();
    }
    
    /**
     * Update protocol statistics display
     */
    updateProtocolStats() {
        const stats = {
            'total-bits': this.protocolData.performanceMetrics.totalBits || this.simulationState.numBits,
            'sifted-bits': this.protocolData.performanceMetrics.siftedBits || this.protocolData.siftedBits.length,
            'final-key-length': this.protocolData.performanceMetrics.finalKeyLength || this.protocolData.finalKey.length,
            'protocol-efficiency': ((this.protocolData.performanceMetrics.overallEfficiency || 0) * 100).toFixed(1) + '%'
        };
        
        Object.entries(stats).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                this.animateValueUpdate(element, value);
            }
        });
    }
    
    /**
     * Update security metrics display
     */
    updateSecurityMetrics() {
        const analysis = this.protocolData.securityAnalysis;
        
        const metrics = {
            'error-rate': ((analysis.errorRate || 0) * 100).toFixed(2) + '%',
            'eavesdropping-detected': analysis.eavesdroppingDetected ? 'YES' : 'NO',
            'security-level': analysis.securityLevel || 'Unknown',
            'security-confidence': ((analysis.confidence || 0) * 100).toFixed(1) + '%'
        };
        
        Object.entries(metrics).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                this.animateValueUpdate(element, value);
                
                // Add color coding
                if (id === 'eavesdropping-detected') {
                    element.style.color = value === 'YES' ? '#ef4444' : '#10b981';
                }
            }
        });
    }
    
    /**
     * Update final key display with enhanced visualization
     */
    updateFinalKeyDisplay() {
        const keyContent = document.getElementById('final-key-content');
        const securityBadge = document.getElementById('security-badge');
        
        if (keyContent && this.protocolData.finalKey.length > 0) {
            // Create interactive bit display
            const keyString = this.protocolData.finalKey.join('');
            const bitElements = this.protocolData.finalKey.map((bit, index) => 
                `<span class="key-bit bit-${bit}" data-index="${index}" title="Bit ${index}: ${bit}">${bit}</span>`
            ).join('');
            
            keyContent.innerHTML = `
                <div class="key-bits">${bitElements}</div>
                <div class="key-string">${keyString}</div>
                <div class="key-info">
                    Length: ${this.protocolData.finalKey.length} bits |
                    Entropy: ${this.calculateKeyEntropy().toFixed(3)} |
                    Security: ${this.protocolData.securityAnalysis.securityLevel || 'Unknown'}
                </div>
            `;
            
            // Add click handlers for individual bits
            keyContent.querySelectorAll('.key-bit').forEach(bitElement => {
                bitElement.addEventListener('click', (e) => {
                    const index = parseInt(e.target.dataset.index);
                    this.showBitDetails(index);
                });
            });
        }
        
        if (securityBadge) {
            const analysis = this.protocolData.securityAnalysis;
            let badgeText, badgeClass;
            
            if (analysis.eavesdroppingDetected) {
                badgeText = '🚨 EAVESDROPPING DETECTED';
                badgeClass = 'security-badge insecure';
            } else if (analysis.isSecure) {
                badgeText = '🔒 SECURE';
                badgeClass = 'security-badge secure';
            } else if (analysis.errorRate > 0.05) {
                badgeText = '⚠️ WARNING';
                badgeClass = 'security-badge warning';
            } else {
                badgeText = '❓ UNKNOWN';
                badgeClass = 'security-badge pending';
            }
            
            securityBadge.textContent = badgeText;
            securityBadge.className = badgeClass;
        }
    }
    
    /**
     * Update all visualizations
     */
    async updateAllVisualizations() {
        try {
            await Promise.all([
                this.updateAliceVisualization(),
                this.updateBobVisualization(),
                this.updateChannelVisualization(),
                this.updateBlochSphere(),
                this.updateCircuitVisualization()
            ]);
        } catch (error) {
            console.warn('Some visualizations failed to update:', error);
        }
    }
    
    /**
     * Generate random bits
     */
    generateRandomBits(numBits) {
        return Array.from({ length: numBits }, () => Math.random() < 0.5 ? 0 : 1);
    }
    
    /**
     * Generate random measurement bases
     */
    generateRandomBases(numBits) {
        return Array.from({ length: numBits }, () => 
            Math.random() < 0.5 ? 'rectilinear' : 'diagonal'
        );
    }
    
    /**
     * Prepare quantum states based on bits and bases
     */
    prepareQuantumStates(bits, bases) {
        return bits.map((bit, index) => {
            const basis = bases[index];
            return this.encodeQuantumState(bit, basis);
        });
    }
    
    /**
     * Encode classical bit into quantum state
     */
    encodeQuantumState(bit, basis) {
        if (basis === 'rectilinear') {
            // Z basis: |0⟩ or |1⟩
            return {
                theta: bit === 0 ? 0 : Math.PI,
                phi: 0,
                basis: 'rectilinear',
                bit: bit
            };
        } else {
            // X basis: |+⟩ or |-⟩
            return {
                theta: Math.PI / 2,
                phi: bit === 0 ? 0 : Math.PI,
                basis: 'diagonal',
                bit: bit
            };
        }
    }
    
    /**
     * Apply channel noise to quantum state
     */
    applyChannelNoise(state) {
        if (this.simulationState.noiseLevel === 0) return state;
        
        const noisyState = { ...state };
        const noiseStrength = this.simulationState.noiseLevel * Math.PI / 4;
        
        noisyState.theta += (Math.random() - 0.5) * noiseStrength;
        noisyState.phi += (Math.random() - 0.5) * noiseStrength;
        
        // Keep angles in valid ranges
        noisyState.theta = Math.max(0, Math.min(Math.PI, noisyState.theta));
        noisyState.phi = ((noisyState.phi % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
        
        return noisyState;
    }
    
    /**
     * Simulate Eve's interference with quantum state
     */
    simulateEveInterference(state, index) {
        // Eve chooses random measurement basis
        const eveBasis = Math.random() < 0.5 ? 'rectilinear' : 'diagonal';
        
        // Eve measures the state
        const eveMeasurement = this.performQuantumMeasurement(state, eveBasis);
        
        // Eve re-encodes and re-transmits
        const modifiedState = this.encodeQuantumState(eveMeasurement, eveBasis);
        
        return {
            modifiedState: modifiedState,
            interceptionData: {
                index: index,
                originalState: state,
                eveBasis: eveBasis,
                eveMeasurement: eveMeasurement,
                modifiedState: modifiedState
            }
        };
    }
    
    /**
     * Perform quantum measurement
     */
    performQuantumMeasurement(state, measurementBasis) {
        // Calculate measurement probability
        let probability0;
        
        if (measurementBasis === 'rectilinear') {
            // Measuring in Z basis
            probability0 = Math.cos(state.theta / 2) ** 2;
        } else {
            // Measuring in X basis
            if (state.basis === 'rectilinear') {
                // State prepared in Z basis, measured in X basis
                probability0 = 0.5; // Always 50/50 for orthogonal bases
            } else {
                // State prepared in X basis, measured in X basis
                probability0 = Math.cos(state.phi / 2) ** 2;
            }
        }
        
        // Add noise effect
        if (this.simulationState.noiseLevel > 0) {
            const noiseEffect = this.simulationState.noiseLevel * 0.5;
            probability0 = probability0 * (1 - noiseEffect) + 0.5 * noiseEffect;
        }
        
        // Perform probabilistic measurement
        return Math.random() < probability0 ? 0 : 1;
    }
    
    /**
     * Sample random bits for testing
     */
    sampleRandomBits(bits, sampleSize) {
        const indices = Array.from({ length: bits.length }, (_, i) => i);
        const sampledIndices = [];
        
        for (let i = 0; i < sampleSize; i++) {
            const randomIndex = Math.floor(Math.random() * indices.length);
            sampledIndices.push(indices.splice(randomIndex, 1)[0]);
        }
        
        return sampledIndices.map(index => bits[index]);
    }
    
    /**
     * Calculate Eve's information
     */
    calculateEveInformation() {
        if (!this.simulationState.eavesdropper) return 0;
        
        // Simplified calculation based on error rate
        const errorRate = this.protocolData.errorRate;
        return Math.min(1, errorRate * 2); // Eve gets more info with higher error rates
    }
    
    /**
     * Apply universal hashing for privacy amplification
     */
    applyUniversalHashing(bits, targetLength) {
        // Simplified universal hashing
        const hashedBits = [];
        const hashFunction = (a, b) => a ^ b; // Simple XOR hash
        
        for (let i = 0; i < targetLength && i * 2 + 1 < bits.length; i++) {
            const hashedBit = hashFunction(bits[i * 2].bobBit, bits[i * 2 + 1].bobBit);
            hashedBits.push({
                aliceBit: hashedBit,
                bobBit: hashedBit,
                index: i
            });
        }
        
        return hashedBits;
    }
    
    /**
     * Determine security level based on error rate
     */
    determineSecurityLevel(errorRate) {
        if (errorRate < 0.05) return 'High';
        if (errorRate < 0.11) return 'Medium';
        return 'Low';
    }
    
    /**
     * Generate security recommendations
     */
    generateSecurityRecommendations() {
        const recommendations = [];
        const analysis = this.protocolData.securityAnalysis;
        
        if (analysis.eavesdroppingDetected) {
            recommendations.push('Eavesdropping detected - abort key distribution');
            recommendations.push('Check quantum channel for security breaches');
        }
        
        if (analysis.errorRate > 0.15) {
            recommendations.push('Error rate too high - consider improving channel quality');
        }
        
        if (!this.simulationState.errorCorrection) {
            recommendations.push('Enable error correction for improved security');
        }
        
        if (!this.simulationState.privacyAmplification) {
            recommendations.push('Enable privacy amplification to eliminate eavesdropper information');
        }
        
        return recommendations;
    }
    
    /**
     * Calculate information leakage
     */
    calculateInformationLeakage() {
        const originalEntropy = this.simulationState.numBits;
        const finalEntropy = this.protocolData.finalKey.length;
        return (originalEntropy - finalEntropy) / originalEntropy;
    }
    
    /**
     * Calculate key entropy
     */
    calculateKeyEntropy() {
        if (this.protocolData.finalKey.length === 0) return 0;
        
        const zeros = this.protocolData.finalKey.filter(bit => bit === 0).length;
        const ones = this.protocolData.finalKey.length - zeros;
        
        if (zeros === 0 || ones === 0) return 0;
        
        const p0 = zeros / this.protocolData.finalKey.length;
        const p1 = ones / this.protocolData.finalKey.length;
        
        return -(p0 * Math.log2(p0) + p1 * Math.log2(p1));
    }
    
    /**
     * Compile all simulation results
     */
    compileSimulationResults(simulationTime = 0) {
        return {
            timestamp: Date.now(),
            simulationTime: simulationTime,
            parameters: { ...this.simulationState },
            protocolData: { ...this.protocolData },
            summary: {
                success: this.protocolData.finalKey.length > 0,
                keyLength: this.protocolData.finalKey.length,
                efficiency: this.protocolData.performanceMetrics.overallEfficiency || 0,
                security: this.protocolData.securityAnalysis.securityLevel || 'Unknown'
            }
        };
    }
    
    // Visualization and Animation Methods
    
    /**
     * Update Alice's visualization
     */
    async updateAliceVisualization() {
        const stateContent = document.getElementById('alice-state-content');
        const basisContent = document.getElementById('alice-basis-content');
        
        if (stateContent && this.protocolData.aliceBits.length > 0) {
            const sampleSize = Math.min(20, this.protocolData.aliceBits.length);
            let html = '<div class="state-grid">';
            
            for (let i = 0; i < sampleSize; i++) {
                const bit = this.protocolData.aliceBits[i];
                const basis = this.protocolData.aliceBases[i];
                const symbol = basis === 'rectilinear' ? (bit === 0 ? '|0⟩' : '|1⟩') : (bit === 0 ? '|+⟩' : '|-⟩');
                
                html += `
                    <div class="quantum-state-item" data-index="${i}">
                        <div class="state-symbol">${symbol}</div>
                        <div class="state-info">Bit ${i}: ${bit} (${basis})</div>
                    </div>
                `;
            }
            
            html += '</div>';
            if (this.protocolData.aliceBits.length > sampleSize) {
                html += `<div class="state-more">... and ${this.protocolData.aliceBits.length - sampleSize} more states</div>`;
            }
            
            stateContent.innerHTML = html;
        }
        
        if (basisContent && this.protocolData.aliceBases.length > 0) {
            const basisCounts = this.protocolData.aliceBases.reduce((acc, basis) => {
                acc[basis] = (acc[basis] || 0) + 1;
                return acc;
            }, {});
            
            basisContent.innerHTML = `
                <div class="basis-stats">
                    <div class="basis-count">Rectilinear (+ basis): ${basisCounts.rectilinear || 0}</div>
                    <div class="basis-count">Diagonal (× basis): ${basisCounts.diagonal || 0}</div>
                </div>
            `;
        }
    }
    
    /**
     * Update Bob's visualization
     */
    async updateBobVisualization() {
        const measurementContent = document.getElementById('bob-measurement-content');
        const basisContent = document.getElementById('bob-basis-content');
        
        if (measurementContent && this.protocolData.bobMeasurements.length > 0) {
            const sampleSize = Math.min(20, this.protocolData.bobMeasurements.length);
            let html = '<div class="measurement-grid">';
            
            for (let i = 0; i < sampleSize; i++) {
                const measurement = this.protocolData.bobMeasurements[i];
                const basis = this.protocolData.bobBases[i];
                const aliceBasis = this.protocolData.aliceBases[i];
                const match = basis === aliceBasis;
                
                html += `
                    <div class="measurement-item ${match ? 'basis-match' : 'basis-mismatch'}" data-index="${i}">
                        <div class="measurement-result">${measurement}</div>
                        <div class="measurement-info">
                            Bob's basis: ${basis}<br>
                            Alice's basis: ${aliceBasis}<br>
                            Match: ${match ? '✅' : '❌'}
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            if (this.protocolData.bobMeasurements.length > sampleSize) {
                html += `<div class="measurement-more">... and ${this.protocolData.bobMeasurements.length - sampleSize} more measurements</div>`;
            }
            
            measurementContent.innerHTML = html;
        }
        
        if (basisContent && this.protocolData.bobBases.length > 0) {
            const basisCounts = this.protocolData.bobBases.reduce((acc, basis) => {
                acc[basis] = (acc[basis] || 0) + 1;
                return acc;
            }, {});
            
            const matchingBases = this.protocolData.aliceBases.filter(
                (basis, i) => basis === this.protocolData.bobBases[i]
            ).length;
            
            basisContent.innerHTML = `
                <div class="basis-comparison">
                    <div class="basis-stats">
                        <div class="basis-count">Rectilinear: ${basisCounts.rectilinear || 0}</div>
                        <div class="basis-count">Diagonal: ${basisCounts.diagonal || 0}</div>
                    </div>
                    <div class="basis-matching">
                        Matching bases: ${matchingBases}/${this.protocolData.bobBases.length} 
                        (${(matchingBases/this.protocolData.bobBases.length*100).toFixed(1)}%)
                    </div>
                </div>
            `;
        }
    }
    
    /**
     * Update channel visualization
     */
    async updateChannelVisualization() {
        const channelContent = document.getElementById('channel-content');
        const qubitStream = document.getElementById('qubit-stream');
        
        if (channelContent) {
            let statusHtml = '<div class="channel-status">';
            
            if (this.protocolData.transmittedStates && this.protocolData.transmittedStates.length > 0) {
                statusHtml += `
                    <div class="transmission-info">
                        <div class="info-item">Transmitted: ${this.protocolData.transmittedStates.length} qubits</div>
                        <div class="info-item">Noise Level: ${(this.simulationState.noiseLevel * 100).toFixed(1)}%</div>
                        <div class="info-item">Channel Status: ${this.simulationState.eavesdropper ? '🚨 Compromised' : '✅ Secure'}</div>
                    </div>
                `;
            } else {
                statusHtml += '<div class="channel-ready">Ready for quantum transmission...</div>';
            }
            
            statusHtml += '</div>';
            channelContent.innerHTML = statusHtml;
        }
        
        if (qubitStream && this.protocolData.transmittedStates) {
            this.animateQubitStream();
        }
    }
    
    /**
     * Update Bloch sphere visualization
     */
    updateBlochSphere() {
        if (this.visualizers.bloch) {
            const stateSelector = document.getElementById('state-selector');
            const selectedState = stateSelector ? stateSelector.value : '0';
            this.updateBlochSphereState(selectedState);
        }
    }
    
    /**
     * Update Bloch sphere state
     */
    updateBlochSphereState(stateType) {
        if (!this.visualizers.bloch) return;
        
        let theta, phi, label;
        switch (stateType) {
            case '0':
                theta = 0; phi = 0; label = '|0⟩';
                break;
            case '1':
                theta = Math.PI; phi = 0; label = '|1⟩';
                break;
            case 'plus':
                theta = Math.PI / 2; phi = 0; label = '|+⟩';
                break;
            case 'minus':
                theta = Math.PI / 2; phi = Math.PI; label = '|-⟩';
                break;
            case 'plus-i':
                theta = Math.PI / 2; phi = Math.PI / 2; label = '|+i⟩';
                break;
            case 'minus-i':
                theta = Math.PI / 2; phi = 3 * Math.PI / 2; label = '|-i⟩';
                break;
            default:
                theta = Math.PI / 2; phi = 0; label = '|+⟩';
        }
        
        this.visualizers.bloch.updateState(theta, phi, label);
    }
    
    /**
     * Update circuit visualization
     */
    async updateCircuitVisualization() {
        const circuitContainer = document.getElementById('bb84-circuit');
        if (circuitContainer && this.visualizers.circuit) {
            // Generate BB84 circuit diagram
            const circuitData = this.generateBB84Circuit();
            this.visualizers.circuit.renderCircuit(circuitData);
        }
    }
    
    /**
     * Generate BB84 circuit representation
     */
    generateBB84Circuit() {
        return {
            qubits: 4,
            gates: [
                { type: 'H', qubit: 0, time: 0 }, // Hadamard for superposition
                { type: 'X', qubit: 1, time: 0 }, // Bit flip
                { type: 'H', qubit: 1, time: 1 }, // Basis rotation
                { type: 'measure', qubit: 0, time: 2 },
                { type: 'measure', qubit: 1, time: 2 }
            ],
            title: 'BB84 Quantum Key Distribution Circuit'
        };
    }
    
    // Animation Methods
    
    /**
     * Animate Alice's state preparation
     */
    async animateAlicePreparation() {
        const stateItems = document.querySelectorAll('.quantum-state-item');
        for (let i = 0; i < stateItems.length; i++) {
            stateItems[i].style.opacity = '0';
            stateItems[i].style.transform = 'translateX(-20px)';
            
            setTimeout(() => {
                stateItems[i].style.transition = 'all 0.3s ease';
                stateItems[i].style.opacity = '1';
                stateItems[i].style.transform = 'translateX(0)';
            }, i * (50 / this.simulationState.animationSpeed));
        }
        
        await this.delay(stateItems.length * (50 / this.simulationState.animationSpeed) + 300);
    }
    
    /**
     * Animate qubit transmission
     */
    async animateQubitTransmission(index, state) {
        const qubitStream = document.getElementById('qubit-stream');
        if (!qubitStream) return;
        
        const qubitElement = document.createElement('div');
        qubitElement.className = `qubit qubit-${state.bit} transmission-animation`;
        qubitElement.textContent = state.bit;
        qubitElement.style.left = '0%';
        
        qubitStream.appendChild(qubitElement);
        
        // Animate across channel
        setTimeout(() => {
            qubitElement.style.left = '100%';
        }, 10);
        
        // Remove after animation
        setTimeout(() => {
            if (qubitElement.parentNode) {
                qubitElement.parentNode.removeChild(qubitElement);
            }
        }, 1000 / this.simulationState.animationSpeed);
        
        await this.delay(100 / this.simulationState.animationSpeed);
    }
    
    /**
     * Animate Bob's measurements
     */
    async animateBobMeasurement(index, measurement) {
        const measurementItems = document.querySelectorAll('.measurement-item');
        if (measurementItems[index % 20]) { // Only animate visible items
            const item = measurementItems[index % 20];
            item.classList.add('measurement-animation');
            
            setTimeout(() => {
                item.classList.remove('measurement-animation');
            }, 500 / this.simulationState.animationSpeed);
        }
        
        await this.delay(50 / this.simulationState.animationSpeed);
    }
    
    /**
     * Animate key sifting process
     */
    async animateKeySifting(siftedIndices) {
        const measurementItems = document.querySelectorAll('.measurement-item');
        
        measurementItems.forEach((item, index) => {
            const isSelected = siftedIndices.includes(index);
            
            setTimeout(() => {
                if (isSelected) {
                    item.classList.add('sifted-bit');
                } else {
                    item.classList.add('discarded-bit');
                }
            }, index * (30 / this.simulationState.animationSpeed));
        });
        
        await this.delay(measurementItems.length * (30 / this.simulationState.animationSpeed) + 500);
    }
    
    /**
     * Animate qubit stream in channel
     */
    animateQubitStream() {
        const qubitStream = document.getElementById('qubit-stream');
        if (!qubitStream) return;
        
        // Clear existing qubits
        qubitStream.innerHTML = '';
        
        // Add sample qubits
        const sampleSize = Math.min(10, this.protocolData.transmittedStates.length);
        for (let i = 0; i < sampleSize; i++) {
            const state = this.protocolData.transmittedStates[i];
            const qubitElement = document.createElement('div');
            qubitElement.className = `qubit qubit-${state.bit}`;
            qubitElement.textContent = state.bit;
            qubitElement.style.animationDelay = `${i * 0.2}s`;
            
            qubitStream.appendChild(qubitElement);
        }
    }
    
    /**
     * Animate results display
     */
    async animateResultsDisplay() {
        const resultCards = document.querySelectorAll('.result-card');
        
        resultCards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.5s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * (200 / this.simulationState.animationSpeed));
        });
        
        await this.delay(resultCards.length * (200 / this.simulationState.animationSpeed) + 500);
    }
    
    /**
     * Animate value updates
     */
    animateValueUpdate(element, newValue) {
        element.style.transition = 'all 0.3s ease';
        element.style.transform = 'scale(1.1)';
        element.textContent = newValue;
        
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 300);
    }
    
    // UI State Management
    
    /**
     * Update simulation progress
     */
    updateSimulationProgress(message, progress) {
        const progressIndicator = document.getElementById('simulation-progress');
        if (progressIndicator) {
            progressIndicator.style.display = 'block';
            progressIndicator.innerHTML = `
                <div class="progress-message">${message}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progress}%"></div>
                </div>
            `;
        }
    }
    
    /**
     * Highlight current step
     */
    highlightCurrentStep(stepIndex) {
        const stepCards = document.querySelectorAll('.step-card');
        stepCards.forEach((card, index) => {
            if (index === stepIndex) {
                card.classList.add('current-step');
            } else {
                card.classList.remove('current-step');
            }
        });
    }
    
    /**
     * Mark step as completed
     */
    completeStep(stepIndex) {
        const stepCards = document.querySelectorAll('.step-card');
        if (stepCards[stepIndex]) {
            stepCards[stepIndex].classList.remove('current-step');
            stepCards[stepIndex].classList.add('completed-step');
        }
    }
    
    /**
     * Update step content
     */
    updateStepContent(contentId, content) {
        const element = document.getElementById(contentId);
        if (element) {
            element.innerHTML = `<p>${content}</p>`;
        }
    }
    
    /**
     * Update step progress
     */
    updateStepProgress(stepId, progress) {
        const stepCard = document.getElementById(stepId);
        if (stepCard) {
            const progressBar = stepCard.querySelector('.step-progress .progress-fill');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }
        }
    }
    
    /**
     * Show loading state
     */
    showLoadingState() {
        const buttons = ['simulate-btn', 'step-btn', 'reset-btn'];
        buttons.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.disabled = true;
                btn.classList.add('loading');
                const icon = btn.querySelector('i');
                if (icon) {
                    icon.className = 'fas fa-spinner fa-spin';
                }
            }
        });
    }
    
    /**
     * Hide loading state
     */
    hideLoadingState() {
        const buttons = [
            { id: 'simulate-btn', icon: 'fas fa-play' },
            { id: 'step-btn', icon: 'fas fa-step-forward' },
            { id: 'reset-btn', icon: 'fas fa-redo' }
        ];
        
        buttons.forEach(btn => {
            const element = document.getElementById(btn.id);
            if (element) {
                element.disabled = false;
                element.classList.remove('loading');
                const icon = element.querySelector('i');
                if (icon) {
                    icon.className = btn.icon;
                }
            }
        });
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const errorContainer = document.getElementById('error-display') || this.createErrorContainer();
        errorContainer.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.style.display='none'">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        errorContainer.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 5000);
    }
    
    /**
     * Create error container if it doesn't exist
     */
    createErrorContainer() {
        const container = document.createElement('div');
        container.id = 'error-display';
        container.className = 'error-container';
        document.body.appendChild(container);
        return container;
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }
    
    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }
    
    /**
     * Show bit details in a modal or tooltip
     */
    showBitDetails(index) {
        if (index < this.protocolData.finalKey.length) {
            const bit = this.protocolData.finalKey[index];
            const details = `
                Bit Index: ${index}
                Value: ${bit}
                Origin: ${this.getBitOriginInfo(index)}
                Security: ${this.protocolData.securityAnalysis.securityLevel || 'Unknown'}
            `;
            
            this.showNotification(`Bit ${index}: ${bit}\n${details}`, 'info', 4000);
        }
    }
    
    /**
     * Get bit origin information
     */
    getBitOriginInfo(index) {
        // Trace back through the protocol steps
        return 'Traced through BB84 protocol';
    }
    
    /**
     * Generate detailed simulation report
     */
    async generateDetailedReport() {
        const report = {
            timestamp: new Date().toISOString(),
            parameters: this.simulationState,
            results: this.protocolData,
            analysis: this.generateAnalysis()
        };
        
        // Store report for potential download
        this.lastReport = report;
        
        console.log('Detailed BB84 Simulation Report:', report);
        return report;
    }
    
    /**
     * Generate analysis summary
     */
    generateAnalysis() {
        return {
            efficiency: {
                sifting: this.protocolData.siftedBits.length / this.simulationState.numBits,
                overall: this.protocolData.finalKey.length / this.simulationState.numBits
            },
            security: this.protocolData.securityAnalysis,
            recommendations: this.protocolData.securityAnalysis.recommendations || []
        };
    }
    
    /**
     * Reset simulation to initial state
     */
    resetSimulation() {
        // Reset protocol data
        this.protocolData = {
            aliceBits: [],
            aliceBases: [],
            aliceStates: [],
            bobBases: [],
            bobMeasurements: [],
            eveInterceptions: [],
            siftedBits: [],
            testBits: [],
            errorRate: 0,
            finalKey: [],
            securityAnalysis: {},
            performanceMetrics: {},
            transmissionLog: []
        };
        
        // Reset UI elements
        this.resetUIElements();
        
        // Clear visualizations
        this.clearVisualizations();
        
        this.showNotification('Simulation reset successfully', 'info');
    }
    
    /**
     * Reset UI elements
     */
    resetUIElements() {
        // Reset step cards
        const stepCards = document.querySelectorAll('.step-card');
        stepCards.forEach(card => {
            card.classList.remove('current-step', 'completed-step');
            const progressBar = card.querySelector('.progress-fill');
            if (progressBar) {
                progressBar.style.width = '0%';
            }
        });
        
        // Reset displays
        const displays = [
            'alice-state-content', 'alice-basis-content',
            'bob-measurement-content', 'bob-basis-content',
            'channel-content', 'final-key-content'
        ];
        
        displays.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.innerHTML = '<p>Ready for simulation...</p>';
            }
        });
        
        // Reset statistics
        const stats = ['total-bits', 'sifted-bits', 'final-key-length', 'protocol-efficiency'];
        stats.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '-';
            }
        });
        
        // Reset metrics
        const metrics = ['error-rate', 'eavesdropping-detected', 'security-level', 'security-confidence'];
        metrics.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '-';
                element.style.color = '';
            }
        });
        
        // Reset security badge
        const securityBadge = document.getElementById('security-badge');
        if (securityBadge) {
            securityBadge.textContent = '⏳ PENDING';
            securityBadge.className = 'security-badge pending';
        }
    }
    
    /**
     * Clear all visualizations
     */
    clearVisualizations() {
        // Clear qubit stream
        const qubitStream = document.getElementById('qubit-stream');
        if (qubitStream) {
            qubitStream.innerHTML = '<p>Ready for qubit transmission...</p>';
        }
        
        // Reset Bloch sphere
        if (this.visualizers.bloch) {
            this.visualizers.bloch.reset();
        }
        
        // Clear circuit
        const circuit = document.getElementById('bb84-circuit');
        if (circuit) {
            circuit.innerHTML = '<p>Quantum circuit will be displayed during simulation...</p>';
        }
    }
    
    /**
     * Update animation timings based on speed setting
     */
    updateAnimationTimings() {
        const speedMultiplier = this.simulationState.animationSpeed;
        this.simulationState.pauseBetweenSteps = Math.max(200, 1000 / speedMultiplier);
        
        // Update CSS animation speeds
        document.documentElement.style.setProperty('--animation-speed', `${speedMultiplier}`);
    }
    
    /**
     * Preload animations and visual resources
     */
    preloadAnimations() {
        // Preload any animation resources or setup
        console.log('Animations preloaded');
    }
    
    /**
     * Generate transmission log
     */
    generateTransmissionLog() {
        return this.protocolData.transmittedStates.map((state, index) => ({
            index: index,
            originalBit: this.protocolData.aliceBits[index],
            originalBasis: this.protocolData.aliceBases[index],
            transmittedState: state,
            noiseApplied: this.simulationState.noiseLevel > 0,
            eveInterference: this.simulationState.eavesdropper
        }));
    }
    
    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Quantum Particle System for visual effects
class QuantumParticleSystem {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.particles = [];
    }
    
    addParticle(bit, basis) {
        const particle = {
            bit: bit,
            basis: basis,
            x: 0,
            y: Math.random() * 100,
            speed: 1 + Math.random(),
            life: 100
        };
        
        this.particles.push(particle);
        this.renderParticle(particle);
    }
    
    renderParticle(particle) {
        const element = document.createElement('div');
        element.className = `quantum-particle bit-${particle.bit} basis-${particle.basis}`;
        element.style.left = particle.x + '%';
        element.style.top = particle.y + '%';
        element.textContent = particle.bit;
        
        if (this.container) {
            this.container.appendChild(element);
            
            setTimeout(() => {
                element.style.left = '100%';
            }, 50);
            
            setTimeout(() => {
                if (element.parentNode) {
                    element.parentNode.removeChild(element);
                }
            }, 2000);
        }
    }
}

// Global functions for HTML onclick handlers
function runBB84Simulation() {
    if (window.bb84Simulator) {
        window.bb84Simulator.runBB84Simulation();
    } else {
        console.error('BB84 Simulator not initialized');
    }
}

function runStepByStep() {
    if (window.bb84Simulator) {
        window.bb84Simulator.runStepByStep();
    } else {
        console.error('BB84 Simulator not initialized');
    }
}

function resetSimulation() {
    if (window.bb84Simulator) {
        window.bb84Simulator.resetSimulation();
    } else {
        console.error('BB84 Simulator not initialized');
    }
}

function updateBlochSphere() {
    if (window.bb84Simulator) {
        const selector = document.getElementById('state-selector');
        if (selector) {
            window.bb84Simulator.updateBlochSphereState(selector.value);
        }
    }
}

// Enhanced initialization with error handling
document.addEventListener('DOMContentLoaded', function() {
    try {
        window.bb84Simulator = new BB84Simulator();
        console.log('BB84 Simulator initialized successfully');
        
        // Add global error handler
        window.addEventListener('error', function(event) {
            console.error('Global error:', event.error);
            if (window.bb84Simulator) {
                window.bb84Simulator.showNotification('An error occurred. Please refresh the page.', 'error');
            }
        });
        
    } catch (error) {
        console.error('Failed to initialize BB84 Simulator:', error);
        
        // Show fallback error message
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `
            <div style="background: #fee; border: 1px solid #fcc; padding: 20px; margin: 20px; border-radius: 8px; color: #c33;">
                <h3>⚠️ Simulator Initialization Failed</h3>
                <p>The BB84 simulator could not be initialized. Please refresh the page or check the browser console for more details.</p>
                <details>
                    <summary>Error Details</summary>
                    <pre>${error.message}</pre>
                </details>
            </div>
        `;
        
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorDiv, main.firstChild);
        }
    }
});

// Export for use in other modules (Node.js compatibility)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BB84Simulator, QuantumParticleSystem };
}

// Enhanced CSS for new animations (inject into document head)
const enhancedStyles = `
    <style>
        .quantum-particle {
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
            transition: all 2s linear;
            z-index: 10;
        }
        
        .transmission-animation {
            animation: qubitTransmit 1s ease-in-out;
        }
        
        .measurement-animation {
            animation: measurementPulse 0.5s ease-in-out;
        }
        
        .sifted-bit {
            background: rgba(16, 185, 129, 0.2) !important;
            border: 2px solid #10b981;
        }
        
        .discarded-bit {
            background: rgba(239, 68, 68, 0.2) !important;
            border: 2px solid #ef4444;
            opacity: 0.5;
        }
        
        .current-step {
            border: 2px solid var(--primary-color);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
            transform: scale(1.02);
        }
        
        .completed-step {
            border: 2px solid var(--success-color);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .key-bits {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 10px 0;
        }
        
        .key-bit {
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .key-bit:hover {
            transform: scale(1.2);
            z-index: 10;
        }
        
        .basis-match {
            border-left: 3px solid var(--success-color);
        }
        
        .basis-mismatch {
            border-left: 3px solid var(--danger-color);
        }
        
        .eve-active {
            animation: eveWarning 1s ease-in-out infinite alternate;
        }
        
        @keyframes qubitTransmit {
            0% { transform: translateX(0) scale(1); opacity: 1; }
            50% { transform: translateX(200px) scale(1.2); opacity: 0.8; }
            100% { transform: translateX(400px) scale(1); opacity: 0.6; }
        }
        
        @keyframes measurementPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); background: rgba(99, 102, 241, 0.3); }
        }
        
        @keyframes eveWarning {
            0% { background: rgba(239, 68, 68, 0.1); }
            100% { background: rgba(239, 68, 68, 0.3); }
        }
        
        .error-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        }
        
        .error-message {
            background: rgba(239, 68, 68, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            backdrop-filter: blur(10px);
        }
        
        .error-message button {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            margin-left: auto;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            z-index: 1000;
            transform: translateX(400px);
            transition: transform 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 350px;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification.success {
            border-left: 4px solid var(--success-color);
        }
        
        .notification.error {
            border-left: 4px solid var(--danger-color);
        }
        
        .notification.warning {
            border-left: 4px solid var(--warning-color);
        }
        
        .notification.info {
            border-left: 4px solid var(--info-color);
        }
    </style>
`;

// Inject enhanced styles
document.head.insertAdjacentHTML('beforeend', enhancedStyles);
