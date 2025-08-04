"""
Key Features:
- Base64 image encoding/decoding
- Matplotlib figure to base64 conversion
- Image optimization for web display
- Error image generation
- SVG to PNG conversion utilities
"""

import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Utility class for image generation and processing
    """
    
    def __init__(self):
        """Initialize the image generator"""
        self.default_dpi = 100
        self.default_format = 'PNG'
        self.compression_quality = 85
        self.max_image_size = (1200, 900)  # Max width, height in pixels
        
    def matplotlib_to_base64(self, fig: plt.Figure, format: str = 'png', 
                           dpi: int = None, optimize: bool = True) -> str:
        """
        Convert matplotlib figure to base64 encoded string
        
        Args:
            fig: Matplotlib figure object
            format: Image format ('png', 'jpg', 'svg')
            dpi: Image resolution (dots per inch)
            optimize: Whether to optimize the image for web
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            if dpi is None:
                dpi = self.default_dpi
            
            # Create buffer
            buffer = io.BytesIO()
            
            # Save figure to buffer
            fig.savefig(buffer, format=format.lower(), dpi=dpi, 
                       bbox_inches='tight', facecolor='white', 
                       edgecolor='none', transparent=False)
            buffer.seek(0)
            
            # Optimize if requested
            if optimize and format.lower() in ['png', 'jpg', 'jpeg']:
                image_data = self._optimize_image(buffer, format)
            else:
                image_data = buffer.getvalue()
            
            # Encode to base64
            encoded_string = base64.b64encode(image_data).decode('utf-8')
            
            # Clean up
            buffer.close()
            plt.close(fig)
            
            return encoded_string
            
        except Exception as e:
            logger.error(f"Matplotlib to base64 conversion failed: {str(e)}")
            plt.close(fig)  # Ensure figure is closed
            return self.generate_error_image_base64(f"Image generation failed: {str(e)}")
    
    def _optimize_image(self, buffer: io.BytesIO, format: str) -> bytes:
        """Optimize image for web display"""
        try:
            # Open image with PIL
            image = Image.open(buffer)
            
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Create new buffer for optimized image
            optimized_buffer = io.BytesIO()
            
            # Save with optimization
            if format.lower() == 'png':
                image.save(optimized_buffer, format='PNG', optimize=True)
            elif format.lower() in ['jpg', 'jpeg']:
                # Convert to RGB if necessary (JPEG doesn't support transparency)
                if image.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
                
                image.save(optimized_buffer, format='JPEG', 
                          quality=self.compression_quality, optimize=True)
            
            optimized_buffer.seek(0)
            optimized_data = optimized_buffer.getvalue()
            optimized_buffer.close()
            
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {str(e)}, using original")
            buffer.seek(0)
            return buffer.getvalue()
    
    def numpy_array_to_base64(self, array: np.ndarray, colormap: str = 'viridis',
                            title: str = None) -> str:
        """
        Convert numpy array to base64 encoded image
        
        Args:
            array: 2D numpy array to visualize
            colormap: Matplotlib colormap name
            title: Optional title for the plot
            
        Returns:
            str: Base64 encoded image
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create image plot
            im = ax.imshow(array, cmap=colormap, aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add title if provided
            if title:
                ax.set_title(title, fontsize=14, pad=20)
            
            # Remove axis ticks if array is large
            if array.shape[0] > 20 or array.shape[1] > 20:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            return self.matplotlib_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Numpy array visualization failed: {str(e)}")
            return self.generate_error_image_base64(str(e))
    
    def generate_error_image_base64(self, error_message: str, 
                                  size: Tuple[int, int] = (400, 300)) -> str:
        """
        Generate base64 encoded error image
        
        Args:
            error_message: Error message to display
            size: Image size (width, height)
            
        Returns:
            str: Base64 encoded error image
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100))
            
            # Set background color
            fig.patch.set_facecolor('#ffebee')
            ax.set_facecolor('#ffebee')
            
            # Add error icon and message
            ax.text(0.5, 0.6, '⚠️', fontsize=48, ha='center', va='center', 
                   transform=ax.transAxes)
            
            # Split long error messages
            words = error_message.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Display error message
            error_text = '\n'.join(lines)
            ax.text(0.5, 0.3, f'Error:\n{error_text}', fontsize=12, ha='center', va='center',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
                   facecolor="white", edgecolor="red", alpha=0.8))
            
            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            return self.matplotlib_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error image generation failed: {str(e)}")
            # Return minimal base64 encoded 1x1 pixel image as last resort
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def create_quantum_gate_image(self, gate_type: str, size: Tuple[int, int] = (60, 60)) -> str:
        """
        Generate quantum gate visualization
        
        Args:
            gate_type: Type of quantum gate ('H', 'X', 'Y', 'Z', 'CNOT')
            size: Image size (width, height)
            
        Returns:
            str: Base64 encoded gate image
        """
        try:
            fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100))
            
            # Gate colors
            gate_colors = {
                'H': '#f59e0b',   # Yellow/orange
                'X': '#ef4444',   # Red
                'Y': '#10b981',   # Green
                'Z': '#3b82f6',   # Blue
                'CNOT': '#8b5cf6' # Purple
            }
            
            color = gate_colors.get(gate_type, '#6b7280')
            
            if gate_type == 'CNOT':
                # Draw CNOT gate (control and target)
                # Control dot
                circle = patches.Circle((0.3, 0.7), 0.1, facecolor=color, edgecolor='black')
                ax.add_patch(circle)
                
                # Vertical line
                ax.plot([0.3, 0.3], [0.3, 0.7], 'k-', linewidth=3)
                
                # Target (circle with plus)
                target_circle = patches.Circle((0.3, 0.3), 0.15, facecolor='white', 
                                             edgecolor=color, linewidth=3)
                ax.add_patch(target_circle)
                ax.plot([0.3, 0.3], [0.15, 0.45], color=color, linewidth=3)
                ax.plot([0.15, 0.45], [0.3, 0.3], color=color, linewidth=3)
            else:
                # Draw single gate box
                box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(box)
                
                # Add gate label
                ax.text(0.5, 0.5, gate_type, fontsize=24, fontweight='bold',
                       ha='center', va='center', color='white')
            
            # Set limits and remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            return self.matplotlib_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Quantum gate image generation failed: {str(e)}")
            return self.generate_error_image_base64(f"Gate {gate_type} generation failed")
    
    def create_bloch_sphere_visualization(self, theta: float, phi: float, 
                                        size: Tuple[int, int] = (400, 400)) -> str:
        """
        Create a Bloch sphere visualization with quantum state
        
        Args:
            theta: Polar angle in radians (0 to π)
            phi: Azimuthal angle in radians (0 to 2π)
            size: Image size (width, height)
            
        Returns:
            str: Base64 encoded Bloch sphere image
        """
        try:
            # Create figure with dark theme
            fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), 
                                  facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            
            # Set up the plot
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Draw the Bloch sphere (circle)
            circle = plt.Circle((0, 0), 1, fill=False, color='#ffffff', linewidth=2)
            ax.add_patch(circle)
            
            # Draw coordinate axes
            # X-axis (|+⟩ direction)
            ax.arrow(0, 0, 1.1, 0, head_width=0.05, head_length=0.05, 
                    fc='#ff6b6b', ec='#ff6b6b', linewidth=2)
            ax.text(1.15, 0, '|+⟩', color='#ff6b6b', fontsize=12, ha='left', va='center')
            
            # Y-axis (|i⟩ direction)
            ax.arrow(0, 0, 0, 1.1, head_width=0.05, head_length=0.05, 
                    fc='#4ecdc4', ec='#4ecdc4', linewidth=2)
            ax.text(0, 1.15, '|i⟩', color='#4ecdc4', fontsize=12, ha='center', va='bottom')
            
            # Z-axis (|0⟩ and |1⟩ directions)
            ax.arrow(0, 0, 0, -1.1, head_width=0.05, head_length=0.05, 
                    fc='#45b7d1', ec='#45b7d1', linewidth=2)
            ax.text(0, -1.15, '|1⟩', color='#45b7d1', fontsize=12, ha='center', va='top')
            ax.text(0, 1.15, '|0⟩', color='#45b7d1', fontsize=12, ha='center', va='bottom')
            
            # Calculate quantum state position
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Draw state vector (projected to 2D)
            state_x = x
            state_y = y
            
            # Handle edge cases for special states
            if abs(theta) < 1e-10:  # |0⟩ state (North pole)
                state_x = 0
                state_y = 1
            elif abs(theta - np.pi) < 1e-10:  # |1⟩ state (South pole)
                state_x = 0
                state_y = -1
            elif abs(theta - np.pi/2) < 1e-10 and abs(phi) < 1e-10:  # |+⟩ state
                state_x = 1
                state_y = 0
            elif abs(theta - np.pi/2) < 1e-10 and abs(phi - np.pi) < 1e-10:  # |-⟩ state
                state_x = -1
                state_y = 0
            elif abs(theta - np.pi/2) < 1e-10 and abs(phi - np.pi/2) < 1e-10:  # |i⟩ state
                state_x = 0
                state_y = 1
            elif abs(theta - np.pi/2) < 1e-10 and abs(phi - 3*np.pi/2) < 1e-10:  # |-i⟩ state
                state_x = 0
                state_y = -1
            
            # Draw state vector line
            ax.plot([0, state_x], [0, state_y], color='#ffaa00', linewidth=3, alpha=0.8)
            
            # Draw state point
            ax.scatter([state_x], [state_y], color='#ffaa00', s=100, zorder=5, 
                      edgecolors='#ffffff', linewidth=2)
            
            # Add state label
            ax.text(state_x + 0.1, state_y + 0.1, '|ψ⟩', color='#ffaa00', 
                   fontsize=14, weight='bold', ha='center', va='center')
            
            # Calculate state vector components
            alpha = np.cos(theta / 2)
            beta_real = np.sin(theta / 2) * np.cos(phi)
            beta_imag = np.sin(theta / 2) * np.sin(phi)
            
            # Add state information
            state_text = f'θ = {np.degrees(theta):.1f}°\nφ = {np.degrees(phi):.1f}°\n'
            state_text += f'|ψ⟩ = {alpha:.3f}|0⟩ + ({beta_real:.3f} + {beta_imag:.3f}i)|1⟩'
            
            ax.text(-1.1, -1.1, state_text, color='#ffffff', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', alpha=0.8))
            
            # Add title
            ax.text(0, 1.3, 'Bloch Sphere Visualization', color='#ffffff', 
                   fontsize=16, weight='bold', ha='center')
            
            return self.matplotlib_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Bloch sphere visualization failed: {str(e)}")
            return self.generate_error_image_base64(f"Bloch sphere failed: {str(e)}")

    def create_circuit_diagram(self, circuit_spec: Dict[str, Any]) -> str:
        """
        Generate quantum circuit diagram
        
        Args:
            circuit_spec: Circuit specification with qubits and gates
            
        Returns:
            str: Base64 encoded circuit diagram
        """
        try:
            num_qubits = circuit_spec.get('num_qubits', 2)
            gates = circuit_spec.get('gates', [])
            
            # Calculate figure size based on circuit complexity
            width = max(8, len(gates) * 1.5 + 2)
            height = max(4, num_qubits * 1.5 + 1)
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Draw qubit lines
            for i in range(num_qubits):
                y_pos = num_qubits - i - 1
                ax.axhline(y=y_pos, color='black', linewidth=2, 
                          xmin=0.1, xmax=0.9)
                
                # Qubit labels
                ax.text(0.05, y_pos, f'|q{i}⟩', fontsize=12, 
                       ha='right', va='center')
            
            # Draw gates
            gate_x_positions = np.linspace(0.2, 0.8, max(1, len(gates)))
            
            for gate_idx, gate in enumerate(gates):
                x_pos = gate_x_positions[gate_idx] if len(gates) > 0 else 0.5
                gate_type = gate.get('type', 'H')
                target_qubit = gate.get('qubit', 0)
                control_qubit = gate.get('control', None)
                
                if gate_type == 'CNOT' and control_qubit is not None:
                    # Draw CNOT gate
                    control_y = num_qubits - control_qubit - 1
                    target_y = num_qubits - target_qubit - 1
                    
                    # Control dot
                    ax.plot(x_pos, control_y, 'ko', markersize=8)
                    
                    # Vertical line
                    ax.plot([x_pos, x_pos], [min(control_y, target_y), max(control_y, target_y)], 
                           'k-', linewidth=2)
                    
                    # Target circle
                    target_circle = patches.Circle((x_pos, target_y), 0.15, 
                                                 facecolor='white', edgecolor='black', linewidth=2)
                    ax.add_patch(target_circle)
                    
                    # Plus sign in target
                    ax.plot([x_pos-0.1, x_pos+0.1], [target_y, target_y], 'k-', linewidth=2)
                    ax.plot([x_pos, x_pos], [target_y-0.1, target_y+0.1], 'k-', linewidth=2)
                else:
                    # Single qubit gate
                    y_pos = num_qubits - target_qubit - 1
                    
                    # Gate colors
                    gate_colors = {
                        'H': '#f59e0b', 'X': '#ef4444', 'Y': '#10b981', 
                        'Z': '#3b82f6', 'I': '#6b7280'
                    }
                    color = gate_colors.get(gate_type, '#6b7280')
                    
                    # Draw gate box
                    gate_box = FancyBboxPatch((x_pos-0.1, y_pos-0.15), 0.2, 0.3,
                                            boxstyle="round,pad=0.02", facecolor=color,
                                            edgecolor='black', linewidth=1)
                    ax.add_patch(gate_box)
                    
                    # Gate label
                    ax.text(x_pos, y_pos, gate_type, fontsize=10, fontweight='bold',
                           ha='center', va='center', color='white')
            
            # Set limits and styling
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, num_qubits - 0.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            plt.tight_layout()
            
            return self.matplotlib_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Circuit diagram generation failed: {str(e)}")
            return self.generate_error_image_base64(str(e))
    
    def base64_to_image(self, base64_string: str, output_format: str = 'PNG') -> Optional[Image.Image]:
        """
        Convert base64 string back to PIL Image
        
        Args:
            base64_string: Base64 encoded image string
            output_format: Desired output format
            
        Returns:
            PIL Image object or None if conversion fails
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Create PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert format if requested
            if output_format.upper() != image.format:
                if output_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                    # Convert to RGB for JPEG
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
            
            return image
            
        except Exception as e:
            logger.error(f"Base64 to image conversion failed: {str(e)}")
            return None
    
    def create_data_url(self, base64_string: str, mime_type: str = 'image/png') -> str:
        """
        Create data URL from base64 string
        
        Args:
            base64_string: Base64 encoded image
            mime_type: MIME type for the image
            
        Returns:
            str: Complete data URL
        """
        return f"data:{mime_type};base64,{base64_string}"
    
    def resize_base64_image(self, base64_string: str, new_size: Tuple[int, int],
                          maintain_aspect_ratio: bool = True) -> str:
        """
        Resize base64 encoded image
        
        Args:
            base64_string: Original base64 image
            new_size: New size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            str: Resized base64 image
        """
        try:
            # Convert to PIL Image
            image = self.base64_to_image(base64_string)
            if image is None:
                return base64_string  # Return original if conversion fails
            
            # Resize
            if maintain_aspect_ratio:
                image.thumbnail(new_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            resized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            
            return resized_base64
            
        except Exception as e:
            logger.error(f"Image resize failed: {str(e)}")
            return base64_string  # Return original on error
    
    def validate_base64_image(self, base64_string: str) -> Dict[str, Any]:
        """
        Validate base64 encoded image
        
        Args:
            base64_string: Base64 string to validate
            
        Returns:
            dict: Validation results
        """
        try:
            # Remove data URL prefix if present
            clean_b64 = base64_string
            if base64_string.startswith('data:image'):
                clean_b64 = base64_string.split(',')[1]
            
            # Try to decode
            image_data = base64.b64decode(clean_b64)
            
            # Try to open with PIL
            image = Image.open(io.BytesIO(image_data))
            
            return {
                'valid': True,
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'data_size_bytes': len(image_data),
                'has_transparency': image.mode in ('RGBA', 'LA', 'P')
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def get_image_info(self, base64_string: str) -> Dict[str, Any]:
        """
        Get detailed information about base64 encoded image
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            dict: Image information
        """
        validation_result = self.validate_base64_image(base64_string)
        
        if validation_result['valid']:
            # Calculate additional metrics
            width, height = validation_result['size']
            pixel_count = width * height
            data_size_kb = validation_result['data_size_bytes'] / 1024
            
            # Estimate compression ratio (rough)
            uncompressed_size = pixel_count * 3  # Assume RGB
            compression_ratio = uncompressed_size / validation_result['data_size_bytes']
            
            validation_result.update({
                'pixel_count': pixel_count,
                'data_size_kb': round(data_size_kb, 2),
                'estimated_compression_ratio': round(compression_ratio, 2),
                'aspect_ratio': round(width / height, 3) if height > 0 else 0
            })
        
        return validation_result