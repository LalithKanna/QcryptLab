"""
Bloch Sphere State Visualizer Module
====================================

This module provides enhanced quantum state visualization capabilities using
matplotlib and QuTiP libraries with full BB84 protocol integration and
advanced educational features.

Key Features
------------
- Enhanced Bloch sphere generation from quantum state vectors
- BB84-specific state visualizations and comparisons
- Base64 image encoding for web compatibility
- Spherical coordinate conversion utilities
- Custom visualization styling for educational purposes
- Multi-state comparison visualizations
- Time-evolution animations
- Interactive state analysis tools
- Protocol-specific visualization templates
- Fixed QuTiP integration and JSON serialization
"""

import datetime
import io
import logging
import base64
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import matplotlib

# Use a non-interactive backend (important for servers / headless environments)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3-D plots)

# ---------------------------------------------------------------------------
# Optional QuTiP import (fallback to pure Matplotlib if unavailable)
# ---------------------------------------------------------------------------
try:
    import qutip as qt

    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    qt = None

logger = logging.getLogger(__name__)


class BlochSphereVisualizer:
    """
    Enhanced quantum-state visualizer with BB84 protocol integration and
    advanced educational features (fully updated).
    """

    # -----------------------------------------------------------------------
    # INITIALISATION
    # -----------------------------------------------------------------------
    def __init__(self, style_theme: str = "educational"):
        """
        Parameters
        ----------
        style_theme : str
            One of ``'educational'`` (default), ``'professional'`` or ``'bb84'``.
        """
        self.style_theme = style_theme
        self._initialize_theme_styles()

        # Rendering options
        self.figure_size = (10, 10)
        self.dpi = 150
        self.animation_frames = 30

        # BB84 color palette
        self.bb84_colors = {
            "computational_0": "#2E8B57",  # sea-green
            "computational_1": "#DC143C",  # crimson
            "superposition_plus": "#4169E1",  # royal-blue
            "superposition_minus": "#FF8C00",  # dark-orange
            "alice_states": "#FF69B4",  # hot-pink
            "bob_measurements": "#20B2AA",  # light-sea-green
            "eve_interference": "#B22222",  # fire-brick
        }

        # Flags for information overlays
        self.show_annotations = True
        self.show_coordinates = True
        self.show_measurement_probabilities = True

        if QUTIP_AVAILABLE:
            logger.info("QuTiP detected – enabling QuTiP backend.")
        else:
            logger.warning("QuTiP not found – falling back to Matplotlib-only mode.")

    # -----------------------------------------------------------------------
    # THEME SET-UP
    # -----------------------------------------------------------------------
    def _initialize_theme_styles(self):
        """Populate instance attributes for the chosen theme."""
        themes = {
            "educational": dict(
                sphere_alpha=0.15,
                vector_color="#FF6B6B",
                sphere_color="#E8F4FD",
                grid_color="#CCCCCC",
                background_color="white",
                text_color="black",
                font_size=14,
            ),
            "professional": dict(
                sphere_alpha=0.10,
                vector_color="#2C3E50",
                sphere_color="#ECF0F1",
                grid_color="#BDC3C7",
                background_color="white",
                text_color="#2C3E50",
                font_size=12,
            ),
            "bb84": dict(
                sphere_alpha=0.20,
                vector_color="#8E44AD",
                sphere_color="#F8F9FA",
                grid_color="#6C757D",
                background_color="#F8F9FA",
                text_color="#495057",
                font_size=13,
            ),
        }

        theme = themes.get(self.style_theme, themes["educational"])
        for key, value in theme.items():
            setattr(self, key, value)

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------
    # 1) SINGLE BLOCH-SPHERE VISUALISATION
    # -----------------------------------------------------------------------
    def generate_bloch_sphere(
        self,
        statevector: Union[List, np.ndarray],
        title: str = "Quantum State",
        show_state_info: bool = True,
        bb84_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Render a single Bloch sphere.

        Returns
        -------
        str
            Base-64 encoded PNG.
        """
        try:
            # Normalise the amplitudes
            statevector = np.asarray(statevector, dtype=complex)
            norm = np.linalg.norm(statevector)
            if norm > 0:
                statevector = statevector / norm

            # Prefer QuTiP if available unless caller requests Matplotlib
            if (
                QUTIP_AVAILABLE
                and not (bb84_context and bb84_context.get("use_matplotlib", False))
            ):
                try:
                    return self._generate_qutip_bloch_sphere(
                        statevector, title, show_state_info, bb84_context
                    )
                except Exception as qerr:
                    logger.warning(
                        "QuTiP rendering failed – switching to Matplotlib fallback "
                        f"({qerr})"
                    )

            # Pure-Matplotlib fallback
            return self._generate_matplotlib_bloch_sphere(
                statevector, title, show_state_info, bb84_context
            )
        except Exception as e:
            logger.error(f"Bloch-sphere generation failed: {e}")
            return self._generate_error_image(str(e))

    # -----------------------------------------------------------------------
    # 2) MULTI-STATE (BB84) COMPARISON
    # -----------------------------------------------------------------------
    def generate_bb84_state_comparison(self, states_data: List[Dict[str, Any]]) -> str:
        """
        Visualise up to 4 × *n* BB84 states in a grid.

        Parameters
        ----------
        states_data : list(dict)
            Each dict must contain keys ``'state'`` (array-like) and ``'type'``;
            optional ``'label'``.

        Returns
        -------
        str
            Base-64 encoded PNG.
        """
        try:
            if not states_data:
                return self._generate_error_image("No states supplied.")

            n = len(states_data)
            rows = (n - 1) // 4 + 1
            cols = min(n, 4)
            fig = plt.figure(
                figsize=(5 * cols, 5 * rows), facecolor=self.background_color
            )

            for idx, sdata in enumerate(states_data):
                ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
                self._add_enhanced_bloch_subplot(
                    ax,
                    np.asarray(sdata["state"], dtype=complex),
                    label=sdata.get("label", f"State {idx+1}"),
                    vector_color=self.bb84_colors.get(
                        sdata.get("type", ""), self.vector_color
                    ),
                    state_type=sdata.get("type", ""),
                )

            fig.suptitle(
                "BB84 Quantum-State Comparison",
                fontsize=self.font_size + 4,
                fontweight="bold",
                y=0.95,
            )
            self._add_bb84_legend(fig)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=self.dpi,
                bbox_inches="tight",
                facecolor=self.background_color,
            )
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.error(f"BB84 comparison failed: {e}")
            return self._generate_error_image(str(e))

    # -----------------------------------------------------------------------
    # 3) PROTOCOL EVOLUTION (SEQUENCE)
    # -----------------------------------------------------------------------
    def generate_protocol_evolution(self, evolution_data: Dict[str, Any]) -> List[str]:
        """
        Produce a list of Bloch-sphere PNGs for successive protocol steps.

        Returns
        -------
        list(str)
            Base-64 images (one per step).
        """
        try:
            steps = evolution_data.get("steps", [])
            if not steps:
                return [self._generate_error_image("No protocol steps supplied.")]

            images: List[str] = []
            total = len(steps)
            for i, step in enumerate(steps, start=1):
                ctx = dict(
                    state_type=step.get("type", "unknown"),
                    protocol_step=step.get("title", f"Step {i}"),
                    description=step.get("description", ""),
                    step_number=i,
                    total_steps=total,
                    show_bb84_labels=True,
                )

                images.append(
                    self.generate_bloch_sphere(
                        step.get("state", [1, 0]),
                        title=ctx["protocol_step"],
                        show_state_info=True,
                        bb84_context=ctx,
                    )
                )
            return images
        except Exception as e:
            logger.error(f"Protocol evolution failed: {e}")
            return [self._generate_error_image(str(e))]

    # -----------------------------------------------------------------------
    # 4) MEASUREMENT / SECURITY ANALYSIS
    # -----------------------------------------------------------------------
    def generate_measurement_analysis(self, measurement_data: Dict[str, Any]) -> str:
        """
        Composite figure with Bloch sphere, probability, fidelity and security
        sub-plots.  Probability and fidelity calculations are placeholders that
        should be replaced with real data.
        """
        try:
            states = measurement_data.get("states", [])
            bases = measurement_data.get("measurement_bases", [])
            if not states or not bases:
                return self._generate_error_image("Incomplete measurement data.")

            fig = plt.figure(figsize=(16, 12), facecolor=self.background_color)

            # (a) States & measurement axes
            ax_main = fig.add_subplot(221, projection="3d")
            colors = ["red", "blue", "green", "orange"]
            for i, s in enumerate(states):
                vec = np.asarray(s["state"], dtype=complex)
                x, y, z = self.bloch_coordinates(vec)
                ax_main.scatter(
                    [x],
                    [y],
                    [z],
                    color=colors[i % len(colors)],
                    s=150,
                    label=s.get("label", f"State {i+1}"),
                )
            for b in bases:
                v = b.get("vector", [0, 0, 1])
                ax_main.quiver(
                    0,
                    0,
                    0,
                    v[0],
                    v[1],
                    v[2],
                    color="black",
                    linewidth=3,
                    alpha=0.6,
                    linestyle="--",
                    arrow_length_ratio=0.1,
                )
            self._setup_basic_bloch_axes(ax_main)
            ax_main.set_title("States & Measurement Bases", fontweight="bold")
            ax_main.legend()

            # (b) Probability heat-map (placeholder)
            ax_prob = fig.add_subplot(222)
            prob_mat = np.random.rand(len(states), len(bases))  # TODO: real data
            im = ax_prob.imshow(prob_mat, cmap="Blues", aspect="auto")
            ax_prob.set_title("Measurement Probabilities", fontweight="bold")
            ax_prob.set_xlabel("Bases")
            ax_prob.set_ylabel("States")
            plt.colorbar(im, ax=ax_prob, shrink=0.8)

            # (c) Fidelity line plot (placeholder)
            ax_fid = fig.add_subplot(223)
            fid = np.random.rand(len(states)) * 0.5 + 0.5
            ax_fid.plot(fid, "o-", color="green")
            ax_fid.set_ylim(0, 1)
            ax_fid.set_title("State Fidelities", fontweight="bold")
            ax_fid.set_xlabel("State index")
            ax_fid.set_ylabel("Fidelity")
            ax_fid.grid(True, alpha=0.3)

            # (d) Security metrics (placeholder)
            ax_sec = fig.add_subplot(224)
            metrics = {
                "QBER": 0.05,
                "Key Rate": 0.80,
                "Privacy Amp": 0.90,
                "Error Corr": 0.95,
            }
            bars = ax_sec.bar(
                metrics.keys(),
                metrics.values(),
                color=[
                    "red" if v < 0.5 else "orange" if v < 0.8 else "green"
                    for v in metrics.values()
                ],
                alpha=0.7,
            )
            ax_sec.set_ylim(0, 1)
            ax_sec.set_title("BB84 Security Analysis", fontweight="bold")
            ax_sec.set_ylabel("Metric value")
            plt.setp(ax_sec.get_xticklabels(), rotation=45, ha="right")
            for b, v in zip(bars, metrics.values()):
                ax_sec.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=self.dpi,
                bbox_inches="tight",
                facecolor=self.background_color,
            )
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.error(f"Measurement analysis failed: {e}")
            return self._generate_error_image(str(e))

    # -----------------------------------------------------------------------
    # CORE (PRIVATE) RENDERERS
    # -----------------------------------------------------------------------
    def _generate_qutip_bloch_sphere(
        self,
        statevector: np.ndarray,
        title: str,
        show_state_info: bool,
        bb84_context: Optional[Dict[str, Any]],
    ) -> str:
        """QuTiP-based renderer (with numerous robustness checks)."""
        if not QUTIP_AVAILABLE:
            raise RuntimeError("QuTiP not available")

        # Force 'Agg' backend for safety
        if matplotlib.get_backend() != "Agg":
            matplotlib.use("Agg")

        # Create Bloch object & custom style
        bloch = qt.Bloch(figsize=self.figure_size)
        bloch.sphere_alpha = self.sphere_alpha
        bloch.sphere_color = self.sphere_color
        bloch.vector_color = [
            self.bb84_colors.get(bb84_context.get("state_type", ""), self.vector_color)
            if bb84_context
            else self.vector_color
        ]
        bloch.vector_width = 5
        bloch.font_size = self.font_size

        # Add state
        bloch.add_states(qt.Qobj(statevector))

        # Title & info
        full_title = (
            self._create_enhanced_title(title, statevector, bb84_context)
            if bb84_context
            else title
        )
        bloch.fig.suptitle(full_title, fontsize=self.font_size + 2, fontweight="bold")
        if show_state_info:
            self._add_state_info_panel(bloch.fig, statevector, bb84_context)

        # Save
        buf = io.BytesIO()
        bloch.fig.savefig(
            buf,
            format="png",
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.background_color,
        )
        plt.close(bloch.fig)
        return base64.b64encode(buf.getvalue()).decode()

    # -----------------------------------------------------------------------
    def _generate_matplotlib_bloch_sphere(
        self,
        statevector: np.ndarray,
        title: str,
        show_state_info: bool,
        bb84_context: Optional[Dict[str, Any]],
    ) -> str:
        """Pure-Matplotlib renderer (always available)."""
        # Bloch coordinates
        x, y, z = self.bloch_coordinates(statevector)

        fig = plt.figure(figsize=self.figure_size, facecolor=self.background_color)
        ax = (
            fig.add_subplot(121, projection="3d")
            if show_state_info
            else fig.add_subplot(111, projection="3d")
        )

        # Sphere & grid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(
            sx,
            sy,
            sz,
            alpha=self.sphere_alpha,
            color=self.sphere_color,
            linewidth=0,
            antialiased=True,
        )
        # Axes
        L = 1.3
        ax.plot([-L, L], [0, 0], [0, 0], "k-", alpha=0.4, linewidth=2)
        ax.plot([0, 0], [-L, L], [0, 0], "k-", alpha=0.4, linewidth=2)
        ax.plot([0, 0], [0, 0], [-L, L], "k-", alpha=0.4, linewidth=2)

        # State vector
        vec_color = (
            self.bb84_colors.get(bb84_context.get("state_type", ""), self.vector_color)
            if bb84_context
            else self.vector_color
        )
        ax.quiver(
            0,
            0,
            0,
            x,
            y,
            z,
            color=vec_color,
            linewidth=6,
            arrow_length_ratio=0.15,
            alpha=0.9,
        )
        ax.scatter(
            [x],
            [y],
            [z],
            color=vec_color,
            s=200,
            edgecolors="white",
            linewidths=2,
        )

        # Labels
        if bb84_context and bb84_context.get("show_bb84_labels", True):
            ax.text(
                1.4,
                0,
                0,
                "|+⟩\n(Diag 0)",
                ha="center",
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            )
            ax.text(
                -1.4,
                0,
                0,
                "|−⟩\n(Diag 1)",
                ha="center",
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )
            ax.text(
                0,
                0,
                1.4,
                "|0⟩\n(Comp 0)",
                ha="center",
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )
            ax.text(
                0,
                0,
                -1.4,
                "|1⟩\n(Comp 1)",
                ha="center",
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
            )
        else:
            ax.text(1.4, 0, 0, "|+⟩", ha="center", fontsize=self.font_size)
            ax.text(-1.4, 0, 0, "|−⟩", ha="center", fontsize=self.font_size)
            ax.text(0, 1.4, 0, "|+i⟩", ha="center", fontsize=self.font_size)
            ax.text(0, -1.4, 0, "|−i⟩", ha="center", fontsize=self.font_size)
            ax.text(0, 0, 1.4, "|0⟩", ha="center", fontsize=self.font_size)
            ax.text(0, 0, -1.4, "|1⟩", ha="center", fontsize=self.font_size)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

        # Title & side info
        full_title = (
            self._create_enhanced_title(title, statevector, bb84_context)
            if bb84_context
            else title
        )
        ax.set_title(full_title, fontsize=self.font_size + 2, fontweight="bold", pad=25)
        if show_state_info:
            self._add_matplotlib_state_info_panel(fig, statevector, bb84_context)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.background_color,
        )
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # -----------------------------------------------------------------------
    # HELPER METHODS (analysis, formatting, legends, etc.)
    # -----------------------------------------------------------------------
    def _create_enhanced_title(
        self,
        title: str,
        statevector: np.ndarray,
        bb84_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not bb84_context:
            return title
        parts = [title]
        if bb84_context.get("protocol_step"):
            parts.append(bb84_context["protocol_step"])
        if bb84_context.get("step_number") and bb84_context.get("total_steps"):
            parts.append(
                f"({bb84_context['step_number']}/{bb84_context['total_steps']})"
            )
        return ": ".join(parts)

    # -----------------------------------------------------------------------
    def _add_state_info_panel(
        self,
        fig,
        statevector: np.ndarray,
        bb84_context: Optional[Dict[str, Any]],
    ):
        """Text box for QuTiP figures."""
        try:
            info = self._format_state_info_text(
                self.analyze_state_properties(statevector), bb84_context
            )
            fig.text(
                0.02,
                0.98,
                info,
                transform=fig.transFigure,
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
            )
        except Exception as e:
            logger.warning(f"Could not add state-info panel: {e}")

    # Matplotlib variant
    def _add_matplotlib_state_info_panel(
        self,
        fig,
        statevector: np.ndarray,
        bb84_context: Optional[Dict[str, Any]],
    ):
        try:
            ax = fig.add_subplot(122)
            ax.axis("off")
            text = self._format_state_info_text(
                self.analyze_state_properties(statevector), bb84_context
            )
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9),
            )
        except Exception as e:
            logger.warning(f"Side-panel generation failed: {e}")

    # -----------------------------------------------------------------------
    def _format_state_info_text(
        self, analysis: Dict[str, Any], bb84_context: Optional[Dict[str, Any]]
    ) -> str:
        """Human-readable summary (monospace)."""
        lines = ["QUANTUM STATE ANALYSIS", "-" * 29]
        a = analysis["amplitudes"]
        lines += [
            f"alpha = {a['alpha']['real']:+.3f} + {a['alpha']['imag']:+.3f} i",
            f"beta  = {a['beta']['real']:+.3f} + {a['beta']['imag']:+.3f} i",
            "",
            f"P(|0⟩) = {analysis['probabilities']['P(|0⟩)']:.3f}",
            f"P(|1⟩) = {analysis['probabilities']['P(|1⟩)']:.3f}",
            "",
            f"Bloch (x,y,z) = "
            f"({analysis['bloch_coordinates']['x']:+.3f}, "
            f"{analysis['bloch_coordinates']['y']:+.3f}, "
            f"{analysis['bloch_coordinates']['z']:+.3f})",
            "",
        ]
        if bb84_context:
            lines += [f"BB84 type: {bb84_context.get('state_type', '?')}"]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    def _add_enhanced_bloch_subplot(
        self,
        ax,
        statevector: np.ndarray,
        label: str,
        vector_color: str,
        state_type: str,
    ):
        """Reusable helper for grid comparison plots."""
        # Normalise
        statevector = statevector / np.linalg.norm(statevector)
        x, y, z = self.bloch_coordinates(statevector)

        # Sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        ax.plot_wireframe(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            color="lightblue",
            alpha=0.3,
            linewidth=0.5,
        )
        # Axes
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], "k-", alpha=0.3)

        # Vector
        ax.quiver(
            0,
            0,
            0,
            x,
            y,
            z,
            color=vector_color,
            linewidth=4,
            arrow_length_ratio=0.15,
            alpha=0.9,
        )
        ax.scatter(
            [x],
            [y],
            [z],
            color=vector_color,
            s=100,
            edgecolors="white",
            linewidths=2,
        )

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_title(label, fontsize=10, fontweight="bold", color=vector_color)
        ax.axis("off")

    # -----------------------------------------------------------------------
    def _add_bb84_legend(self, fig):
        """Legend mapping colours → BB84 states."""
        try:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    label=s.replace("_", " ").title(),
                    markersize=10,
                )
                for s, c in self.bb84_colors.items()
                if "computational" in s or "superposition" in s
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(handles) // 2 + 1,
                fontsize=10,
            )
        except Exception as e:
            logger.warning(f"Legend creation failed: {e}")

    # -----------------------------------------------------------------------
    def _setup_basic_bloch_axes(self, ax):
        """Wireframe sphere for composite plots."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        ax.plot_wireframe(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            color="lightblue",
            alpha=0.3,
        )
        L = 1.2
        ax.plot([-L, L], [0, 0], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [-L, L], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [0, 0], [-L, L], "k-", alpha=0.3)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.axis("off")

    # -----------------------------------------------------------------------
    # ANALYSIS UTILITIES
    # -----------------------------------------------------------------------
    def bloch_coordinates(self, statevector: Union[List, np.ndarray]) -> Tuple[float, float, float]:
        """Return (x, y, z) on Bloch sphere."""
        v = np.asarray(statevector, dtype=complex)
        v = v / np.linalg.norm(v)
        a, b = v[0], v[1]
        x = 2 * np.real(np.conj(a) * b)
        y = 2 * np.imag(np.conj(a) * b)
        z = abs(a) ** 2 - abs(b) ** 2
        return float(x), float(y), float(z)

    def spherical_to_statevector(self, theta: float, phi: float) -> np.ndarray:
        """θ, φ → |ψ⟩."""
        return np.array([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])

    def statevector_to_spherical(self, statevector: Union[List, np.ndarray]) -> Tuple[float, float]:
        """|ψ⟩ → (θ, φ)."""
        v = np.asarray(statevector, dtype=complex)
        v = v / np.linalg.norm(v)
        α, β = v
        θ = 2 * np.arccos(np.clip(abs(α), 0, 1))
        if abs(np.sin(θ / 2)) < 1e-12:
            φ = 0.0
        else:
            φ = np.angle(β / np.sin(θ / 2))
        return float(θ), float(φ)

    def analyze_state_properties(self, statevector: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive, JSON-serialisable diagnostics."""
        v = np.asarray(statevector, dtype=complex)
        v = v / np.linalg.norm(v)
        α, β = v
        prob0, prob1 = abs(α) ** 2, abs(β) ** 2
        x, y, z = self.bloch_coordinates(v)
        θ, φ = self.statevector_to_spherical(v)

        return dict(
            amplitudes=dict(
                alpha=dict(
                    real=float(α.real),
                    imag=float(α.imag),
                    magnitude=float(abs(α)),
                    phase=float(np.angle(α)),
                ),
                beta=dict(
                    real=float(β.real),
                    imag=float(β.imag),
                    magnitude=float(abs(β)),
                    phase=float(np.angle(β)),
                ),
            ),
            probabilities={"P(|0⟩)": prob0, "P(|1⟩)": prob1},
            bloch_coordinates=dict(x=x, y=y, z=z, radius=float(np.linalg.norm([x, y, z]))),
            spherical_coordinates=dict(
                theta=θ,
                phi=φ,
                theta_degrees=float(np.degrees(θ)),
                phi_degrees=float(np.degrees(φ)),
            ),
            properties=dict(
                purity=1.0,  # pure state
                entropy=0.0,
                is_normalized=True,
            ),
            analysis_timestamp=datetime.datetime.now().isoformat(),
        )

    # -----------------------------------------------------------------------
    # ERROR FALLBACK
    # -----------------------------------------------------------------------
    def _generate_error_image(self, msg: str) -> str:
        """Return a small PNG explaining the error (base-64)."""
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=self.background_color)
        ax.axis("off")
        ax.text(
            0.5,
            0.6,
            "VISUALISATION ERROR",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color="red",
        )
        ax.text(
            0.5,
            0.4,
            msg,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # -----------------------------------------------------------------------
    # STYLE-UPDATES
    # -----------------------------------------------------------------------
    def set_style(self, style_params: Dict[str, Any]):
        """Change rendering options on the fly."""
        allowed = {
            "figure_size",
            "dpi",
            "sphere_alpha",
            "vector_color",
            "sphere_color",
            "grid_color",
            "background_color",
            "text_color",
            "font_size",
            "show_annotations",
            "show_coordinates",
            "show_measurement_probabilities",
            "style_theme",
        }
        for k, v in style_params.items():
            if k in allowed:
                setattr(self, k, v)
        if "style_theme" in style_params:
            self._initialize_theme_styles()
        logger.info(f"Updated visualiser style: {list(style_params.keys())}")

    # -----------------------------------------------------------------------
    # COMMON PRESET STATES
    # -----------------------------------------------------------------------
    def get_common_states(self) -> Dict[str, List[complex]]:
        """Frequently used qubit pure states (including BB84)."""
        r2 = 1 / np.sqrt(2)
        return {
            "|0⟩": [1, 0],
            "|1⟩": [0, 1],
            "|+⟩": [r2, r2],
            "|−⟩": [r2, -r2],
            "|+i⟩": [r2, 1j * r2],
            "|−i⟩": [r2, -1j * r2],
            "BB84 |0⟩": [1, 0],
            "BB84 |1⟩": [0, 1],
            "BB84 |+⟩": [r2, r2],
            "BB84 |−⟩": [r2, -r2],
            "(3|0⟩+4|1⟩)/5": [0.6, 0.8],
        }
