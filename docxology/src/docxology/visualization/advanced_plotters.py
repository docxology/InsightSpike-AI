"""Advanced Visualization and Animation Generators.

Provides enhanced visualization types including:
- Pie charts and donut charts
- Radar/spider charts
- Treemap visualizations  
- Network/dependency graphs
- Animated visualizations (GIF/MP4)
- Sunburst charts
- Sankey diagrams
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedPlotter:
    """Advanced visualization generator with enhanced chart types and animations.
    
    Provides pie charts, radar charts, treemaps, network graphs, 
    sunburst charts, and animated visualizations.
    
    Example:
        >>> plotter = AdvancedPlotter()
        >>> plotter.pie_chart({"A": 30, "B": 50, "C": 20}, output="pie.png")
        >>> plotter.animate_discovery(methods, output="discovery.gif")
    """
    
    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 150,
    ) -> None:
        """Initialize the advanced plotter.
        
        Args:
            style: Matplotlib style to use.
            figsize: Default figure size (width, height).
            dpi: Default DPI for saved figures.
        """
        self.figsize = figsize
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except Exception:
            logger.warning(f"Style '{style}' not available, using default")
    
    # ──────────────────────────────────────────────────────────────────────────
    # PIE CHARTS
    # ──────────────────────────────────────────────────────────────────────────
    
    def pie_chart(
        self,
        data: dict[str, float],
        title: str = "",
        output: Path | str | None = None,
        explode_largest: bool = True,
        show_percentage: bool = True,
        colors: list[str] | None = None,
    ) -> plt.Figure:
        """Create a pie chart.
        
        Args:
            data: Dictionary of label to value.
            title: Chart title.
            output: Optional output path.
            explode_largest: Whether to explode the largest slice.
            show_percentage: Show percentage on slices.
            colors: Optional custom colors.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Create explode array
        explode = [0] * len(values)
        if explode_largest and values:
            max_idx = values.index(max(values))
            explode[max_idx] = 0.05
        
        # Color palette
        if not colors:
            colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
        
        autopct = '%1.1f%%' if show_percentage else ''
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            explode=explode,
            colors=colors,
            autopct=autopct,
            shadow=True,
            startangle=90,
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def donut_chart(
        self,
        data: dict[str, float],
        title: str = "",
        center_text: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a donut chart (pie chart with center cut out).
        
        Args:
            data: Dictionary of label to value.
            title: Chart title.
            center_text: Text to display in center.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(data.keys())
        values = list(data.values())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(values)))
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white'),
        )
        
        # Add center circle for donut effect
        center_circle = plt.Circle((0, 0), 0.35, fc='white')
        ax.add_patch(center_circle)
        
        # Center text
        if center_text:
            ax.text(0, 0, center_text, ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    # ──────────────────────────────────────────────────────────────────────────
    # RADAR / SPIDER CHARTS
    # ──────────────────────────────────────────────────────────────────────────
    
    def radar_chart(
        self,
        data: dict[str, dict[str, float]],
        title: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a radar/spider chart for multi-dimensional comparison.
        
        Args:
            data: Dict of series name -> dict of category -> value.
                  Example: {"Module A": {"complexity": 0.8, "coverage": 0.9}}
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        # Get categories from first series
        first_series = list(data.values())[0]
        categories = list(first_series.keys())
        num_vars = len(categories)
        
        # Compute angle for each category
        angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        
        for idx, (name, values_dict) in enumerate(data.items()):
            values = [values_dict.get(cat, 0) for cat in categories]
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    # ──────────────────────────────────────────────────────────────────────────
    # TREEMAP
    # ──────────────────────────────────────────────────────────────────────────
    
    def treemap(
        self,
        data: dict[str, float],
        title: str = "",
        output: Path | str | None = None,
        colormap: str = "viridis",
    ) -> plt.Figure:
        """Create a treemap visualization.
        
        Args:
            data: Dictionary of label to value/size.
            title: Chart title.
            output: Optional output path.
            colormap: Matplotlib colormap.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by value for better layout
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]
        
        # Normalize sizes
        total = sum(sizes)
        if total == 0:
            total = 1
        normalized = [s / total for s in sizes]
        
        # Create squarified treemap layout
        rects = self._squarify(normalized, 0, 0, 1, 1)
        
        # Color mapping
        cmap = plt.cm.get_cmap(colormap)
        colors = cmap(np.linspace(0.2, 0.8, len(rects)))
        
        patches = []
        for i, (rect, label, size) in enumerate(zip(rects, labels, sizes)):
            x, y, w, h = rect
            # Scale to figure coordinates
            rect_patch = Rectangle((x, y), w, h, facecolor=colors[i], edgecolor='white', linewidth=2)
            patches.append(rect_patch)
            
            # Add label
            if w * h > 0.01:  # Only label if rect is big enough
                ax.text(
                    x + w/2, y + h/2,
                    f"{label}\n{size:.0f}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if i < len(rects)/2 else 'black'
                )
        
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def _squarify(self, sizes: list[float], x: float, y: float, w: float, h: float) -> list[tuple]:
        """Compute squarified treemap layout using simple algorithm."""
        if not sizes:
            return []
        
        if len(sizes) == 1:
            return [(x, y, w, h)]
        
        # Simple row-based layout
        rects = []
        total = sum(sizes)
        if total == 0:
            return [(x, y, w, h) for _ in sizes]
        
        if w >= h:
            # Lay out horizontally
            curr_x = x
            for size in sizes:
                rect_w = (size / total) * w
                rects.append((curr_x, y, rect_w, h))
                curr_x += rect_w
        else:
            # Lay out vertically
            curr_y = y
            for size in sizes:
                rect_h = (size / total) * h
                rects.append((x, curr_y, w, rect_h))
                curr_y += rect_h
        
        return rects
    
    # ──────────────────────────────────────────────────────────────────────────
    # NETWORK / DEPENDENCY GRAPHS
    # ──────────────────────────────────────────────────────────────────────────
    
    def network_graph(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[tuple[str, str, float]],
        title: str = "",
        output: Path | str | None = None,
        layout: str = "spring",
    ) -> plt.Figure:
        """Create a network/dependency graph visualization.
        
        Args:
            nodes: Dict of node_id -> {label, size, color, ...}.
            edges: List of (source, target, weight) tuples.
            title: Chart title.
            output: Optional output path.
            layout: Layout algorithm ("spring", "circular", "random").
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Compute positions
        positions = self._compute_layout(list(nodes.keys()), edges, layout)
        
        # Draw edges
        for source, target, weight in edges:
            if source in positions and target in positions:
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                alpha = min(0.8, 0.2 + weight * 0.3)
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color='gray', alpha=alpha, lw=weight)
                )
        
        # Draw nodes
        for node_id, attrs in nodes.items():
            if node_id in positions:
                x, y = positions[node_id]
                size = attrs.get('size', 300)
                color = attrs.get('color', 'steelblue')
                label = attrs.get('label', node_id)
                
                ax.scatter(x, y, s=size, c=[color], alpha=0.8, edgecolors='white', linewidth=2)
                ax.annotate(label, (x, y), fontsize=8, ha='center', va='center')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def _compute_layout(
        self,
        nodes: list[str],
        edges: list[tuple[str, str, float]],
        layout: str,
    ) -> dict[str, tuple[float, float]]:
        """Compute node positions using specified layout algorithm."""
        n = len(nodes)
        if n == 0:
            return {}
        
        positions = {}
        
        if layout == "circular":
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                positions[node] = (math.cos(angle), math.sin(angle))
        
        elif layout == "spring":
            # Simple force-directed layout
            np.random.seed(42)
            pos = np.random.rand(n, 2) * 2 - 1
            
            # Iterate to find equilibrium
            for _ in range(50):
                # Repulsion between all nodes
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 0.01
                        force = diff / (dist ** 2)
                        pos[i] += force * 0.01
                        pos[j] -= force * 0.01
                
                # Attraction along edges
                node_idx = {node: i for i, node in enumerate(nodes)}
                for src, tgt, weight in edges:
                    if src in node_idx and tgt in node_idx:
                        i, j = node_idx[src], node_idx[tgt]
                        diff = pos[j] - pos[i]
                        dist = np.linalg.norm(diff) + 0.01
                        force = diff * dist * weight * 0.001
                        pos[i] += force
                        pos[j] -= force
            
            for i, node in enumerate(nodes):
                positions[node] = (pos[i][0], pos[i][1])
        
        else:  # random
            np.random.seed(42)
            for node in nodes:
                positions[node] = (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1)
        
        return positions
    
    # ──────────────────────────────────────────────────────────────────────────
    # SUNBURST CHART
    # ──────────────────────────────────────────────────────────────────────────
    
    def sunburst(
        self,
        data: dict[str, dict[str, float]],
        title: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a sunburst chart for hierarchical data.
        
        Args:
            data: Dict of parent -> {child: value, ...}.
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(aspect='equal'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        # Outer ring - children
        all_children = []
        child_colors = []
        for i, (parent, children) in enumerate(data.items()):
            for child, value in children.items():
                all_children.append((child, value, colors[i]))
        
        if all_children:
            child_labels = [c[0] for c in all_children]
            child_values = [c[1] for c in all_children]
            child_colors = [c[2] for c in all_children]
            
            ax.pie(
                child_values,
                radius=1.0,
                labels=child_labels,
                colors=child_colors,
                wedgeprops=dict(width=0.3, edgecolor='white'),
                labeldistance=1.1,
                pctdistance=0.85,
                textprops={'fontsize': 8}
            )
        
        # Inner ring - parents
        parent_values = [sum(children.values()) for children in data.values()]
        if parent_values:
            ax.pie(
                parent_values,
                radius=0.7,
                labels=list(data.keys()),
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='white'),
                labeldistance=0.5,
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    # ──────────────────────────────────────────────────────────────────────────
    # ANIMATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    def animate_discovery(
        self,
        discovery_data: list[dict[str, Any]],
        title: str = "Method Discovery Progress",
        output: Path | str | None = None,
        fps: int = 2,
    ) -> animation.FuncAnimation | None:
        """Create an animated visualization of method discovery progress.
        
        Args:
            discovery_data: List of dicts with cumulative discovery stats.
                            Each dict should have 'step', 'classes', 'functions', 'modules'.
            title: Animation title.
            output: Output path for GIF.
            fps: Frames per second.
            
        Returns:
            Animation object.
        """
        if not discovery_data:
            logger.warning("No discovery data for animation")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        def animate(frame_idx):
            for ax in axes:
                ax.clear()
            
            frame = discovery_data[frame_idx]
            
            # Left: Cumulative bar chart
            ax1 = axes[0]
            categories = ['Classes', 'Functions']
            values = [frame.get('classes', 0), frame.get('functions', 0)]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax1.bar(categories, values, color=colors)
            ax1.set_ylim(0, max([d.get('classes', 0) + d.get('functions', 0) for d in discovery_data]) * 1.1)
            ax1.set_title(f"Discovery Step {frame_idx + 1}/{len(discovery_data)}")
            ax1.set_ylabel("Count")
            
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(int(value)), ha='center', fontsize=12, fontweight='bold')
            
            # Right: Progress line
            ax2 = axes[1]
            steps = list(range(1, frame_idx + 2))
            classes_history = [discovery_data[i].get('classes', 0) for i in range(frame_idx + 1)]
            functions_history = [discovery_data[i].get('functions', 0) for i in range(frame_idx + 1)]
            
            ax2.plot(steps, classes_history, 'o-', color='#3498db', label='Classes', linewidth=2)
            ax2.plot(steps, functions_history, 's-', color='#e74c3c', label='Functions', linewidth=2)
            ax2.set_xlim(0, len(discovery_data) + 1)
            ax2.set_ylim(0, max([d.get('classes', 0) for d in discovery_data]) * 1.1)
            ax2.set_xlabel("Discovery Step")
            ax2.set_ylabel("Cumulative Count")
            ax2.set_title("Discovery Timeline")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(discovery_data),
            interval=1000 // fps,
            repeat=True
        )
        
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as GIF
            if str(output).endswith('.gif'):
                anim.save(str(output), writer='pillow', fps=fps)
            else:
                anim.save(str(output), writer='ffmpeg', fps=fps)
            
            logger.info(f"Saved animation: {output}")
        
        plt.close(fig)
        return anim
    
    def animate_metrics(
        self,
        metrics_timeline: list[dict[str, float]],
        title: str = "Metrics Over Time",
        output: Path | str | None = None,
        fps: int = 3,
    ) -> animation.FuncAnimation | None:
        """Animate metrics changing over time.
        
        Args:
            metrics_timeline: List of dicts, each with metric values at a point in time.
            title: Animation title.
            output: Output path.
            fps: Frames per second.
            
        Returns:
            Animation object.
        """
        if not metrics_timeline:
            return None
        
        # Get all metric names
        all_metrics = set()
        for frame in metrics_timeline:
            all_metrics.update(frame.keys())
        all_metrics = sorted(all_metrics)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame_idx):
            ax.clear()
            
            # Create bars for current frame
            values = [metrics_timeline[frame_idx].get(m, 0) for m in all_metrics]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_metrics)))
            
            bars = ax.bar(all_metrics, values, color=colors)
            
            max_val = max(max(frame.values()) for frame in metrics_timeline if frame)
            ax.set_ylim(0, max_val * 1.2)
            ax.set_title(f"{title} - Frame {frame_idx + 1}/{len(metrics_timeline)}")
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                       f"{value:.1f}", ha='center', fontsize=9)
            
            plt.tight_layout()
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(metrics_timeline),
            interval=1000 // fps,
            repeat=True
        )
        
        if output:
            self._save_animation(anim, output, fps)
        
        plt.close(fig)
        return anim
    
    def animate_pipeline(
        self,
        stages: list[dict[str, Any]],
        title: str = "Pipeline Execution",
        output: Path | str | None = None,
        fps: int = 1,
    ) -> animation.FuncAnimation | None:
        """Animate pipeline stage execution.
        
        Args:
            stages: List of stage dicts with 'name', 'status', 'duration', 'progress'.
            title: Animation title.
            output: Output path.
            fps: Frames per second.
            
        Returns:
            Animation object.
        """
        if not stages:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        stage_names = [s['name'] for s in stages]
        n_stages = len(stage_names)
        
        def animate(frame_idx):
            ax.clear()
            
            # Draw all stages as boxes
            for i, stage in enumerate(stages):
                x = i * 1.5
                
                # Determine color based on completion
                if i < frame_idx:
                    color = '#2ecc71'  # Completed - green
                    status = '✓'
                elif i == frame_idx:
                    color = '#f39c12'  # In progress - orange
                    status = '⟳'
                else:
                    color = '#95a5a6'  # Pending - gray
                    status = '○'
                
                # Draw box
                rect = Rectangle((x, 0), 1.2, 0.8, facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                
                # Stage name
                ax.text(x + 0.6, 0.4, stage['name'], ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
                ax.text(x + 0.6, 0.1, status, ha='center', va='center',
                       fontsize=14, color='white')
                
                # Draw arrow to next stage
                if i < n_stages - 1:
                    ax.annotate('', xy=(x + 1.5, 0.4), xytext=(x + 1.2, 0.4),
                              arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
            
            ax.set_xlim(-0.5, n_stages * 1.5)
            ax.set_ylim(-0.3, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"{title} - Stage {min(frame_idx + 1, n_stages)}/{n_stages}",
                        fontsize=14, fontweight='bold')
        
        # Create frames for each stage plus one final
        n_frames = n_stages + 1
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=n_frames,
            interval=1000 // fps,
            repeat=True
        )
        
        if output:
            self._save_animation(anim, output, fps)
        
        plt.close(fig)
        return anim
    
    # ──────────────────────────────────────────────────────────────────────────
    # ADDITIONAL VISUALIZATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    def stacked_bar_chart(
        self,
        data: dict[str, dict[str, float]],
        title: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a stacked bar chart.
        
        Args:
            data: Dict of category -> {subcategory: value, ...}.
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        categories = list(data.keys())
        all_subcats = set()
        for cat_data in data.values():
            all_subcats.update(cat_data.keys())
        subcats = sorted(all_subcats)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(subcats)))
        
        bottom = np.zeros(len(categories))
        
        for i, subcat in enumerate(subcats):
            values = [data[cat].get(subcat, 0) for cat in categories]
            ax.bar(categories, values, bottom=bottom, label=subcat, color=colors[i])
            bottom += values
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def bubble_chart(
        self,
        data: list[dict[str, float]],
        x_key: str,
        y_key: str,
        size_key: str,
        label_key: str | None = None,
        title: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a bubble chart.
        
        Args:
            data: List of dicts with x, y, size, and optionally label values.
            x_key: Key for x-axis values.
            y_key: Key for y-axis values.
            size_key: Key for bubble size values.
            label_key: Optional key for labels.
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_vals = [d[x_key] for d in data]
        y_vals = [d[y_key] for d in data]
        sizes = [d[size_key] * 100 for d in data]  # Scale for visibility
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        
        scatter = ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.6, edgecolors='white', linewidth=2)
        
        if label_key:
            for i, d in enumerate(data):
                ax.annotate(d.get(label_key, ''), (x_vals[i], y_vals[i]),
                           fontsize=8, ha='center', va='center')
        
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def waterfall_chart(
        self,
        data: dict[str, float],
        title: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a waterfall chart showing cumulative effect.
        
        Args:
            data: Dict of label to delta value.
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(data.keys())
        values = list(data.values())
        
        cumulative = [0]
        for v in values:
            cumulative.append(cumulative[-1] + v)
        
        # Colors: green for positive, red for negative
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
        
        x = np.arange(len(labels))
        
        # Draw bars
        for i, (label, value) in enumerate(zip(labels, values)):
            bottom = cumulative[i]
            ax.bar(i, value, bottom=bottom, color=colors[i], edgecolor='white', linewidth=2)
            
            # Value label
            y_pos = bottom + value/2
            ax.text(i, y_pos, f'{value:+.0f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Add total bar
        ax.bar(len(labels), cumulative[-1], color='#3498db', edgecolor='white', linewidth=2)
        ax.text(len(labels), cumulative[-1]/2, f'{cumulative[-1]:.0f}', 
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax.set_xticks(list(range(len(labels) + 1)))
        ax.set_xticklabels(labels + ['Total'], rotation=45, ha='right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    # ──────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────────────────
    
    def _save(self, fig: plt.Figure, output: Path | str) -> None:
        """Save a figure to file."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output}")
        plt.close(fig)
    
    def _save_animation(self, anim: animation.FuncAnimation, output: Path | str, fps: int) -> None:
        """Save an animation to file."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if str(output).endswith('.gif'):
            anim.save(str(output), writer='pillow', fps=fps)
        else:
            try:
                anim.save(str(output), writer='ffmpeg', fps=fps)
            except Exception:
                # Fallback to pillow if ffmpeg not available
                gif_output = output.with_suffix('.gif')
                anim.save(str(gif_output), writer='pillow', fps=fps)
                logger.warning(f"ffmpeg not available, saved as GIF: {gif_output}")
        
        logger.info(f"Saved animation: {output}")
