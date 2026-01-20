"""Plotter for generating visualizations.

Provides matplotlib-based plotting utilities for performance charts,
distributions, comparisons, and timelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Plotter:
    """Generates matplotlib-based visualizations.
    
    Provides methods for creating performance charts, metric distributions,
    comparison plots, and timeline visualizations.
    
    Example:
        >>> plotter = Plotter()
        >>> plotter.bar_chart({"A": 10, "B": 15, "C": 8}, output="chart.png")
    """
    
    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple[int, int] = (10, 6),
        dpi: int = 150,
    ) -> None:
        """Initialize the plotter.
        
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
    
    def bar_chart(
        self,
        data: dict[str, float],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        output: Path | str | None = None,
        color: str = "steelblue",
    ) -> plt.Figure:
        """Create a bar chart.
        
        Args:
            data: Dictionary of label to value.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            output: Optional output path for saving.
            color: Bar color.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(labels, values, color=color)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Rotate labels if many bars
        if len(labels) > 5:
            plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def line_chart(
        self,
        data: dict[str, list[float]],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a line chart.
        
        Args:
            data: Dictionary of series name to values.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for name, values in data.items():
            ax.plot(values, label=name, marker="o", markersize=4)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def histogram(
        self,
        data: list[float],
        title: str = "",
        xlabel: str = "",
        bins: int = 20,
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a histogram.
        
        Args:
            data: List of values.
            title: Chart title.
            xlabel: X-axis label.
            bins: Number of bins.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(data, bins=bins, color="steelblue", edgecolor="white", alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        
        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(mean + std, color="orange", linestyle=":", alpha=0.7)
        ax.axvline(mean - std, color="orange", linestyle=":", alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def comparison_chart(
        self,
        baseline: dict[str, float],
        experiment: dict[str, float],
        title: str = "Comparison",
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a grouped bar chart comparing two sets.
        
        Args:
            baseline: Baseline values.
            experiment: Experiment values.
            title: Chart title.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get common keys
        keys = list(set(baseline.keys()) & set(experiment.keys()))
        x = np.arange(len(keys))
        width = 0.35
        
        baseline_vals = [baseline[k] for k in keys]
        experiment_vals = [experiment[k] for k in keys]
        
        ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="steelblue")
        ax.bar(x + width/2, experiment_vals, width, label="Experiment", color="coral")
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.legend()
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def heatmap(
        self,
        data: list[list[float]],
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        title: str = "",
        output: Path | str | None = None,
        cmap: str = "viridis",
    ) -> plt.Figure:
        """Create a heatmap.
        
        Args:
            data: 2D array of values.
            row_labels: Row labels.
            col_labels: Column labels.
            title: Chart title.
            output: Optional output path.
            cmap: Colormap name.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        data_array = np.array(data)
        im = ax.imshow(data_array, cmap=cmap, aspect="auto")
        
        if row_labels:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
        if col_labels:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if output:
            self._save(fig, output)
        
        return fig
    
    def performance_chart(
        self,
        stats: dict[str, Any],
        output: Path | str | None = None,
    ) -> plt.Figure:
        """Create a performance summary chart.
        
        Args:
            stats: Statistics dictionary with 'summary' key.
            output: Optional output path.
            
        Returns:
            Matplotlib figure.
        """
        summary = stats.get("summary", {})
        
        if not summary:
            logger.warning("No summary data to plot")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig
        
        # Extract metric means
        data = {}
        for key, value in summary.items():
            if isinstance(value, dict) and "mean" in value:
                data[key] = value["mean"]
            elif isinstance(value, (int, float)):
                data[key] = value
        
        return self.bar_chart(
            data,
            title="Performance Summary",
            ylabel="Value",
            output=output,
        )
    
    def _save(self, fig: plt.Figure, output: Path | str) -> None:
        """Save a figure to file.
        
        Args:
            fig: Matplotlib figure.
            output: Output path.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure: {output}")
