import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class XYData:
    """
    Interface defining plot settings for a single trace/series using matplotlib.pyplot.

    Parameters
    ----------
    x_data : list
        X-axis data
    y_data : list
        Y-axis data
    label : str
        Label for plot legend
    plot_num : int, optional
        Index of subplot, by default 1
    color : str, optional
        Color of line, by default None
    linestyle : str, optional
        Style of line, by default '-'
    marker : str, optional
        Style of marker, by default ''

    Attributes
    ----------
    x_data : list
        X-axis data
    y_data : list
        Y-axis data
    label : str
        Label for plot legend
    plot_num : int
        Index of subplot
    color : str
        Color of line
    linestyle : str
        Style of line
    marker : str
        Style of marker

    Examples
    --------
    Define plot settings for a horizontal dashed, black line.
    
    >>> data = XYData(x_data=[1,2,3,4,5], y_data=[2,2,2,2,2], label='horizontal line', color='black', linestyle='--')
    >>> data.color
    black
    >>> data.x_data
    [1, 2, 3, 4, 5]
    """

    x_data: list
    y_data: list
    label: str
    plot_num: int = 1
    color: str = None
    linestyle: str = '-'
    marker: str = ''

    def __repr__(self) -> str:
        return f'XYData("{self.label}")'


def plot(
    xy_data: list[dict],
    x_label: str = '',
    y_label: str | dict = {},
    title: str = '',
    fig_size: tuple[float, float] = (10.0, 7.0),
    show_plt: bool = True,
) -> tuple[Figure, list[Axes]]:
    """
    Creates and formats a single scatter plot or subplots. Supports multiple traces in the same plot.

    Parameters
    ----------
    xy_data : list[dict]
        List of dictionaries with key value pairs to initialize XYData instance
    x_label : str
        X-axis label
    y_label : str | dict
        Y-axis label. If subplots are used, provide dictionary with key indicating plot_num
        and value for Y-axis label.
    title : str, optional
        Title of plot, by default ''
    fig_size : tuple[float, float], optional
        Tuple of figure size in width and height, by default (10.0, 7.0)
    show_plt : bool, optional
        Boolean on whether to show plot, by default True

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and axes objects

    Examples
    --------
    Create plot for a single line.

    >>> fig, axes = plot(xy_data=[{'x_data': [1,2,3,4,5], 'y_data': [5,4,3,2,1], 'label': 'Line 1'}], x_label='X', y_label='Y', title='Foo', show_plt=False)
    >>> fig
    Figure(1000x700)
    >>> axes
    [<Axes: xlabel='X', ylabel='Y'>]
    """

    grid_opacity = 0.35
    title_font_size = 16
    
    xy_data = list(map(lambda d: XYData(**d), xy_data))
    num_of_plots = max(map(lambda d: d.plot_num, xy_data))
    
    fig, axes = plt.subplots(num_of_plots, 1, figsize=fig_size)
    if num_of_plots == 1:
        axes = [axes]
        y_label = {1: y_label} if type(y_label) is str else y_label

    for index, data in enumerate(xy_data):
        ax = axes[data.plot_num - 1]
        ax.plot(
            data.x_data, 
            data.y_data, 
            label=data.label, 
            c=data.color if data.color else f'C{index}', 
            linestyle=data.linestyle, 
            marker=data.marker
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label.get(data.plot_num))
        ax.grid(alpha=grid_opacity, linestyle='-')
    
    legend_handles = sum([ax.get_legend_handles_labels()[0] for ax in axes], [])
    legend_labels = sum([ax.get_legend_handles_labels()[-1] for ax in axes], [])
    axes[0].legend(
        legend_handles,
        legend_labels,
        loc='upper left', 
        bbox_to_anchor=(1, 1), 
        draggable=True, 
    )

    fig.suptitle(title, fontsize=title_font_size)
    plt.tight_layout()
       
    if show_plt:
        plt.show()

    return fig, axes


def round_to_sig_fig(
    x: float,
    sig_fig: int
) -> float:
    """
    Round a float to a specified number of significant figures.

    Parameters
    ----------
    x : float
        Number to be rounded
    sig_fig : int
        Number of significant figures to be rounded to

    Returns
    -------
    float
        Number rounded to the specified number of significant figures.

    Raises
    ------
    TypeError
        If sig_fig is not an integer
    """

    if not isinstance(sig_fig, int):
        raise TypeError(f'sig_fig need to be of type int.') 

    return float(np.format_float_positional(x, precision=sig_fig, fractional=False, trim='k'))

