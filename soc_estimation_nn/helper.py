import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from dataclasses import dataclass

@dataclass
class XYData:
    x_data: list
    y_data: list
    label: str
    secondary_y: bool = False
    color: str = None
    linestyle: str = '-'
    marker: str = ''


def plot(
    xy_data: list[dict],
    x_label: str,
    y_label: str,
    y2_label: str = '',
    title: str = '',
    show_plt: bool = True
) -> tuple[Figure, list[Axes]]:

    grid_opacity = 0.35
    title_font_size = 16

    fig, primary_ax = plt.subplots()

    plot_secondary_y = any(map(lambda d: d.get('secondary_y') is True, xy_data))
    if plot_secondary_y:
        secondary_ax = primary_ax.twinx()

    for index, data in enumerate(xy_data):
        d = XYData(**data)
        ax = secondary_ax if d.secondary_y else primary_ax
        ax.plot(
            d.x_data, 
            d.y_data, 
            label=d.label, 
            c=d.color if d.color else f'C{index}', 
            linestyle=d.linestyle, 
            marker=d.marker
        )

    primary_ax.set_xlabel(x_label)
    primary_ax.set_ylabel(y_label)
    primary_ax.grid(alpha=grid_opacity, linestyle='-')
    
    legend_handles, legend_labels = primary_ax.get_legend_handles_labels()

    if plot_secondary_y:
        secondary_ax.set_ylabel(y2_label)
        secondary_ax.grid(alpha=grid_opacity, linestyle=':')

        secondary_ax_handles, secondary_ax_labels = secondary_ax.get_legend_handles_labels()
        legend_handles += secondary_ax_handles
        legend_labels += secondary_ax_labels

        primary_ax.legend(
            legend_handles, 
            legend_labels, 
            loc='center', 
            bbox_to_anchor=(0.5, 1.1), 
            draggable=True, 
            ncol=min(len(xy_data), 2)
        )

    else:
        plt.legend(
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

    axes = [primary_ax, secondary_ax] if plot_secondary_y else [primary_ax]
    return fig, axes

