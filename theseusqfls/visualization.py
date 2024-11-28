import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar

def imshow(ax, im, vmin=None, vmax=None, cbar_label=None, scalebar_mag=None, cmap='magma'):
    plot = ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    if cbar_label is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label(cbar_label)
    if scalebar_mag is not None:
        bar = ScaleBar(6.5 / scalebar_mag, 'um', box_alpha=0, color='white', location='lower left', length_fraction=0.2, height_fraction=0.025, font_properties={'size': 12}, pad=1)
        ax.add_artist(bar)
    return plot
