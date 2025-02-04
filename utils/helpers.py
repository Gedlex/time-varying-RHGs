'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
import matplotlib.figure as figure

def adjust_margins(fig : figure.Figure, width=None, height=None, top=None, bottom=None, wspace=None, hspace=None, wshift=None, pad=0.5, textwidth=5.90552) -> figure.Figure:
    # Change layout engine for export
    if (fig.get_layout_engine() is not None):
        fig.set_layout_engine(None)

    # Get figure size
    width = fig.get_figwidth() if width is None else width
    height = fig.get_figheight() if height is None else height

    # Check width
    if textwidth < width:
        raise ValueError('The figure width specified is too large for the specified textwidth.')

    # Compute tight margins
    fig.tight_layout(pad=pad)
    parms = fig.subplotpars

    # Get margins
    top_inch = (1 - parms.top) * fig.get_figheight() if top is None else top
    bottom_inch = parms.bottom * fig.get_figheight() if bottom is None else bottom
    wspace = parms.wspace * fig.get_figwidth() if wspace is None else wspace
    hspace = parms.hspace * fig.get_figheight() if hspace is None else hspace

    # Compute adjusted figure height
    adjusted_height = height + top_inch + bottom_inch

    # Resize figure
    fig.set_size_inches(textwidth, adjusted_height)

    # Compute adjusted margins
    margin = 1 - (width/textwidth)
    left = margin/2 * (1 + wshift) if wshift is not None else margin/2
    right = (1 - margin/2 * (1 - wshift)) if wshift is not None else (1 - margin/2)

    # Update margins
    fig.subplots_adjust(left = left,
                        right = right,
                        top = 1 - top_inch/adjusted_height,
                        bottom = bottom_inch/adjusted_height,
                        wspace = wspace/adjusted_height,
                        hspace = hspace/adjusted_height)

    # Return modified figure
    return fig