import numpy as np

def calc_mean_hist(hist_vals, hist_edges):
    # Calculate bin midpoints
    bin_midpoints = (hist_edges[:-1] + hist_edges[1:]) / 2
    # Calculate mean
    mean = np.average(bin_midpoints, weights=hist_vals)
    return mean

def calc_mode_hist(hist_vals, hist_edges):
    # Find the index of the bin with the highest value (mode)
    mode_index = np.argmax(hist_vals)
    # Calculate the midpoint of the mode bin
    mode = (hist_edges[mode_index] + hist_edges[mode_index + 1]) / 2
    return mode