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

def build_matrix(pred_labels, true_labels, n_classes = 5):
    assert pred_labels.shape == true_labels.shape
    hist = np.zeros((n_classes,n_classes))
    for i,t in enumerate(true_labels): #Get true labels
        if t == 5: continue #skip kaons for now
        p = pred_labels[i] #Get associated predicted label
        if p == -1 or t == -1 or np.isnan(p) or np.isnan(t): continue #Dummy labels
        hist[int(p),int(t)] += 1
    return hist