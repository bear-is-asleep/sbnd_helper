
## Flux
For flux extraction, use here - https://github.com/sjgardiner/stv-analysis-new/blob/eb182611dd92496e39d2ecd770056a73eab6e4b0/fv_flux_gsimple.C#L241

## XSec

## MicroBooNE


### Things i want to do
Ok so something i want to do is save histograms of the number of events, scaled by genweight, for different keys, with a given binning. I think the best way to do this is to make a function part of @parent.py that takes in an array of bins (bin edges), a key (variable), and category key, and returns an array of histograms, which has the histogram of the key in the bins specified, for each category key. For example, the category key could be event_type, but i can also imagine for particles we want to save the momentum per true particle. I want this function to return the array of histograms. I also want to add a function to @makedf1muX.py that makes a dictionary of histograms, in order to return this as a dataframe to be stored by the h5. I also want to add the ability to compute the uncertainties for the given binning.