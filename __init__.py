__version__ = "0.1"
__author__ = ["Thomas Berger"]
__license__ = "MIT"

import get_spectra_data as get_data
import preprocess_baseline_methods as basecorrecter
import preprocess_smoothing_methods as smoother
import preprocess_normalization_methods as normalizer
import preprocess_cropping_methods as cropper
import pandas as pd
import numpy as np