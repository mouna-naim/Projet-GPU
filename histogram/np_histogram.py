#!/usr/bin/env python

import os.path
import numpy as np

BINS_FILENAME = "bins.csv"
ARRAY_FILENAME = "array.csv"

if not os.path.exists(BINS_FILENAME):
    print(f"bins file {BINS_FILENAME} missing")
    exit(1)

if not os.path.exists(ARRAY_FILENAME):
    print(f"array file {ARRAY_FILENAME} missing")
    exit(1)

bins = np.genfromtxt(BINS_FILENAME)
array = np.genfromtxt(ARRAY_FILENAME)

histogram, _ = np.histogram(array, bins)

print(histogram)
