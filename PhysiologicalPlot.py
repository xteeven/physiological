# %%
from os import listdir
from os.path import isfile, join, dirname, splitext, abspath
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psignifit as ps
from matplotlib.ticker import ScalarFormatter 
from sys import getsizeof, argv
from klepto.archives import file_archive
# %%
db = file_archive("User_0_Modellogistic.mde")