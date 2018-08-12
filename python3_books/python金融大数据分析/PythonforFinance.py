# -*- coding: utf-8 -*-
# package
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader.data as web

# %%
goog = web.DataReader('GooG', data_source='yahoo', start='3/14/2009', end='4/14/2014')