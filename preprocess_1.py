import datetime as dt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


def date_handel(d, cols):

    # converting to datatime datatype

    d[cols] = pd.to_datetime(d[cols], infer_datetime_format=True)
    d['date_year'] = d[cols].dt.year
    d['date_month_no'] = d[cols].dt.month
    d['date_day'] = d[cols].dt.day
    # we can also Extract Month Name, Day of Week-Name ,  Extract Day of Week
    return d


def time_handel(d2, cols2):

    # converting to datatime datatype
    d2[cols2] = pd.to_datetime(d2[cols2])
    # all dep_time in minute
    d2['new_dep_time'] = d2[cols2].dt.minute + d2[cols2].dt.hour*100


def time_handel2(d3, cols3):

    # converting to datatime datatype
    d3[cols3] = pd.to_datetime(d3[cols3])
    # all arr_time in minute
    d3['new_arr_time'] = d3[cols3].dt.minute + d3[cols3].dt.hour*100


def time_taken(d4, time1, time2):
    d4['new_time_taken'] = abs(d4[time1]-d4[time2])


def feature_scaling(x):
    """ Standardisation """
    standardisation = preprocessing.StandardScaler()
    # Scaled feature
    x_after_standardisation = standardisation.fit_transform(x)
    return x_after_standardisation


