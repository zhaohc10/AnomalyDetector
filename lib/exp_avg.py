import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *


def mean_stddev(value, series, param):
    mean = series.mean()
    std_dev = series.std()
    return abs(value - mean) > param * std_dev


def median_deviation(value, series, param):
    median = series.median()
    demedianed = np.abs(value - median)

    if demedianed == 0 or value == 0:
        return False

    test_statistic = value / demedianed
    if test_statistic < param:
        return True
    else:
        return False


def trend_statistic(value,series):
    smoothed_val = []
    prediction = []
    N = len(series)
    alpha = 0.3 # param to define how quickly you forget the past values
    beta = 0.5 # param to define how quickly you forget past slope
    std_window = 3 #define std window range

    smoothed_val.append(value[0])
    smoothed_slope = np.zeros((N, ))
    prediction.append(value[0])

    for i in range(1, N):
        #calculate new smoothed value from exponetial weighted average of current value and old value
        smoothed_val.append(alpha*value[i] + (1-alpha) * (smoothed_val[i-1] + smoothed_slope[i-1]))

        smoothed_slope[i] = (beta * (smoothed_val[i] - smoothed_val[i-1]) + (1-beta)*smoothed_slope[i-1])

        #Calculate prediction of value at current time, using only information from previous time.
        prediction.append(smoothed_val[i - 1] + smoothed_slope[i-1])

    #MEASURE PREDICTION ERROR

    #Calculate 'normal' limits of deviation based on variance earlier in the process
    #Use expotentially weighted standard deviation (so past data has less influence).
    #The shift by 1 is to exclude the difference of the current predicted and actual values from
    #the assessment of past prediction accurancy.

    smoothed_std_ts = pd.Series(
        pd.ewma(np.sqrt((value-prediction)**2), halflife=4), # halflife - how quickly std measure adjusts to change
        name='smoothed_std').shift(1)
    prediction_ts = pd.Series(prediction, name='predicted')
    results = pd.DataFrame(pd.Series(value),columns=['actual']).join(
        prediction_ts).join(smoothed_std_ts)
    results['lower']= results.predicted - std_window * results.smoothed_std
    results['upper']=results.predicted + std_window * results.smoothed_std

    counter = []
    for x in range(len(results['actual'])):
        if results.actual[x] < results.lower[x]:
            counter.append(x)
        elif results.actual[x] > results.upper[x]:
            counter.append(x)

    results['anomalies'] = results.actual[counter]

    results.plot() #plotin all result df columns
    plt.plot(results.anomalies, 'ro') #ploting nomalies
    show()
    return results.anomalies


def calculate_trend(series):
    x = np.arange(0,len(series))
    y = np.array(series)
    z = np.polyfit(x,y,1)
    print ("{0}x + {1}".format(*z))
    return z


def anomaly_trend(trends, param):
    results = []
    for i,t in enumerate(trends):
        tmp_trends = list(trends)
        tmp_trends.pop(i)
        mean = np.mean(tmp_trends)
        std = np.std(trends)
        if abs(t - mean) > param*std:
            results.append(i)
            print ("Day {0} is an anomaly trend. Value: {1}, mean: {2}, stddev: {3}".format(i+1, t, mean, std))
    return results
