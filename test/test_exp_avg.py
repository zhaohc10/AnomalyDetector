import csv
import time
import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
import re
import sys


from lib.exp_avg import (mean_stddev, median_deviation, trend_statistic, calculate_trend, anomaly_trend)


class Runner:

    def __init__(self, filename):
        self.filename = filename
        self.path = "../data/" + filename
        self.time_series = []
        self.read_file()
        self.anomalies = []
        self.results = []
        self.result_t = []
        self.result_t_reduced = []
        self.series = self.__set_series()
        self.algorithms = [mean_stddev, median_deviation]
        #self.trends = [trend_statistic]
        self.results_true = []
        self.daily_trends = []

    def read_file(self):
        with open(self.path, 'rb') as csv_file:
            print(self.path)
            reader = csv.reader(csv_file)
            reader.next()  # skip line 'timestamp' => 'value'
            self.__create_time_series(reader)

    def get_results(self):
        return self.results

    def run(self, algo, param):
        self.results = [self.run_algorithm(i, algo, param) for i in range(len(self.time_series))]

    def run_algorithm(self, index, algo, param):
        series_part = self.__set_series_part(index)
        value = self.series[index]
        check = self.algorithms[algo](value, series_part, param)
        if check:
            self.add_to_anomalies(index)
            # print "anomaly fonund! value: {0}, index: {1}".format(value, index)
        return check

    #algorithm that runs a trend statistics for data and searches for anomalies
    def run_trend_algorithm(self):
        self.result_t = trend_statistic(self.series, self.time_series)
        for i in range(len(self.result_t)):
            if ~np.isnan(self.result_t[i]): #reducing the result data to values that are numbers
                self.result_t_reduced.append(self.result_t[i])

    def calculate_daily_trend(self, param):
        # 1 value for each 5 min = > 12 values/hour = > 12*24/day
        day_length = 12*24
        series_length = len(self.time_series)
        days = series_length / day_length
        print ("Calculating trend for {0} days in {1} series long. each day {2} long".format(days, series_length, day_length))
        for d in range(days):
            print ("day: " + str(d+1))
            self.daily_trends.append(calculate_trend(self.series.values[d:d+day_length]))
        print ("daily trends: ")
        print ([x for x,i in self.daily_trends])
        results = anomaly_trend([x for x,i in self.daily_trends], param)
        tmp = [0 for i in range(len(self.daily_trends))]
        for i in results:
            tmp[i] = self.daily_trends[i][0]

        values_to_plot = []
        for d in range(days):
            values_to_plot.append(self.daily_trends[d][0])
        plt.plot(values_to_plot, 'ro', tmp, 'b^')
        plt.show()


    def add_to_anomalies(self, index):
        self.anomalies.append(self.time_series[index][1])

    def define_range(self, i):
        if i<50:
            return self.time_series[:2 * i + 1]
        if i+50 > len(self.time_series):
            return self.time_series[len(self.time_series) - 2 * (len(self.time_series) - i) - 1:]
        return self.time_series[i - 50:i + 51]

    def update_results(self):
        for i in range(len(self.results)):
            if self.results[i]:
                self.results[i] = self.series[i]

    def plot_with_anomalies(self):

        plt.plot(self.series, 'ro', self.results, 'b^')
        plt.show()


    def add_anomalies_to_results(self):
        for i in range(len(self.results)):
            if self.results[i]:
                self.results_true.append(self.results[i])

    def __create_time_series(self, reader):
      for row in reader:
          s = time.mktime(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple())
          self.time_series.append([s, float(row[1])])

    def __set_series_part(self, index):
          return pandas.Series(x[1] for x in self.define_range(index))

    def __set_series(self):
        return pandas.Series([x[1] for x in self.time_series])


def set_param_for_file(filename):
    if filename == 'ec2_cpu_utilization_5f5533.csv':
        param = [2.5, 6]
    elif re.match('exchange-\d.*\.csv', filename):
        param = [2.5, 1.5]
    elif re.match('Twitter.*\.csv', filename):
        param = [3, 1]
    else:
        param = [3, 6]
    return param


def create_fusion_results():
    fusion_results = list(runners[0].results)
    for j in range(len(runner.algorithms) - 1):
        for i in range(len(runners[j+1].results)):
            if runners[j+1].results[i]:
                fusion_results[i] = runners[j+1].results[i]
    for i in range(len(runner.result_t)): #fusion of the trend algorithm results, which runs independently from others
        if ~np.isnan(runner.result_t[i]):
            fusion_results[i] = runner.result_t[i]
    return fusion_results


def do_the_staff(i, plot_results):
    runners.append(Runner(filename))
    runners[i].run(i, set_param_for_file(filename)[i])
    runners[i].update_results()
    if plot_results:
        runners[i].plot_with_anomalies()
    runners[i].add_anomalies_to_results()

if __name__ == '__main__':

    filename = 'ec2_cpu_utilization_5f5533.csv'

    try:
        if sys.argv[1]:
            filename = sys.argv[1]
    except:
        print ("default filename chosen")


    try:
        if sys.argv[2]:
            plot_results = sys.argv[1]
    except:
        plot_results = 0

    runner = Runner(filename)
    runners = []

    for i in range(len(runner.algorithms)):
        do_the_staff(i, plot_results)
        print ("number of anomalies: " + str(len(runners[i].anomalies)))
        print ("length of input data: " + str(len(runners[i].time_series)))

    runner.run_trend_algorithm()

    print ("trend algorithm")
    print ("number of anomalies: " + str(len(runner.result_t_reduced)))
    print ("length of input data: " + str(len(runner.time_series)))

    fusion = create_fusion_results()
    fusion_anomalies = [x for x in fusion if x]
    print ("total number of anomalies: " + str(len(fusion_anomalies)))


    print ("calculate daily trends: ")
    runner.calculate_daily_trend(2)