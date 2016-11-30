
from lib.svm import SvmDetector
from lib.hdbscan import HdbscanDetector
from lib.poisson import PoissonDetector


anomaly_detector_algorithms = {
    'svm': SvmDetector,
    'hdbscan': HdbscanDetector,
    'poisson': PoissonDetector
}

class AnomalyDetector(object):
    """
    Base Class for AnomalyDetector algorithm.
    """

    def __init__(self, algo_name, param_dict={}):
        """
        Initializer
        :param str class_name: extended class name.
        """
        self.algo = anomaly_detector_algorithms[algo_name]
        self.model = self.algo(param_dict)

    def fit(self, x):
        self.model.fit(x)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x):
        return self.model.score(x)