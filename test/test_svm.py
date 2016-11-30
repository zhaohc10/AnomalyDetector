import numpy as np
import matplotlib.pyplot as plt
from lib.base import AnomalyDetector

# Generate train data
X_train = 0.3 * np.random.randn(100, 2)

# Generate some regular novel observations
X_test = 0.3 * np.random.randn(20, 2)

# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

param_dict = {'gamma': 0.1, 'kernel': 'rbf', 'nu': 0.1}
# fit the model
outlier_worker = AnomalyDetector(algo_name='svm', param_dict=param_dict)
outlier_worker.fit(X_train)

test = outlier_worker.predict(X_test)
out = outlier_worker.predict(X_outliers)
train = outlier_worker.predict(X_train)

print (outlier_worker.score(X_outliers))
print (outlier_worker.score(X_train[:20]))

plt.figure()
plt.subplot(121)
x = [r[0] for r in X_train]
y = [r[1] for r in X_train]
c = ['g' if r == 1 else 'r' for r in train]
plt.scatter(x, y, color=c)
plt.subplot(122)
x = [r[0] for r in X_test]
y = [r[1] for r in X_test]
c = ['g' if r == 1 else 'r' for r in test]
plt.scatter(x, y, color=c)
plt.show()