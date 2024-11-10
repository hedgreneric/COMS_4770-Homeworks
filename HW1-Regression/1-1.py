import numpy as np 
from sklearn import linear_model
X = np.array([1, 0, 5, 2, 3]).reshape(-1,1)
Y = np.array([0, 2, 6, 2, 2])
"""fit y=ax+b"""
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)
print ("a = %s, b=%s, score=%s" % 
        ("{:.3f}".format(model.coef_[0]), 
        "{:.3f}".format(model.intercept_), 
        "{:.3f}".format(model.score(X, Y))))