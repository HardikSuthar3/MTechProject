import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split

data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/Feature/ClassificationFeature.mat")
X = data['X']
Y = data['Y']

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.6)
print(X_test.shape)


