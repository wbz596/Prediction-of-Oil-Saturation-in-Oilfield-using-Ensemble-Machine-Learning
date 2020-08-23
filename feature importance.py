from pandas import read_csv
from sklearn.tree import ExtraTreeRegressor
# load data
dataframe = read_csv('useformodel.csv')
array = dataframe.values

X = array[:,0:26]
Y = array[:,26]
# feature extraction
model = ExtraTreeRegressor(random_state=0)
model.fit(X, Y)
print(model.feature_importances_)