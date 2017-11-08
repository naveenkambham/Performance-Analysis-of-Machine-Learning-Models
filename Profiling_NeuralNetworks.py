import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from memory_profiler import profile
import cProfile
import psutil

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=12, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
@profile(precision=4)
def RunModel():
	# Scaling Input and only input layers and output layers
	# load dataset
	dataframe = pandas.read_csv("Finalinput.csv")
	# dataframe = pandas.read_csv(r'Synth_generated_K.csv')
	# split into input (X) and output (Y) variables
	X = dataframe.iloc[:, 2:14].values
	# print(X)

	Y = dataframe.iloc[:, [14, 15,16,17,18]].values
	# Y = dataframe.iloc[:, [14]].values
	# print(Y)
	# print(Y)
	seed = 7
	numpy.random.seed(seed)
	estimators = []
	# estimators.append(('standardize', StandardScaler()))
	# nb_epoch=50, batch_size=5
	estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=10, batch_size=100, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=15, random_state=seed)
	results = cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_squared_error')
	print(results)
	print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#Time taken to run
pr = cProfile.Profile()
pr.enable()
RunModel()
pr.disable()
pr.print_stats()

#CPU Usage
p = psutil.Process()
print(p.cpu_times())
print(p.cpu_percent(interval=0))
