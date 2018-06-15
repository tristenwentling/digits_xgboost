from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb
import pandas as pd
import mlflow as mlf


class DigitModel(object):
	"""XGboost model for MNIST digits (Kaggle format) as MLflow example
	"""

	def __init__(self, train_file="train.csv", test_file="test.csv", out_file="submission.csv"):
		""" Defines the initial settings for using the model
		"""
		self.training_f_name = train_file
		self.test_set_f_name = test_file
		self.save_out_f_name = out_file
		self.silent = 0  # 0=verbose 1=quiet; passed to xgb model
		self.num_round = 10  # num training rounds
		self.metrics = dict()

	def create_train_and_label_sets(self):
		""" Separates labels from data and formats each appropriately. uses these to create
			XGBoost DMatrix data object
		"""
		training_set = pd.read_csv(self.training_f_name)
		labels = training_set.label.values #.apply(lambda x: str(x)).values
		training_set.drop("label", inplace=True, axis=1)
		train_data = training_set.values
		self.dtrain = xgb.DMatrix(train_data, label=labels)
		print("training set loaded")

	def create_test_set(self):
		"""Loads the test set for model
		"""
		test_set = pd.read_csv(self.test_set_f_name)
		test_data = test_set.values
		self.dtest = xgb.DMatrix(test_data)
		print("test set loaded")

	def create_model(self):
		"""Creates parameters used for  XGboost model multiclassifier
		"""
		param = {'max_depth': 50,
				 'eta': 0.25,
				 'silent': self.silent,
				 'objective': 'multi:softmax',
				 'lambda': 1.1,
				 'gamma': 0.1,
				 'alpha': 0.1,
				 'num_class':10,
				 'tree_method': 'exact'}
		param['nthread'] = 10 # Using 1 per core
		param['eval_metric'] = 'merror'
		self.params = param  # storing parameters
		print("model prepared")

	def train_model(self, x_validate=False):
		""" Trains the XGBoost model with parameters set by create_model
		"""
		if x_validate:
			self.bst = xgb.cv(self.params.items(),
							  self.dtrain,
							  self.num_round)
		else:
			self.bst = xgb.train(self.params.items(),
								 self.dtrain,
								 self.num_round)
		print("training completed")

	def save_model(self, path="model01.model"):
		""" Add-on to save out model
		"""
		self.bst.save_model(path)

	def load_model(self, path="model01.model"):
		"""Add on to load previously used model
		"""
		self.bst = xgb.Booster({'nthread': 4})  # init model
		self.bst.load_model('model.bin')  # load data

	def make_predictions(self):
		"""Makes predictions and writes out results file using Pandas
		"""
		ypred = self.bst.predict(self.dtest)
		results = pd.DataFrame([[i for i in range(1, len(ypred) + 1)], list(ypred)]).T
		results.columns = ["ImageId","Label"]
		for i in results.columns:
			results[i] = results[i].astype("int")
		results.to_csv(self.save_out_f_name, index=False)
		self.results = results
		print("Ran predictions and wrote out file successfully")

	def plot_tree(self):
		"""Should use matplotlib to try and plot, not finished implementing
		"""
		print("Trying the graph")
		xgb.plot_importance(self.bst)
		xgb.plot_tree(self.bst, num_trees=2)

	def mlf_logging(self):
		"""Uses MLflow to monitor training metrics and track parameters
		"""
		for par in self.params.keys():
			mlf.log_param(par, self.params[par])
		for met in self.metrics.keys():
			mlf.log_metric(met, self.metrics[met])

	def run(self):
		"""Runs all methods to train, make predictions, and monitor metrics
		"""
		self.create_train_and_label_sets()
		self.create_model()
		self.train_model()
		self.create_test_set()
		self.make_predictions()
		self.mlf_logging()


if __name__ == "__main__":
	digit_model = DigitModel()
	digit_model.run()