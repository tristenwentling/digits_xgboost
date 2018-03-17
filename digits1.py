import xgboost as xgb
import pandas as pd


class DigitModel(object):
	"""XGboost model for MNIST digits (Kaggle format)"""


	def __init__(self):
		self.training_f_name = "train.csv"
		self.test_set_f_name = "test.csv"
		self.save_out_f_name = "submission.csv"
		self.silent = 1 # 0=quiet 1=verbose
		self.num_round = 1000


	def create_train_and_label_sets(self):
		""" Separates labels from data and formats each appropriately."""
		training_set = pd.read_csv("train.csv")
		labels = training_set.label.values #.apply(lambda x: str(x)).values
		training_set.drop("label", inplace=True, axis=1)
		train_data = training_set.values
		self.dtrain = xgb.DMatrix(train_data, label=labels)
		print("training set loaded")


	def create_model(self):
		"""Creates and trains XGboost model multiclassifier"""
		param = {'max_depth': 120,
				 'eta': 1,
				 'silent': self.silent,
				 'objective': 'multi:softmax',
				 'lambda': 2,
				 'alpha': 1,
				 'num_class':10}
		param['nthread'] = 10 # Use 1 per core best?
		param['eval_metric'] = 'merror'
		self.params = param
		print("model prepared")
		

	def train_model(self):
		"""train the prepared model"""
		self.bst = xgb.train(self.params.items(), self.dtrain, self.num_round)
		print("training completed")


	def save_model(self, path="model01.model"):
		"""Add on to save out model"""
		self.bst.save_model(path)


	def load_model(self, path="model01.model"):
		"""Add on to load previously used model"""
		self.bst = xgb.Booster({'nthread': 4})  # init model
		self.bst.load_model('model.bin')  # load data


	def create_test_set(self):
		"""Creates test set for model"""
		test_set = pd.read_csv("test.csv")
		test_data = test_set.values
		self.dtest = xgb.DMatrix(test_data)
		print("test set created")


	def make_predictions(self):
		"""Makes predictions and writes out results using Pandas"""
		ypred = self.bst.predict(self.dtest)
		results = pd.DataFrame([[i for i in range(1, len(ypred) + 1)], list(ypred)]).T
		results.columns = ["ImageId","Label"]
		for i in results.columns:
			results[i] = results[i].astype("int")
		results.to_csv(self.save_out_f_name, index=False)
		print("Ran predictions and wrote out file successfully")


	def plot_tree(self):
		"""Should use matplotlib to try and plot, not finished implementing"""
		print("Trying the graph")
		xgb.plot_importance(self.bst)
		xgb.plot_tree(self.bst, num_trees=2)


def run_model():
	"""Builds digit_model class and runs everything"""
	digit_model = DigitModel()
	digit_model.create_train_and_label_sets()
	digit_model.create_model()
	digit_model.train_model()
	digit_model.create_test_set()
	digit_model.make_predictions()


if __name__ == "__main__":
	run_model()