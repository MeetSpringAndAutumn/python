# import warnings
# import numpy as np
# import argparse
# from sklearn.datasets import load_boston
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, PolynomialFeatures
# from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_squared_error, r2_score
# from joblib import dump, load
# # import timeit
#
#
# def get_arguments():
#     parser = argparse.ArgumentParser(description='LinearRegression')
#     parser.add_argument('--data_name', type=str, default='boston', choices=('boston', 'california'),
#                         help='choose datasets')
#     parser.add_argument('--test_size', type=float, default=0.33, help='the proportion of test data')
#     parser.add_argument('--random_state', type=int, default=42, help='the random seed of dataset split')
#     parser.add_argument('--normalization', type=int, default=3, choices=(0, 1, 2, 3),
#                         help='select the type of data normalization,'
#                              '0: no normalization,'
#                              '1: rescale the data to [0, 1],'
#                              '2: rescale the data to [-1, 1],'
#                              '3: z-score normalization')
#     parser.add_argument('--Regression', type=int, default=2, choices=(1, 2, 3, 4, 5),
#                         help='select the type of Regression,'
#                              '1: normal equation of LinearRegression,'
#                              '2: SGD LinearRegression,'
#                              '3: Ridge Regression,'
#                              '4: Lasso Regression,'
#                              '5: Polynomial Regression')
#     parser.add_argument('--loss', type=int, default=1, choices=(1, 2),
#                         help='select the type of loss,'
#                              '1: R^2,'
#                              '2: MSE')
#     parser.add_argument('--max_iteration', type=int, default=1000, help='the max iteration of SGD')
#     parser.add_argument('--eta0', type=float, default=0.01, help='the learning rate of SGD')
#     parser.add_argument('--alpha', type=float, default=0.5,
#                         help='Intensity of regularization, must be a positive floating')
#     parser.add_argument('--degree', type=int, default=2, help='the degree of PolynomialFeatures')
#
#     args = parser.parse_args()
#     return args
#
#
# class MyLinearRegression:
#     def __init__(self, parser):
#         self.data_name = parser.data_name
#         self.test_size = parser.test_size
#         self.random_state = parser.random_state
#         self.normalization = parser.normalization
#         self.Regression = parser.Regression
#         self.loss = parser.loss
#         self.max_iter = parser.max_iteration
#         self.eta0 = parser.eta0
#         self.alpha = parser.alpha
#         self.degree = parser.degree
#
#     def load_dataset(self):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             if self.data_name == 'boston':
#                 dataset = load_boston()
#                 print("The boston datasets is loaded successfully!")
#             elif self.data_name == 'california':
#                 dataset = fetch_california_housing()
#                 print("The california datasets is loaded successfully!")
#             else:
#                 raise ValueError("Please choose 'boston' or 'california'")
#         description = dataset.DESCR
#         feature_names = dataset.feature_names
#         datas = dataset.data
#         target = dataset.target
#         print("The description of datasets is: ", end="")
#         print(description)
#         print("The feature names of datasets is: ", end="")
#         print(*feature_names)
#         if self.data_name == 'california':
#             target_names = dataset.target_names
#             print("The target names of datasets is: ", end="")
#             print(*target_names)
#         print("The shape of dataset is: ", end="")
#         print(datas.shape)
#
#         return datas, target
#
#     def split_dataset(self, X, y):
#         assert 0 < self.test_size < 1, "Please choose right test size between 0 and 1"
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=self.test_size, random_state=self.random_state)
#         return X_train, X_test, y_train, y_test
#
#     def normalize_dataset(self, X_train, X_test):
#         if self.normalization == 0:
#             # 不进行任何操作
#             X_train_normalization = X_train
#             X_test_normalization = X_test
#         elif self.normalization == 1:
#             # 将数值放缩到[0, 1]
#             min_max_scaler = MinMaxScaler()
#             X_train_normalization = min_max_scaler.fit_transform(X_train)
#             X_test_normalization = min_max_scaler.fit_transform(X_test)
#         elif self.normalization == 2:
#             # 将数值放缩到[-1, 1]
#             max_abs_scaler = MaxAbsScaler()
#             X_train_normalization = max_abs_scaler.fit_transform(X_train)
#             X_test_normalization = max_abs_scaler.fit_transform(X_test)
#         elif self.normalization == 3:
#             # 将数值进行z-score标准化
#             scaler = StandardScaler()
#             X_train_normalization = scaler.fit_transform(X_train)
#             X_test_normalization = scaler.fit_transform(X_test)
#         else:
#             raise ValueError("Please choose right normalization type", self.normalization)
#         return X_train_normalization, X_test_normalization
#
#     def regression(self, X_train, y_train):
#         if self.Regression == 1:
#             reg = LinearRegression().fit(X_train, y_train)
#             print("The score of LinearRegression is: {}".format(reg.score(X_train, y_train)))
#             print("The coefficient of LinearRegression is: {}".format(reg.coef_))
#             print("The intercept of LinearRegression is: {}".format(reg.intercept_))
#             dump(reg, 'LinearRegression.joblib')
#         elif self.Regression == 2:
#             reg = SGDRegressor(loss='squared_loss', fit_intercept=True, learning_rate='invscaling',
#                                eta0=self.eta0, max_iter=self.max_iter)
#             reg.fit(X_train, y_train)
#             print("The coefficient of SGDRegressor is: {}".format(reg.coef_))
#             print("The intercept of SGDRegressor is: {}".format(reg.intercept_))
#             dump(reg, 'SGDRegressor.joblib')
#         elif self.Regression == 3:
#             reg = Ridge(self.alpha)
#             reg.fit(X_train, y_train)
#             print("The coefficient of Ridge is: {}".format(reg.coef_))
#             print("The intercept of Ridge is: {}".format(reg.intercept_))
#             dump(reg, 'Ridge.joblib')
#         elif self.Regression == 4:
#             reg = Lasso(self.alpha)
#             reg.fit(X_train, y_train)
#             print("The coefficient of Lasso is: {}".format(reg.coef_))
#             print("The intercept of Lasso is: {}".format(reg.intercept_))
#             dump(reg, 'Lasso.joblib')
#         elif self.Regression == 5:
#             reg = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
#             reg.fit(X_train, y_train)
#             dump(reg, 'PolynomialFeatures.joblib')
#         else:
#             raise ValueError('Please choose right regression model', self.Regression)
#
#     def evaluate(self, X_train, X_test, y_train, y_test):
#         if self.Regression == 1:
#             reg = load('LinearRegression.joblib')
#             print("The pretrain model of LinearRegression is loaded successfully!")
#         elif self.Regression == 2:
#             reg = load('SGDRegressor.joblib')
#             print("The pretrain model of SGDRegressor is loaded successfully!")
#         elif self.Regression == 3:
#             reg = load('Ridge.joblib')
#             print("The pretrain model of RidgeRegression is loaded successfully!")
#         elif self.Regression == 4:
#             reg = load('Lasso.joblib')
#             print("The pretrain model of LassoRegression is loaded successfully!")
#         elif self.Regression == 5:
#             reg = load('PolynomialFeatures.joblib')
#             print("The pretrain model of PolynomialFeatures is loaded successfully!")
#         else:
#             raise ValueError('Please choose right regression model', self.Regression)
#
#         if self.loss == 1:
#             train_loss = r2_score(reg.predict(X_train), y_train)
#             test_loss = r2_score(reg.predict(X_test), y_test)
#         elif self.loss == 2:
#             train_loss = mean_squared_error(reg.predict(X_train), y_train)
#             test_loss = mean_squared_error(reg.predict(X_test), y_test)
#         else:
#             raise ValueError('Please choose right loss function', self.loss)
#         print("The loss of train data is: {}".format(train_loss))
#         print("The loss of test data is: {}".format(test_loss))
#
#
# if __name__ == "__main__":
#     parser = get_arguments()
#     MyLinearRegression = MyLinearRegression(parser)
#     # 获取样本的特征数据和标签数据
#     datas, target = MyLinearRegression.load_dataset()
#     # 划分数据，分成训练集和测试集
#     X_train, X_test, y_train, y_test = MyLinearRegression.split_dataset(datas, target)
#     # 数据归一化
#     X_train, X_test = MyLinearRegression.normalize_dataset(X_train, X_test)
#     # b = timeit.default_timer()
#     # 进行数据拟合
#     MyLinearRegression.regression(X_train, y_train)
#     # e = timeit.default_timer()
#     # 进行模型评估
#     MyLinearRegression.evaluate(X_train, X_test, y_train, y_test)
#     # print("Running time of normal equation of LinearRegression/SGD LinearRegression: {:.24f} seconds".format(e-b))
