import numpy as np
import pandas as pd
from libpyhat.examples import get_path
from libpyhat.transform.norm import norm
from libpyhat.regression.regression import regression
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler
np.random.seed(1)

def read_test_data():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    df = norm(df, [[580, 600]])
    x = df['wvl']
    y = df[('comp', 'SiO2')]
    return x,y

def test_PLS():
    x,y = read_test_data()
    regress = regression(method=['PLS'],
                         params=[{'n_components': 3, 'scale': False}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 9.568890617713505
    np.testing.assert_almost_equal(rmse, expected)

    regress.calc_Qres_Lev(x)
    Qres_expected = [0.04055878, 0.04188589, 0.04159104, 0.04374264, 0.04080776, 0.04072383, 0.04057845, 0.04053754,
                     0.04056575, 0.04077855]
    np.testing.assert_array_almost_equal(regress.Q_res[0:10], Qres_expected)
    leverage_expected = [0.01225164, 0.01219529, 0.01431885, 0.03043435, 0.05013193, 0.01418457, 0.01055998, 0.00554777,
                         0.00891671, 0.00912439]
    np.testing.assert_array_almost_equal(regress.leverage[0:10], leverage_expected)


def test_badfit():
    x, y = read_test_data()
    regress = regression(method=['OMP'],
                         params=[{'n_nonzero_coefs': 1000}])
    regress.fit(x, y)
    assert regress.goodfit == False

def test_OLS():
    x, y = read_test_data()
    regress = regression(method=['OLS'],
                         params=[{'fit_intercept': True,
                                  'normalize': False,
                                  'copy_X': True,
                                  'positive': False}])

    x_temp = np.array(x)[0:20,:]  # use truncated versions of input data to fit the model to keep it well behaved
    y_temp = np.array(y)[0:20]

    regress.fit(x_temp, y_temp)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 26.30786143321134
    np.testing.assert_almost_equal(rmse, expected)


def test_OMP():
    x, y = read_test_data()
    regress = regression(method=['OMP'], params=[{'fit_intercept': True}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 9.835802028648189
    np.testing.assert_almost_equal(rmse, expected)

def test_LASSO():
    x, y = read_test_data()
    regress = regression(method=['LASSO'],
                         params=[{'alpha': 1.0,
                                  'fit_intercept': True,
                                  'positive': False}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 22.815757879917708
    np.testing.assert_almost_equal(rmse, expected)


def test_Elastic_Net():
    x, y = read_test_data()
    regress = regression(method=['Elastic Net'],
                         params=[{'alpha': 1.0,
                                  'l1_ratio': 0.5,
                                  'fit_intercept': True,
                                  'positive': False}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 22.800420060822468
    np.testing.assert_almost_equal(rmse, expected)



def test_Ridge():
    x, y = read_test_data()
    regress = regression(method=['Ridge'],
                         params=[{'alpha': 1.0,
                                  'fit_intercept': True}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 19.29172384871638
    np.testing.assert_almost_equal(rmse, expected)

def test_Bayesian_Ridge():
    x, y = read_test_data()
    regress = regression(method=['BRR'],
                         params=[{'n_iter': 300,
                                  'tol': 0.001,
                                  'alpha_1': 0.001,
                                  'alpha_2': 1e-06,
                                  'lambda_1': 1e-06,
                                  'lambda_2': 1e-06,
                                  'compute_score': False,
                                  'fit_intercept': True,
                                  'normalize': False}])
    regress.fit(x, y)
    prediction, pred_std = np.squeeze(regress.predict(x,return_std=True))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 6.3894201026386135
    expected_std = [290.12684166, 290.1432507, 290.13760848, 290.14289498]
    np.testing.assert_almost_equal(rmse, expected)
    np.testing.assert_array_almost_equal(pred_std[0:4], expected_std)


def test_ARD():
    x, y = read_test_data()
    regress = regression(method=['ARD'],
                         params=[{'n_iter': 300,
                                  'tol': 0.001,
                                  'alpha_1': 0.001,
                                  'alpha_2': 1e-06,
                                  'lambda_1': 1e-06,
                                  'lambda_2': 1e-06,
                                  'compute_score': False,
                                  'threshold_lambda': 100000,
                                  'fit_intercept': True,
                                  'normalize': False,
                                  'copy_X': True,
                                  'verbose': False}])
    regress.fit(x, y)
    prediction, pred_std = np.squeeze(regress.predict(x, return_std=True))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 6.714452573751844
    expected_std = [ 8.55834607, 9.01277664, 9.03318737, 8.61304938]
    np.testing.assert_almost_equal(rmse, expected)
    np.testing.assert_array_almost_equal(pred_std[0:4],expected_std)


def test_LARS():
    x, y = read_test_data()
    regress = regression(method=['LARS'],
                         params=[{'n_nonzero_coefs': 5,
                                  'fit_intercept': True,
                                  'normalize': False,
                                  'precompute': True,
                                  'copy_X': True,
                                  'eps': 2.220445,
                                  'fit_path': True}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 21.952591101815294
    np.testing.assert_almost_equal(rmse, expected)


def test_SVR():
    x, y = read_test_data()
    regress = regression(method=['SVR'], params=[{'C': 1.0, 'epsilon': 0.1,
                                                   'kernel': 'rbf',
                                                   'degree': 0,
                                                   'gamma': 'auto',
                                                   'coef0': 0.0,
                                                   'shrinking': False,
                                                   'tol': 0.001,
                                                   'cache_size': 200,
                                                   'verbose': False,
                                                   'max_iter': -1}])
    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 23.740048198035947
    np.testing.assert_almost_equal(rmse, expected)


def test_KRR():
    x, y = read_test_data()
    regress = regression(method=['KRR'],
                         params=[{'alpha': 0,
                                  'kernel': 'linear',
                                  'gamma': 'None',
                                  'degree': 3.0,
                                  'coef0': 1.0,
                                  'kernel_params': 'None'}])
    x_temp = np.array(x)[0:20,:] #use truncated versions of input data to fit the model to avoid singular matrix warning
    y_temp = np.array(y)[0:20]
    regress.fit(x_temp,y_temp)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 26.299088128039653
    np.testing.assert_almost_equal(rmse, expected,decimal=5)

def test_GBR():
    x, y = read_test_data()
    regress = regression(method=['GBR'],
                         params = [{'learning_rate':0.1,
                                   'n_estimators':10,
                                   'alpha':0.9,
                                   'subsample':1,
                                   'min_samples_split':2,
                                   'min_samples_leaf':1,
                                   'min_weight_fraction_leaf':0,
                                   'max_depth':3,
                                   'min_impurity_decrease':0.0}])

    regress.fit(x, y)
    prediction = np.squeeze(regress.predict(x))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 9.75587930063076
    np.testing.assert_almost_equal(rmse, expected, decimal=5)

def test_GP():
    x, y = read_test_data()
    regress = regression(method=['GP'],
                         params = [{'kernel': Matern(length_scale=1.0, nu=3.0),
                                    'alpha':1e-10,
                                    'optimizer':"fmin_l_bfgs_b",
                                    'n_restarts_optimizer':0,
                                    'normalize_y':False,
                                    'copy_X_train':True,
                                    'random_state':0}])
    x_temp = np.array(x)[0:20,:]  # use truncated versions of input data to fit the model
    y_temp = np.array(y)[0:20]

    scaler = StandardScaler()
    x_temp = scaler.fit_transform(x_temp)

    regress.fit(x_temp, y_temp)
    prediction = np.squeeze(regress.predict(scaler.transform(x)))
    rmse = np.sqrt(np.average((prediction - y) ** 2))
    expected = 25.613588134522985
    np.testing.assert_almost_equal(rmse, expected, decimal=4)
