import hw5_utils as utils
import hw5 as hw
import numpy as np
from scipy.stats import multivariate_normal as gauss
from sklearn.metrics import mean_squared_error as mse


# DONE
def _2c():
	X, y, _, _ = utils.load_iris_data()
	pur = []
	for k in range(2, 22):
		C = hw.k_means(X, k)
		frac = hw.get_purity_score(X, y, C)
		pur.append(frac)

	utils.line_plot(pur, save=False)


# DONE
def _2f():
	X_train, X_test, y_train, y_test = utils.load_iris_data(ratio=0.8)

	l1_train_mse = []
	l1_test_mse = []
	l3_train_mse = []
	l3_test_mse = []

	for k in range(3, 21):


		f1, C1 = hw.classify_using_k_means(X_train, y_train, k)
		A_train = hw.feature_extract_k_means(C1, X_train, k)
		y_pred = f1.predict(A_train)
		l1_train_mse.append(mse(y_train, y_pred))

		A_test = hw.feature_extract_k_means(C1, X_test, k)
		y_pred = f1.predict(A_test)
		l1_test_mse.append(mse(y_test, y_pred))


		f3, C3 = hw.classify_using_k_means(X_train, y_train, k, l=3)
		A_train = hw.feature_extract_k_means(C3, X_train, k, l=3)
		y_pred = f3.predict(A_train)
		l3_train_mse.append(mse(y_train, y_pred))

		A_test = hw.feature_extract_k_means(C3, X_test, k, l=3)
		y_pred = f3.predict(A_test)
		l3_test_mse.append(mse(y_test, y_pred))


	utils.line_plot(l1_train_mse, l1_test_mse, l3_train_mse, l3_test_mse)

# DONE
def _2g():
	X, _, _, _ = utils.load_iris_data()

	k = 4
	C = hw.k_means(X, k)
	A = hw.feature_extract_k_means(C, X, k, l=1)
	utils.scatter_plot_2d_project(X[np.nonzero(A[:,0])], X[np.nonzero(A[:,1])], X[np.nonzero(A[:,2])], X[np.nonzero(A[:,3])], C)


# DONE
def _3b():
	X, _, _, _ = utils.load_iris_data()

	log_like = []
	for k in range(2,11):
		print(k)
		mu, sigma, pi = hw.gmm(X, k)
		log_like.append(sum(np.log(sum(pi[c] * gauss.pdf(x, mean=mu[c], cov=sigma[c]) for c in range(k))) for x in X))

	utils.line_plot(log_like, min_k=2, save=False)

# DONE
def _3e():
	k = 4
	X, _, _, _ = utils.load_iris_data()
	mu, sigma, pi = hw.gmm(X, k)

	A = feature_extract_gmm(mu, sigma, pi, X, k)

	# utils.gaussian_plot_2d_project(mu, sigma, X_train[A[:,0],:], X_train[A[:,1],:], X_train[A[:,2],:], X_train[A[:,3],:])
	utils.gaussian_plot_2d_project(mu, sigma, X[np.nonzero(A[:,0])], X[np.nonzero(A[:,1])], X[np.nonzero(A[:,2])], X[np.nonzero(A[:,3])])




if __name__=='__main__':
    # _2c()
    # _2f()
    # _2g()
    _3b()
    # _3e()


