import sys

import numpy as np
import numpy.linalg as npl
import numpy.random as npr


def gauss_QR(X, k=1):
    """
    Description:
        this calculates a fast approximation of the QR decomposition with a
        gauss Vector
    Parameter:
        X - np.array : the Matrix to decompose
    Return:
        Q - np.array : the Q part
    """
    # compress X into a approximation
    n, d = X.shape

    # mapping function to compress n x d - matrix into d**2 x d - matrix
    # f: {0,...,n-1} -> {0,...,d**2-1}
    f = npr.randint(d ** 2, size=n)
    # mapping function to determine the sign
    # g: {0,...,n-1} -> {-1,1}
    g = npr.randint(2, size=n) * 2 - 1

    # init the sketch
    X_ = np.zeros((d ** 2, d))
    for i in range(n):
        X_[f[i]] += g[i] * X[i]

    R_ = np.linalg.qr(X_, mode="r")
    try:
        R_inv = np.linalg.inv(R_)
    except np.linalg.LinAlgError as err:
        print("LinAlgError: {0}".format(err), file=sys.stderr)
        print(
            "Error in gauss_QR: R_ is not invertable, because singulare matrix!"
            + " continuing with pseudo invers."
        )
        R_inv = np.linalg.pinv(R_)

    n, d = R_inv.shape
    g = np.random.normal(loc=0, scale=1 / np.sqrt(k), size=(d, k))
    # print(g)
    # print(str(R_inv.shape) + "//" + str(g.shape))
    r = np.dot(R_inv, g)
    # print(r)
    # print(str(X.shape) + "//" + str(r.shape))
    Q_ = np.dot(X, r)
    return Q_


def _calculate_sensitivities(Q, n):
    s = []

    for q in Q:
        s.append(npl.norm(q) + 1 / n)

    return np.array(s)


def l2s_sampling(
    data,
    size=100,
    k=20,
):
    num_samples, num_features = data.shape

    Q = gauss_QR(data, k)
    s = _calculate_sensitivities(Q, num_samples)

    # calculate probabilities
    p = s / np.sum(s)

    coreset_indices = npr.choice(p.shape[0], size=size, p=p, replace=False)

    # calculate the weight
    weights = 1 / (p[coreset_indices] * size)

    return data[coreset_indices], weights
