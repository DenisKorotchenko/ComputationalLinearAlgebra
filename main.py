import numpy as np
from numpy import linalg as la


def gershgorin_circles(A):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for Gershgorin Circles method"
    answer = [(0., 0.) for _ in range(size)]
    for i in range(size):
        sum = 0.
        for j in range(size):
            if i != j:
                sum += abs(A[i, j])
        answer[i] = (sum, A[i, i])
    return answer


def iteration_method(A, b, eps):
    A = A.astype(float)
    b = b.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for iteration method"
    nA = np.eye(size) - A
    is_in_Gershgorin = True
    circles = gershgorin_circles(nA)
    for i in range(size):
        r, c = circles[i]
        if c + r > 1 or c - r < -1:
            is_in_Gershgorin = False
    hm = 0
    prev_x = np.ones((len(A[0]), 1))
    while la.norm(prev_x - nA.dot(prev_x) - b) >= eps and hm <= 20:
        x = nA.dot(prev_x) + b
        if (not is_in_Gershgorin) and la.norm(x) - la.norm(prev_x) - 1 > 0:
            hm += 1
        else:
            hm = 0
        prev_x = x
    if hm > 20:
        return 0
    else:
        return prev_x


def gauss_seidel(A, b, eps):
    A = A.astype(float)
    b = b.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix in Gauss-Seidel method"
    L = np.tril(A)
    U = A - L
    is_in_Gershgorin = True
    circles = gershgorin_circles(A)
    for i in range(size):
        r, c = circles[i]
        if c + r > 1 or c - r < -1:
            is_in_Gershgorin = False
    if not is_in_Gershgorin:
        return 0
    hm = 0
    prev_x = np.ones((len(A[0]), 1))
    while la.norm(A.dot(prev_x) - b) >= eps:
        x = np.copy(-U.dot(prev_x) + b)
        for i in range(size):
            for j in range(i):
                x[i] -= L[i, j] * x[j]
            x[i] /= L[i, i]
        if la.norm(x) - la.norm(prev_x) - 1 > 0:
            hm += 1
        else:
            hm = 0
        if (hm > 20):
            return 0
        prev_x = x
    return prev_x


def dot_on_givens(A, i, j, c, s):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix in givens method"
    assert abs(c ** 2 + s ** 2 - 1) <= 0.001, "Wrong cosines and sinus values"
    g_i_line = np.array([0.0 for _ in range(size)])
    g_i_line[i] = c
    g_i_line[j] = s

    g_j_line = np.array([0.0 for _ in range(size)])
    g_j_line[i] = -s
    g_j_line[j] = c

    i_line = np.array([0. for _ in range(size)])
    j_line = np.array([0. for _ in range(size)])
    for t in range(size):
        i_line[t] = g_i_line[i] * A[i, t] + g_i_line[j] * A[j, t]
    for t in range(size):
        j_line[t] = g_j_line[i] * A[i, t] + g_j_line[j] * A[j, t]
    A[i] = i_line
    A[j] = j_line
    return A


def givens_QR(A, n=-1):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for givens method"

    eps = 0.0000001

    if n > -1:
        size = n

    QT = np.eye(size)
    for k in range(size):
        for i in range(k + 1, size):
            if abs(A[i, k]) > eps:
                c = A[k, k] / ((A[k, k] ** 2 + A[i, k] ** 2) ** 0.5)
                s = A[i, k] / ((A[k, k] ** 2 + A[i, k] ** 2) ** 0.5)
                A = dot_on_givens(A, k, i, c, s)
                QT = dot_on_givens(QT, k, i, c, s)
    return QT.T, A


def dot_on_householder(A, v):
    A = A.astype(float)
    v = v.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for dot on Householder matrix"

    return A - (v.dot(2).reshape(size, 1).dot(v.reshape(1, size).dot(A)))


def dot_on_householder_right(A, v):
    A = A.astype(float)
    v = v.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for dot on Householder matrix"

    return A - (A.dot(v.dot(2).reshape(size, 1)).dot(v.reshape(1, size)))


def householder_QR(A):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for Householder QR"

    QT = np.eye(size)
    for k in range(size - 1):
        v = np.copy(A[:, k])
        for i in range(k):
            v[i] = 0.
        v = v / la.norm(v)
        v[k] -= 1.
        v = v / la.norm(v)
        A = dot_on_householder(A, v)
        QT = dot_on_householder(QT, v)
    return QT.T, A


def iteration_eigenvalue(A, x0, eps):
    A = A.astype(float)
    x0 = x0.astype(float)
    x0 = x0 / la.norm(x0)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for iteration eigenvalues method"

    max_iterations = 10 ** 6 // (size * size)

    prev_x = x0
    for i in range(max_iterations):
        mult = A.dot(prev_x)
        x = mult / la.norm(mult)
        lam = x.T.dot(A).dot(x)
        if la.norm(A.dot(x) - x.dot(lam)) < eps:
            return lam, x
        prev_x = x
    return 0, 0


def QR_algorithm(A, eps):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for QR algorithm"
    max_iteration = 10 ** 5

    Qk = np.eye(size)
    for i in range(max_iteration):
        Q, R = givens_QR(A)
        A = R @ Q
        Qk = Qk @ Q
        circles = gershgorin_circles(A)
        max_r = 0.
        for (r, _) in circles:
            max_r = max(max_r, r)
        if max_r < eps:
            lambdas = np.array([0. for _ in range(size)])
            for j in range(size):
                lambdas[j] = A[j, j]
            return lambdas, Qk
    return None


def tridiagonalisation(A):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for tridiagonalisation algorithm"

    QT = np.eye(size)
    for k in range(size - 2):
        v = np.copy(A[:, k])
        for i in range(k + 1):
            v[i] = 0.
        v = v / la.norm(v)
        v[k + 1] -= 1.
        v = v / la.norm(v)
        A = dot_on_householder_right(A, v)
        A = dot_on_householder(A, v)
        QT = dot_on_householder(QT, v)
    return A, QT.T


def givens_QR_tridiagonal(A, n=-1):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for givens method"

    eps = 0.0000001

    QT = np.eye(size)

    if n > -1:
        size = n

    for k in range(size - 1):
        i = k + 1

        if abs(A[i, k]) > eps:
            c = A[k, k] / ((A[k, k] ** 2 + A[i, k] ** 2) ** 0.5)
            s = A[i, k] / ((A[k, k] ** 2 + A[i, k] ** 2) ** 0.5)
            A = dot_on_givens(A, k, i, c, s)
            QT = dot_on_givens(QT, k, i, c, s)
    return QT.T, A


def QR_algorithm_tridiagonal(A, eps):
    A = A.astype(float)
    size, size2 = np.shape(A)
    A_0 = np.copy(A)
    assert size == size2, "Not squared matrix for QR algorithm"
    max_iteration = 10 ** 5

    Qk = np.eye(size)
    for i in range(max_iteration):
        Q, R = givens_QR_tridiagonal(A)
        A = R @ Q
        Qk = Qk @ Q
        circles = gershgorin_circles(A)
        max_r = 0.
        for (r, _) in circles:
            max_r = max(max_r, r)
        if max_r < eps:
            lambdas = np.array([0. for _ in range(size)])
            for j in range(size):
                lambdas[j] = A[j, j]
            return lambdas, Qk
    return None


def wilkinson_shift(A, eps):
    A = A.astype(float)
    size, size2 = np.shape(A)
    assert size == size2, "Not squared matrix for fast QR algorithm"
    max_iteration = 10 ** 5

    Qk = np.eye(size)
    En = np.eye(size)
    iterations = 0
    for i in range(size - 1, 0, -1):
        while True:
            r, _ = gershgorin_circles(A)[i]
            if r < eps:
                break
            iterations += 1
            if iterations > max_iteration:
                return None

            lambda1 = (A[i, i] + A[i - 1, i - 1] + (
                    ((A[i - 1, i - 1] - A[i, i]) ** 2 + 4. * A[i, i - 1] ** 2.) ** 0.5)) / 2
            lambda2 = (A[i, i] + A[i - 1, i - 1] - (
                    ((A[i - 1, i - 1] - A[i, i]) ** 2 + 4. * A[i, i - 1] ** 2.) ** 0.5)) / 2

            if abs(lambda1 - A[i, i]) < abs(lambda2 - A[i, i]):
                s = lambda1
            else:
                s = lambda2
            Q, R = givens_QR_tridiagonal(A - En.dot(s), i + 1)
            A = R @ Q + En.dot(s)
            Qk = Qk @ Q
    lambdas = np.array([0. for _ in range(size)])
    for j in range(size):
        lambdas[j] = A[j, j]
    return lambdas, Qk


def non_isomorphic(G1, G2):
    G1 = G1.astype(float)
    G2 = G2.astype(float)
    size1, size12 = np.shape(G1)
    assert size1 == size12, "Not graph matrix"
    size2, size22 = np.shape(G2)
    assert size2 == size22, "Not graph matrix"

    eps = 0.000001

    if size1 != size2:
        return 0

    tG1, _ = tridiagonalisation(np.copy(G1))
    tG2, _ = tridiagonalisation(np.copy(G2))

    lam1, _ = QR_algorithm_tridiagonal(tG1, 0.00001)
    lam2, _ = QR_algorithm_tridiagonal(tG2, 0.00001)
    lam1.sort()
    lam2.sort()
    for i in range(size1):
        if abs(lam1[i] - lam2[i]) > eps:
            return 0

    e1 = 0
    e2 = 0
    for i in range(size1):
        for j in range(size1):
            e1 += G1[i, j]
            e2 += G2[i, j]

    if abs(e1 - e2) > eps:
        return 0

    return 1


def to_index(i, j, n):
    return (i % n) * n + (j % n)


def alpha_counter1(n):
    G = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            t = to_index(i, j, n)
            tt = [to_index(i + 2 * j, j, n),
                  to_index(i - 2 * j, j, n),
                  to_index(i + 2 * j + 1, j, n),
                  to_index(i - 2 * j - 1, j, n),
                  to_index(i, j + 2 * i, n),
                  to_index(i, j - 2 * i, n),
                  to_index(i, j + 2 * i + 1, n),
                  to_index(i, j - 2 * i - 1, n)]
            for ta in tt:
                G[t, ta] += 1
    d = 0
    for i in range(n * n):
        d += G[0, i]
    tG, _ = tridiagonalisation(G)
    lambdas, _ = wilkinson_shift(tG, 0.00001)
    lambdas.sort()
    c = max(abs(lambdas[n * n - 2]), abs(lambdas[0]))
    return c / d


def alpha_counter2(p):
    G = np.zeros((p + 1, p + 1))
    for i in range(1, p):
        G[i, (i + 1) % p] += 1
        G[i, (i - 1) % p] += 1
        G[i, pow(i, p - 2, p)] += 1
    G[0, 1] += 1
    G[0, p - 1] += 1
    G[0, p] += 1
    G[p, p] += 2
    G[p, 0] += 1
    d = 0
    for i in range(p + 1):
        d += G[0, i]
    tG, _ = tridiagonalisation(G)
    lambdas, _ = wilkinson_shift(tG, 0.0000001)
    lambdas.sort()

    c = max(abs(lambdas[p - 1]), abs(lambdas[0]))
    return c / d
