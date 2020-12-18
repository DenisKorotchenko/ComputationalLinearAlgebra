import main
import numpy as np


def task1():
    print("Задача 1")
    print("Ожидается: [2, 2]")
    print(main.iteration_method(np.array([[0.5, 0], [0, 0.5]]), np.array([1, 1]).reshape((2, 1)), 0.0001))
    print()


def task2():
    print("Задача 2")
    print("Ожидается: [2, 2]")
    print(main.gauss_seidel(np.array([[0.5, 0], [0, 0.5]]), np.array([1, 1]).reshape((2, 1)), 0.0001))
    print()


def task3():
    print("Задача 3")
    print("Ожидается: [[6.56, 7.93, 9.29], [4, 5, 6], [2.63, 2.27, 1.90]]")
    print(main.dot_on_givens(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), 0, 2, 1 / 2, 3 ** 0.5 / 2))
    print()


def task4():
    print("Задача 4")
    A = np.array([[-2, 2, 3], [-9, 7, 5], [-5, 2, 6]])
    Q, R = main.givens_QR(np.copy(A))
    print("Проверка ортогональности (должна быть близка к единичной:")
    print(Q@Q.T)
    print("R - матрица (должна быть верхнетреугольная):")
    print(R)
    print("Проверка разложения (должно быть близко к нулю)")
    print(Q@R - A)
    A = np.array([[2, 2, 3], [2, 7, 5], [3, 5, 6]])
    Q, R = main.givens_QR(np.copy(A))
    print("Проверка ортогональности (должна быть близка к единичной:")
    print(Q @ Q.T)
    print("R - матрица (должна быть верхнетреугольная):")
    print(R)
    print("Проверка разложения (должно быть близко к нулю)")
    print(Q @ R - A)
    print()


def task5():
    print("Задача 5")
    print("Ожидается: [[-303, -334, -365, -396], [-907, -1002, -1097, -1192], [-1511, -1670, -1829, -1988], [-2115, -2338, -2561, -2784]]")
    print(main.dot_on_householder(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), np.array([1, 3, 5, 7])))
    print()


def task6():
    print("Задача 6")
    A = np.array([[-2, 2, 3], [-9, 7, 5], [-5, 2, 6]])
    Q, R = main.householder_QR(np.copy(A))
    print("Проверка ортогональности (должна быть близка к единичной:")
    print(Q @ Q.T)
    print("R - матрица (должна быть верхнетреугольная):")
    print(R)
    print("Проверка разложения (должно быть близко к нулю)")
    print(Q @ R - A)
    A = np.array([[2, 2, 3], [2, 7, 5], [3, 5, 6]])
    Q, R = main.householder_QR(np.copy(A))
    print("Проверка ортогональности (должна быть близка к единичной:")
    print(Q @ Q.T)
    print("R - матрица (должна быть верхнетреугольная):")
    print(R)
    print("Проверка разложения (должно быть близко к нулю)")
    print(Q @ R - A)
    print()


def task7():
    print("Задача 7")
    print("Ожидается: (4.37, [0.94, 0.35])")
    print(main.iteration_eigenvalue(np.array([[4., 1.], [2, -1]]), np.array([2.5, 1.]), 0.0001))
    print()


def task8():
    print("Задача 8")
    A = np.array([[2, 2, 3], [2, 7, 5], [3, 5, 6]])
    print("Ожидается: [12.68, 2.05, 0.27]")
    lambdas, Qk = main.QR_algorithm(A, 0.00001)
    print(lambdas)
    print()


def task9():
    print("Задача 9")
    A = np.array([[2, 2, 3, 1], [2, 7, 5, 4], [3, 5, 6, 10], [1, 4, 10, 6]])
    T, Q = main.tridiagonalisation(np.copy(A))
    print("Тридиагонализация")
    print(T)
    print("Проверка Q - результат ожидается близким к 0")
    print(Q.T.dot(A).dot(Q) - T)
    print()


def task10():
    print("Задача 10")
    A, _ = main.tridiagonalisation(np.array([[2, 2, 3], [2, 7, 5], [3, 5, 6]]))
    print("Ожидается: [12.68, 2.05, 0.27]")
    lambdas, Qk = main.QR_algorithm_tridiagonal(A, 0.00001)
    print(lambdas)
    print()


def task11():
    print("Задача 11")
    A, _ = main.tridiagonalisation(np.array([[2, 2, 3], [2, 7, 5], [3, 5, 6]]))
    print("Ожидается: [12.68, 2.05, 0.27]")
    lambdas, Qk = main.wilkinson_shift(np.copy(A), 0.00001)
    print(lambdas)
    print("Ожидается диагональная с диагональю: [12.68, 2.05, 0.27]")
    print(Qk.T@A@Qk)
    print("Проверка ортогональности Qk (ожидается единичная)")
    print(Qk@(Qk.T))
    print(lambdas)
    print()


def task12():
    print("Задача 12")
    G1 = np.array([[0,0,1,0,0,1,0,0],
                   [0,0,0,1,0,0,0,0],
                   [1,0,0,0,0,1,0,0],
                   [0,1,0,0,1,1,1,0],
                   [0,0,0,1,0,0,0,1],
                   [1,0,1,1,0,0,1,0],
                   [0,0,0,1,0,1,0,0],
                   [0,0,0,0,1,0,0,0]])
    G2 = np.array([[0,0,0,1,0,0,0,1],
                   [0,0,0,1,1,1,1,0],
                   [0,0,0,0,0,0,1,0],
                   [1,1,0,0,1,0,0,1],
                   [0,1,0,1,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,1,1,0,0,0,0,0],
                   [1,0,0,1,0,0,0,0]])
    print("Ожидается: 1")
    print(main.non_isomorphic(G1, G2))

    G3 = np.array([[1, 0, 1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 1, 1, 0, 0, 1],
                   [1, 0, 1, 1, 0, 1, 1, 0],
                   [0, 0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0]])
    G4 = np.array([[1, 0, 0, 1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [1, 1, 0, 1, 1, 0, 0, 1],
                   [0, 1, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0],
                   [0, 1, 1, 0, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0, 0, 0, 1]])
    print("Ожидается: 1")
    print(main.non_isomorphic(G3, G4))

    G5 = np.array([[1, 0, 1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 1, 1, 0, 0, 1],
                   [1, 0, 1, 1, 0, 1, 1, 0],
                   [0, 0, 0, 1, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0]])
    G6 = np.array([[1, 0, 0, 1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0, 0, 1, 0],
                   [1, 1, 0, 1, 1, 0, 0, 1],
                   [0, 1, 0, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0],
                   [0, 1, 1, 0, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0, 0, 0, 0]])
    print("Ожидается: 0")
    print(main.non_isomorphic(G5, G6))
    print()


def task13():
    print("Задача 13")
    print("Alpha для графа 1 для n = 6")
    print(main.alpha_counter1(6))
    print("Alpha для графа 2 для p = 23")
    print(main.alpha_counter2(23))
    print()


np.set_printoptions(precision=2, suppress=True)
task1()
task2()
task3()
task4()
task5()
task6()
task7()
task8()
task9()
task10()
task11()
task12()
task13()
