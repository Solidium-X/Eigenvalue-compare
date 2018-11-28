"""Placeholder text
YOLO! """

# Import modules under here:
import numpy as np
import timeit
import math
import matplotlib as plt

# User defined input parameters:
# N = int(input('Please specify the dimension of the matrix \'N\' (>100, <200): '))
# tol = float(input('Please specify the error tolerance \'tol\': '))
# M = int(input('Please specify the maximum amount of iterations for power calculations (>100, <10000): '))
N = 10
tol = 0.000000001
M = 1000

# Matrix A generation using user-defined number:
A = np.random.rand(N, N)

# Test array for approach 3, make sure to set N = 3 for code-test to work
test_A = np.array([[6, 3, 2],
                   [7, 2, 3],
                   [5, 5, 1]])

# Test array for non-convergence, make sure to set N = 2 for code-test to work
test_B = np.array([[1, 1], [0, -1]])

# Initial vector to be multiplied by matrix using user-defined number
guess_vector = np.ones(N)


# APPROACH 1 - For loops
# Calculates the 2-norm using a python loop.
def norm2_diff_loop(vector_a, vector_b):
    sum_a = 0
    for i in range(len(vector_a)):
        sum_a += (vector_a[i] - vector_b[i]) ** 2
    return math.sqrt(sum_a)


# Multiplies a matrix by vector, returns equally size vector.
def matrix_vector_multiply(matrix, vector):
    result = []  # final result
    for i in range(len(matrix)):
        for j in range(len(vector)):
            product = 0  # the new element in the new row
            for v in range(len(matrix[i])):
                product += matrix[i][v] * vector[v]
        result.append(product)  # append calculated product to the final result

    return result


# Divides the elements of a vector with a given value.
def vector_element_divide(vector, value):
    new_vector = [element / value for element in vector]
    return new_vector


# Runs the matrix and guess vector through various iterations to calculate eigenvalue via power method,
# uses pre-defined for loops.
def power_eig_python(matrix, start_vector, max_iterations):
    # Counts iterations
    iterations = 0

    while iterations < max_iterations:
        # Multiplies the given matrix by the given start vector, via pre-defined python loops
        eigen_value = matrix_vector_multiply(matrix, start_vector)
        # Divides the calculated vector by its current maximum value, via pre-defined python loops
        eigen_vector = vector_element_divide(eigen_value, max(eigen_value))
        # 2-norm of the current and previous vector values, via pre-defined python loops
        euclid_dist = norm2_diff_loop(eigen_vector, start_vector)

        # Gives the calculation time and breaks the loop if 2-norm becomes lower then user-defined tolerance.
        if euclid_dist < tol:
            elapsed_time = timeit.default_timer() - start_time
            print(f'Got it! Eigenvalue for approach 1 = {max(eigen_value)}, norm = {euclid_dist:.10f}\n'
                  f'Elapsed time for approach 1 = {elapsed_time:.8f}')
            return elapsed_time
            break

        # Moves to the next iteration
        iterations += 1
        # Takes the recently calculated eigen_vector and uses it for the next iteration
        start_vector = eigen_vector

        # Force break loop if convergence is not made
        if iterations == max_iterations:
            print('Unable to converge matrix')
            break


# Running and timing of approach 1
start_time = timeit.default_timer()
approach_2_time = power_eig_python(A, guess_vector, M)


# APPROACH 2 - NumPy universal functions
def norm2_diff_numpy(vector_A, vector_B):
    diff = vector_A - vector_B
    norm = np.linalg.norm(diff)
    return norm


# Runs the matrix and guess vector through various iterations to calculate eigenvalue via power method,
# uses numpy universal functions.
def power_eig_numpy(matrix, start_vector, max_iterations):
    # Counts iterations
    iterations = 0

    while iterations < max_iterations:
        # Multiplies the given matrix by the given start vector, via numpy functions
        eigen_value = matrix.dot(start_vector)
        # Divides the calculated vector by its current maximum value, via numpy functions
        eigen_vector = eigen_value / eigen_value.max()
        # 2-norm of the current and previous vector values, via numpy functions
        euclid_dist = norm2_diff_numpy(eigen_vector, start_vector)

        # Gives the calculation time and breaks the loop if 2-norm becomes lower then user-defined tolerance.
        if euclid_dist < tol:
            elapsed_time = timeit.default_timer() - start_time
            print(f'\nGot it! Eigenvalue for approach 2 = {max(eigen_value)}, norm = {euclid_dist:.10f}\n'
                  f'Elapsed time for approach 2 = {elapsed_time:.8f}')
            return elapsed_time
            break

        # Moves to the next iteration
        iterations += 1
        # Takes the recently calculated eigen_vector and uses it for the next iteration
        start_vector = eigen_vector

        # Force break loop if convergence is not made
        if iterations == max_iterations:
            print('Unable to converge matrix')
            break

# Running and timing of approach 2
start_time = timeit.default_timer()
power_eig_numpy(A, guess_vector, M)


# APPROACH 3 - NumPy implementation
def numpy_eig(a):
    eig_values, eig_vectors = np.linalg.eig(a)
    numpy_eig.elapsed_time = timeit.default_timer() - start_time
    print(f'\nEigenvalue for approach 3 = {max(eig_values.real)} \n'
          f'Elapsed time for approach 3 = {elapsed_time:.8f}')


# Running and timing of approach 3
start_time = timeit.default_timer()
numpy_eig(A)
print(numpy_eig.elapsed_time)

# Matplot plotting of results

'''
objects = ('Approach 1', 'Approach 2', 'Approach 3')
y_pos = np.arange(len(objects))
performance = [power_eig_python(A, guess_vector, M),
               power_eig_numpy(A, guess_vector, M),
               numpy_eig(A)]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()'''