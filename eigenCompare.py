#! /usr/bin/python3
"""Assignment 2 by Sebastian Pineda
 This script generates 5 different N*N (user-defined) matrices and calculates the eigenvalue using 3 different
 approaches. The computation time for each approach per matrix is timed and output in to a horizontal bar chart."""

# Import modules under here:
import numpy as np
import timeit
import math
import matplotlib.pyplot as plt

# User defined input parameters:
N = int(input("Please specify the N*N dimension of the matrices (>100, <200): "))
tol = float(input("Please specify the 2-norm tolerance (<1e-06): "))
M = int(input("Please specify the maximum amount of iterations for power convergence (>100, <10000): "))

# Confirms user-set values
print(f'\nN set to : {N}\n'
      f'Tolerance set to: {tol}\n'
      f'Maximum iterations set to: {M}\n')

# Matrix generation using user-defined number:
matrix_A = np.random.rand(N, N)
matrix_B = np.random.rand(N, N)
matrix_C = np.random.rand(N, N)
matrix_D = np.random.rand(N, N)
matrix_E = np.random.rand(N, N)

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


# Multiplies a matrix by vector, returns appropriate sized vector.
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
# uses pre-defined for-loops.
def power_eig_python(matrix, start_vector, max_iterations):
    start_time = timeit.default_timer()
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
            print(f'\nGot it! Eigenvalue for approach 1 = {max(eigen_value)}, norm = {euclid_dist:.10f}\n'
                  f'Elapsed time for approach 1 = {elapsed_time:.8f}')
            return max(eigen_value), elapsed_time

        # Moves to the next iteration
        iterations += 1
        # Takes the recently calculated eigen_vector and uses it for the next iteration
        start_vector = eigen_vector

        # Break loop if convergence is not made
        if iterations == max_iterations:
            print('Unable to converge matrix')
            break

# Test run of approach 1
#ap1_eigval, ap1_time = power_eig_python(matrix_A, guess_vector, M)
#print(ap1_eigval, ap1_time)


# APPROACH 2 - NumPy universal functions
def norm2_diff_numpy(vector_A, vector_B):
    diff = vector_A - vector_B
    norm = np.linalg.norm(diff)
    return norm


# Runs the matrix and guess vector through various iterations to calculate eigenvalue via power method,
# uses numpy universal functions.
def power_eig_numpy(matrix, start_vector, max_iterations):
    start_time = timeit.default_timer()
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
            return max(eigen_value), elapsed_time
            break

        # Moves to the next iteration
        iterations += 1
        # Takes the recently calculated eigen_vector and uses it for the next iteration
        start_vector = eigen_vector

        # Break loop if convergence is not made
        if iterations == max_iterations:
            print('Unable to converge matrix')
            break


# Test run of approach 2
#start_time = timeit.default_timer()
#ap2_eigval, ap2_time = power_eig_numpy(matrix_A, guess_vector, M)
#print(ap2_eigval, ap2_time)


# APPROACH 3 - NumPy implementation
def numpy_eig(a):
    start_time = timeit.default_timer()
    eig_values, eig_vectors = np.linalg.eig(a)
    elapsed_time = timeit.default_timer() - start_time
    print(f'\nEigenvalue for approach 3 = {max(eig_values.real)} \n'
          f'Elapsed time for approach 3 = {elapsed_time:.8f}')
    return max(eig_values.real), elapsed_time

# Test run of approach 3
#start_time = timeit.default_timer()
#ap3_eigval, ap3_time = numpy_eig(matrix_A)


# Running the approaches on each matrix, saving the eigenvalue and computation time
print('#'*80, '\nApproach 1 on 5 different matrices:')
ap1_eigval_A, ap1_time_A = power_eig_python(matrix_A, guess_vector, M)
ap1_eigval_B, ap1_time_B = power_eig_python(matrix_B, guess_vector, M)
ap1_eigval_C, ap1_time_C = power_eig_python(matrix_C, guess_vector, M)
ap1_eigval_D, ap1_time_D = power_eig_python(matrix_D, guess_vector, M)
ap1_eigval_E, ap1_time_E = power_eig_python(matrix_E, guess_vector, M)

print('#'*80, '\nApproach 2 on 5 different matrices:')
ap2_eigval_A, ap2_time_A = power_eig_numpy(matrix_A, guess_vector, M)
ap2_eigval_B, ap2_time_B = power_eig_numpy(matrix_B, guess_vector, M)
ap2_eigval_C, ap2_time_C = power_eig_numpy(matrix_C, guess_vector, M)
ap2_eigval_D, ap2_time_D = power_eig_numpy(matrix_D, guess_vector, M)
ap2_eigval_E, ap2_time_E = power_eig_numpy(matrix_E, guess_vector, M)

print('#'*80, '\nApproach 3 on 5 different matrices:')
ap3_eigval_A, ap3_time_A = numpy_eig(matrix_A)
ap3_eigval_B, ap3_time_B = numpy_eig(matrix_B)
ap3_eigval_C, ap3_time_C = numpy_eig(matrix_C)
ap3_eigval_D, ap3_time_D = numpy_eig(matrix_D)
ap3_eigval_E, ap3_time_E = numpy_eig(matrix_E)
print('#'*80)


# Gather corresponding eigenvalues into a list
all_eigval_A = [ap1_eigval_A, ap2_eigval_A, ap3_eigval_A]
all_eigval_B = [ap1_eigval_B, ap2_eigval_B, ap3_eigval_B]
all_eigval_C = [ap1_eigval_C, ap2_eigval_C, ap3_eigval_C]
all_eigval_D = [ap1_eigval_D, ap2_eigval_D, ap3_eigval_D]
all_eigval_E = [ap1_eigval_E, ap2_eigval_E, ap3_eigval_E]


# Checks eigenvalues to see if they are equal to each other up to 1e-06
def compare_eigvals(eig_list):
    print(eig_list)
    print("The above eigen values are equal up to 1e-06: ",
          all(round(x, 6) == round(eig_list[0], 6) for x in eig_list), '\n')


# Compute if eigenvalues for each matrix and approach are equal up to 10-6
print('\nEigenvalue checks for 5 different matrices:\n')
compare_eigvals(all_eigval_A)
compare_eigvals(all_eigval_B)
compare_eigvals(all_eigval_C)
compare_eigvals(all_eigval_D)
compare_eigvals(all_eigval_E)

# Calculate averages  computation times
ap1_average_time = (ap1_time_A + ap1_time_B + ap1_time_C + ap1_time_D + ap1_time_E) / 5
ap2_average_time = (ap2_time_A + ap2_time_B + ap2_time_C + ap2_time_D + ap2_time_E) / 5
ap3_average_time = (ap3_time_A + ap3_time_B + ap3_time_C + ap3_time_D + ap3_time_E) / 5


# Testing of the average times
print('#'*80)
print('Time average for approach 1 (in seconds): ', ap1_average_time)
print('Time average for approach 2 (in seconds): ', ap2_average_time)
print('Time average for approach 3 (in seconds): ', ap3_average_time)


# Matplotlib bar plot of results
plt.rcdefaults()
fig, ax = plt.subplots()

objects = ('Approach 1', 'Approach 2', 'Approach 3')
y_pos = np.arange(len(objects))
performance = [ap1_average_time,
               ap2_average_time,
               ap3_average_time]

plt.barh(y_pos, performance, align='center', color='blue', alpha=0.5)
plt.yticks(y_pos, objects)
ax.invert_yaxis()
plt.xlabel('Time in seconds\n(Less is better)')
plt.title('Comparison of approach calculation speeds')

plt.savefig('assignment_2_eigen_comparison.png')
plt.show()
