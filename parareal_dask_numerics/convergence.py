import matplotlib.pyplot as plt
import random

def read_convergence_values_from_file(filename):
    with open(filename, 'r') as file:
        convergence_values = [float(line.strip()) for line in file]
    return convergence_values



def plot_parareal_convergence():
    convergence_values = read_convergence_values_from_file(filename='parareal/convergence.txt')
    convergence_values_ml = read_convergence_values_from_file(filename='parareal/convergence_ml.txt')
    iterations = list(range(1, len(convergence_values) + 1))

    plt.semilogy(iterations, convergence_values, marker='o', linestyle='-', color='b', label='Num')
    plt.semilogy(iterations, convergence_values_ml, marker='x',
                 linestyle='-', color='r', label='ML')
    plt.title('Parareal Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(False)
    plt.legend()
    plt.show()


def plot_parareal_convergence_st():
    convergence_values = read_convergence_values_from_file(filename='parareal/convergence_st.txt')
    convergence_values_ml = read_convergence_values_from_file(filename='parareal/convergence_ml_st.txt')
    iterations = list(range(1, len(convergence_values) + 1))

    plt.plot(iterations, convergence_values, marker='o', linestyle='-', color='b', label='Num')
    plt.plot(iterations, convergence_values_ml, marker='x', linestyle='-', color='r', label='ML')
    plt.title('Parareal Convergence - Space Time')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(False)
    plt.legend()
    plt.show()


# plot_parareal_convergence()
# plot_parareal_convergence_difference()