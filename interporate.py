# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    interporate.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: krchuaip <krittin@42bangkok.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/29 02:35:21 by krchuaip          #+#    #+#              #
#    Updated: 2024/01/29 12:18:35 by krchuaip         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

# Define the function to create the design matrix for polynomial fitting
# Input: (x, degree)
# Return: np.vander obj
def create_design_matrix(x, degree):
    return np.vander(x, degree+1, increasing=True)

# Define the function to compute the coefficients for ridge regression
# Input: (x, y, degree, rho)
# Return: calculation of coeffs ridge
def compute_ridge_coefficients(x, y, degree, rho):
    A = create_design_matrix(x, degree)
    A_plus_ridge = np.linalg.inv(A.T @ A + (rho * np.identity(A.shape[1]))) @ A.T
    coeffs_ridge = A_plus_ridge @ y
    return coeffs_ridge

# Define the function to plot data and polynomial
# Input: (x, y, coeffs, degree, ax, label,, type_deg)
# Return: several funcs of ax plt.subplots called
def plot_data_and_polynomial(x, y, coeffs, degree, ax, label, type_deg):
    ax.scatter(x, y, label='Data', color='blue')
    x_model = np.linspace(min(x), max(x), 1000)
    y_model = np.polyval(coeffs[::-1], x_model)  # Reverse coefficients for np.polyval
    if type_deg == "rho_small":
        ax.plot(x_model, y_model, label=f'Degree {degree} polynomial rho small', color='red')
        ax.set_title(f'Polynomial Degree {degree} Rho Small')
    elif type_deg == "rho_large":
        ax.plot(x_model, y_model, label=f'Degree {degree} polynomial rho large', color='red')
        ax.set_title(f'Polynomial Degree {degree} Rho Large')
    else:
        ax.plot(x_model, y_model, label=f'Degree {degree} polynomial', color='red')
        ax.set_title(f'Polynomial Degree {degree}')
    ax.legend()

# Load the given data points
data_points = [
    (0.10, 0.79), (0.13, 0.81), (0.16, 0.86), (0.19, 0.88), (0.22, 0.92),
    (0.26, 0.84), (0.29, 0.95), (0.32, 0.92), (0.35, 0.78), (0.38, 0.78),
    (0.41, 0.65), (0.44, 0.61), (0.47, 0.58), (0.50, 0.48), (0.53, 0.46),
    (0.57, 0.40), (0.60, 0.24), (0.63, 0.18), (0.66, 0.09), (0.69, 0.17),
    (0.72, 0.08), (0.75, 0.07), (0.78, 0.08), (0.81, 0.14), (0.84, 0.11),
    (0.88, 0.15), (0.91, 0.36), (0.94, 0.37), (0.97, 0.43), (1.00, 0.45)
]

# Convert the data into numpy arrays for processing
x_data = np.array([point[0] for point in data_points])
y_data = np.array([point[1] for point in data_points])

# Compute the coefficients for polynomial of degree 1, 3, and 7
coeffs_degree_1 = compute_ridge_coefficients(x_data, y_data, 1, 0)
coeffs_degree_3 = compute_ridge_coefficients(x_data, y_data, 3, 0)
coeffs_degree_7 = compute_ridge_coefficients(x_data, y_data, 7, 0)

# Compute the coefficients for ridge regression with rho = 10^-6 and rho = 0.1
coeffs_ridge_small_rho_1 = compute_ridge_coefficients(x_data, y_data, 1, 1e-6)
coeffs_ridge_small_rho_2 = compute_ridge_coefficients(x_data, y_data, 3, 1e-6)
coeffs_ridge_small_rho_3 = compute_ridge_coefficients(x_data, y_data, 7, 1e-6)

coeffs_ridge_large_rho_1 = compute_ridge_coefficients(x_data, y_data, 1, 0.1)
coeffs_ridge_large_rho_2 = compute_ridge_coefficients(x_data, y_data, 3, 0.1)
coeffs_ridge_large_rho_3 = compute_ridge_coefficients(x_data, y_data, 7, 0.1)

# Plotting the results with actual data
fig, axs = plt.subplots(9, 1, figsize=(10, 25))

plot_data_and_polynomial(x_data, y_data, coeffs_degree_1, 1, axs[0], 'Degree 1', "poly")
plot_data_and_polynomial(x_data, y_data, coeffs_degree_3, 3, axs[1], 'Degree 3', "poly")
plot_data_and_polynomial(x_data, y_data, coeffs_degree_7, 7, axs[2], 'Degree 7', "poly")

plot_data_and_polynomial(x_data, y_data, coeffs_ridge_small_rho_1, 1, axs[3], 'Ridge Small Rho Degree 1', "rho_small")
plot_data_and_polynomial(x_data, y_data, coeffs_ridge_small_rho_2, 3, axs[4], 'Ridge Small Rho Degree 3', "rho_small")
plot_data_and_polynomial(x_data, y_data, coeffs_ridge_small_rho_3, 7, axs[5], 'Ridge Small Rho Degree 7', "rho_small")

plot_data_and_polynomial(x_data, y_data, coeffs_ridge_large_rho_1, 1, axs[6], 'Ridge Large Rho Degree 1', "rho_large")
plot_data_and_polynomial(x_data, y_data, coeffs_ridge_large_rho_2, 3, axs[7], 'Ridge Large Rho Degree 3', "rho_large")
plot_data_and_polynomial(x_data, y_data, coeffs_ridge_large_rho_3, 7, axs[8], 'Ridge Large Rho Degree 7', "rho_large")

plt.tight_layout()
plt.show()

