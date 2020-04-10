# this script is used to calculate the partial derivatives for various transformations

import sympy
from sympy import cos as scos, sin as ssin
from sympy.matrices import Matrix


def state_transition_function_jacobians():
    # -------------- state transition function -----------------

    # This is the nonlinear function f_r(X_r, U, N) that calculates the new robot pose.
    # f_r forms part of the overall state transition function for all states, robot pose and landmarks, although the
    # landmarks do not move and so there is no need to calculate partial derivatives for those states.
    # The robot pose is defined as X_r = [x_r; y_r; alpha_r].
    # The control input is defined as U = [x_u; alpha_u].
    # The perturbation to the input is defined as N = [x_n; alpha_n].

    x_r, y_r, alpha_r, x_u, alpha_u, x_n, alpha_n = sympy.symbols('x_r y_r alpha_r x_u alpha_u x_n alpha_n')

    X_r = Matrix([[x_r], [y_r], [alpha_r]])
    U = Matrix([[x_u], [alpha_u]])
    N = Matrix([[x_n], [alpha_n]])

    alpha_new = alpha_r + alpha_u + alpha_n
    R = Matrix([[scos(alpha_new), -ssin(alpha_new)], [ssin(alpha_new), scos(alpha_new)]])
    X_r_new = Matrix([[x_r], [y_r]]) + R * Matrix([[x_u + x_n], [0]])
    f_r = Matrix([[X_r_new[0]],
                  [X_r_new[1]],
                  [alpha_new]])

    # -----------  Jacobian of f_r w.r.t X_r (the robot pose) --------------

    d_f_r_by_X_r = f_r.jacobian(X_r)

    # ------------ Jacobian of f_r w.r.t N (the noise perturbation) --------------

    d_f_r_by_N = f_r.jacobian(N)


if __name__ == "__main__":
    state_transition_function_jacobians()
