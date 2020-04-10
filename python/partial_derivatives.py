# this script is used to calculate the partial derivatives for various transformations

import sympy
from sympy import cos as scos, sin as ssin, atan2, sqrt
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


def observe_range_bearing_jacobians():
    # --------------- nonlinear range-bearing sensor measurement function ----------------

    # This function maps the landmark positions (in world rectangular coordinates) to range-bearing (local polar)
    # coordinates.

    # we need to use the estimated robot states in the Jacobian since the true states are not available to the estimator
    x_r, y_r, alpha_r, l_i_x, l_i_y = sympy.symbols('x_r y_r alpha_r l_i_x l_i_y')

    # transform landmark from world (rectangular)  to local (rectangular) coordinates
    R = Matrix([[scos(alpha_r), -ssin(alpha_r)], [ssin(alpha_r), scos(alpha_r)]])

    t = Matrix([x_r, y_r])
    p_world = Matrix([l_i_x, l_i_y])

    p_local_x, p_local_y = R.T * (p_world - t)

    y_range = sqrt(p_local_x ** 2 + p_local_y ** 2)
    y_bearing = atan2(p_local_y, p_local_x)

    # calculate Jacobian in one go -> number of terms explodes!
    h = Matrix([y_range, y_bearing])

    # ---------- Jacobian of h w.r.t X_r (robot position) ---------

    X_r = Matrix([[x_r], [y_r], [alpha_r]])

    # calculate Jacobian in one go
    d_h_r_by_X_r = h.jacobian(X_r)

    # # calculate Jacobian in steps -> much fewer terms
    #
    # h1 = Matrix([p_local_x, p_local_y])
    #
    # p_local = Matrix([p_local_x, p_local_y])
    # d_h1_by_p_local = h1.jacobian(p_local)  # calculate first part of Jacobian
    #
    # # p_local_x_sym, p_local_y_sym = sympy.symbols('p_local_x_sym p_local_y_sym')
    #
    # # h2 = Matrix([y_range, y_bearing])
    # # p_local_syms = Matrix([p_local_x_sym, p_local_y_sym])
    # # d_h2_by_X_r = h2.jacobian(X_r)          # calculate second part of Jacobian
    # # # therefore d_h_by_X_r == d_h1_by_p_local * d_h2_by_X_r
    #
    # # ---------- Jacobian of h w.r.t L_i (landmark position) ---------
    #
    # L_i = Matrix([[l_i_x], [l_i_y]])
    # d_h_r_by_L_i = h.jacobian(L_i)

    # ---------- Jacobian of h w.r.t L_i (landmark position) ---------

    L_i = Matrix([[l_i_x], [l_i_y]])

    # calculate Jacobian in one go
    d_h_r_by_L_i = h.jacobian(L_i)

    print("test")


def inv_observe_range_bearing_jacobians():
    # --------------- nonlinear inverse range-bearing measurement function ----------------

    # This function maps the landmark positions (in world rectangular coordinates) to range-bearing (local polar)
    # coordinates.

    # we need to use the estimated robot states in the Jacobian since the true states are not available to the estimator
    x_r, y_r, alpha_r, rho, psi = sympy.symbols('x_r y_r alpha_r rho psi')

    # transform from polar to rectangular coordinates
    x = rho * scos(psi)
    y = rho * ssin(psi)

    # transform from local rectangular coordinates to world rectangular coordinates
    R = Matrix([[scos(alpha_r), -ssin(alpha_r)], [ssin(alpha_r), scos(alpha_r)]])

    t = Matrix([x_r, y_r])
    p_local = Matrix([x, y])

    g = p_world = R * p_local + t

    # calculate Jacobian in one go -> number of terms explodes!

    # ---------- Jacobian of g w.r.t X_r (robot position) ---------

    X_r = Matrix([[x_r], [y_r], [alpha_r]])

    # calculate Jacobian in one go
    d_g_r_by_X_r = g.jacobian(X_r)

    # ---------- Jacobian of g w.r.t y_i (range-bearing sensor measurement) ---------

    y_i = Matrix([[rho], [psi]])

    # calculate Jacobian in one go
    d_h_r_by_y_i = g.jacobian(y_i)

    print("test2")


if __name__ == "__main__":
    state_transition_function_jacobians()
    observe_range_bearing_jacobians()
    inv_observe_range_bearing_jacobians()
