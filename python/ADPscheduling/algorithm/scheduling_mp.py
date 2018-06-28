# Multiperiod scheduling optimization implementation
# Aaron Ji

import numpy as np
#import scipy as sp
import sys
import cvxpy as cvx
from scipy.sparse import coo_matrix

## This function computes the optimal flow and the left/right value of each resource
## Inputs
# n: number of location
# tau_max: the largest travel time between any pair of locations
# R: an n-by-1 matrix characterizing the initial resource vector (at the left node
# Ru: an n-by-tau_max matrix characterizing the upcoming resource vector which are
# initially unavailable, but will arrive within tau_max time intervals as planned
# M: an n-by-n matrix with M(i,j) characterizing the demand from i to j
# W: an n-by-n matrix with W(i,j) characterizing the value of the demand from i to j
# C: an n-by-n matrix with c(i,j) characterizing the cost of repositioning a resource from i to j
# rep_matrix: an n-by-n matrix with rep_matrix(i,j) = 1 if it is possible
# to reposition a resource from i to j, otherwise, rep_matrix(i,j) = 0
# tauX: an n-by-n matrix with tauX(i,j) means the travel time of task from i to j; note no task from i to i
# tauY: an n-by-n matrix with tauY(i,j) means the travel time of reposition from i to j; tauY(i,i) = 0 since it means no reposition
# P: an N-by-tau_max*n matrix characterizing the coefficients of the piecewise
# linear function at the future end node, with P(k, (tau-1)*n+i) = V_tau(k) - V_tau(k - 1) at node i, time t+tau
# PLen: an N-by-tau_max*n matrix characterizing the interval corresponding to each element of P,
#  with PLen(k, (tau-1)*n+i) = u_tau(k) - u_tau(k - 1) at node i, time t+tau

## Outputs
# Vopt: the optimal value of the current stage problem
# Xopt: an n-by-n matrix with X(i,j) being the amount of demand from i to j satisfied
# Yopt: an n-by-n matrix with Y(i,j) being the amount of reposition of resource from i to j
# end_resource: an n-by-tau_max matrix with the (ith,tau_th)-entry being the amount of resource at node i, time t+tau
# lambda_right: an n-by-tau_max+1 matrix, the right derivative of each resource (the value of
# increasing one unit) at time t+tau, tau = 0,1,...,tau_max
# lambda_left: an n-by-tau_max+1 matrix, the left derivative of each resource (the value of
# decreasing one unit) at time t+tau, tau = 0,1,...,tau_max
def scheduling_mp(n, tau_max, R, Ru, M, W, C, tauX, tauY, P, PLen, rep_matrix):

    # This part exploits the sparsity of demand (thus sparsity of X)
    M_row_num, M_col_num, M_coeff = sp.sparse.find(M)
    num_nonzero_M = len(M_row_num)
    W_coeff = np.zeros(num_nonzero_M)
    for i in range(num_nonzero_M):
        W_coeff[i] = W[M_row_num[i]][M_col_num[i]]

    # construct the coefficient matrix for X (need to find the entries that correspond to each row/column)
    row_sum_matrix_X = coo_matrix((np.ones(num_nonzero_M), (M_row_num, range(num_nonzero_M))), shape=(n, num_nonzero_M))

    col_sum_matrix_X = np.zeros((tau_max*n, num_nonzero_M))
    for i in range(num_nonzero_M):
        tau = tauX[M_row_num[i]][M_col_num[i]]
        if tau < 0 or tau > tau_max:
            print >> sys.stderr, 'Contradiction in definitions of M and tauX!'
            return -1
        else:
            if tau == 0:
                tau = 1
            # for the column - summing vector, write it vertically separated by values of corresponding tau
            col_sum_matrix_X[(tau - 1)*n+M_col_num[i]][i] = 1

    # This part exploits the sparcity of the repositioning matrix, where rep_matrix has the only feasible reposition movements
    rep_row_num, rep_col_num, C_coeff = sp.sparse.find(rep_matrix)
    num_nonzero_rep = len(rep_row_num)
    for i in range(num_nonzero_rep):
        C_coeff[i] = C[rep_row_num[i]][rep_col_num[i]]

    # construct the coefficient matrix for Y (need to find the entries those correspond to each row/column
    row_sum_matrix_Y = coo_matrix((np.ones(num_nonzero_rep), (rep_row_num, range(num_nonzero_rep))), shape=(n, num_nonzero_rep))

    col_sum_matrix_Y = np.zeros((tau_max * n, num_nonzero_rep))
    for i in range(num_nonzero_rep):
        tau = tauY[rep_row_num[i]][rep_col_num[i]]
        if tau < 0 or tau > tau_max:
            print >> sys.stderr, 'Contradiction in definitions of rep_matrix and tauY!'
            return -1
        else:
            # for reposition, we have tau = 0 cases; just treat them as tau = 1 cases
            if tau == 0:
                tau = 1
            # for the column - summing vector, write it vertically separated by values of corresponding tau
            col_sum_matrix_Y[(tau - 1)*n+rep_col_num[i]][i] = 1

    # initialize results
    Xopt = np.zeros((n, n))
    Yopt = np.zeros((n, n))
    end_resource = np.zeros((n, tau_max))
    lambda_right = np.zeros((n, tau_max + 1))
    lambda_left = np.zeros((n, tau_max + 1))

    ## Generating optimal flow and right derivative
    # R_right = R
    R_right = R + 0.0001*np.ones((n,1))
    Ru_right = np.reshape(Ru, (tau_max*n, 1), order='F') + 0.0001 * np.ones((tau_max * n, 1))

    # Construct the problem.
    X_right = cvx.Variable(num_nonzero_M, 1)
    Y_right = cvx.Variable(num_nonzero_rep, 1)
    Z_right = cvx.Variable(P.shape[0], tau_max*n)

    obj_right = W_coeff*X_right - C_coeff*Y_right + cvx.sum(cvx.multiply(P, Z_right))

    cons_right = [R_right == row_sum_matrix_X*X_right + row_sum_matrix_Y*Y_right,
                  cvx.sum(Z_right, axis=0).T == col_sum_matrix_X * X_right + col_sum_matrix_Y * Y_right + Ru_right,
                  0 <= X_right, X_right <= M_coeff, 0 <= Y_right, 0 <= Z_right, Z_right <= PLen]

    prob_right = cvx.Problem(cvx.Maximize(obj_right), cons_right)

    # Solve with ECOS.
    # prob_right.solve(solver=cvx.ECOS_BB) #, mi_max_iters=100
    prob_right.solve(solver=cvx.ECOS)

    Vopt = prob_right.value
    Xval = np.array(X_right.value)
    Yval = np.array(Y_right.value)
    Zval = np.array(Z_right.value)

    if num_nonzero_M == 1:
        Xopt[M_row_num[0]][M_col_num[0]] = np.round(Xval)
    else:
        for i in range(num_nonzero_M):
            Xopt[M_row_num[i]][M_col_num[i]] = np.round(Xval[i][0])
    if num_nonzero_rep == 1:
        Yopt[rep_row_num[0]][rep_col_num[0]] = np.round(Yval)
    else:
        for i in range(num_nonzero_rep):
            Yopt[rep_row_num[i]][rep_col_num[i]] = np.round(Yval[i][0])
    dual_right = np.array(cons_right[0].dual_value)
    for i in range(n):
        lambda_right[i][0] = dual_right[i][0]
    dual_right = np.array(cons_right[1].dual_value)
    for tau in range(tau_max):
        for i in range(n):
            end_resource[i][tau] = np.sum(Zval[:, tau*n+i])
            lambda_right[i][1+tau] = dual_right[tau*n+i][0]

    ## Generating left derivative
    small = 1.0e-6
    R_left = np.maximum(R - 0.0001 * np.ones((n, 1)), small*np.ones((n, 1))) # R_left should be positive
    Ru_left = np.maximum(np.reshape(Ru, (tau_max*n, 1), order='F') - 0.0001 * np.ones((n, tau_max)), small*np.ones((n, tau_max)))


    # Construct the problem.
    X_left = cvx.Variable(num_nonzero_M, 1)
    Y_left = cvx.Variable(num_nonzero_rep, 1)
    Z_left = cvx.Variable(P.shape[0], tau_max*n)

    obj_left = W_coeff*X_left - C_coeff*Y_left + cvx.sum(cvx.multiply(P, Z_left))

    cons_left = [R_left == row_sum_matrix_X*X_left + row_sum_matrix_Y*Y_left,
                  cvx.sum(Z_left, axis=0).T == col_sum_matrix_X * X_left + col_sum_matrix_Y * Y_left + Ru_left,
                  0 <= X_left, X_left <= M_coeff, 0 <= Y_left, 0 <= Z_left, Z_left <= PLen]

    prob_left = cvx.Problem(cvx.Maximize(obj_left), cons_left)

    # Solve with ECOS.
    # prob_left.solve(solver=cvx.ECOS_BB) #, mi_max_iters=100
    prob_left.solve(solver=cvx.ECOS)

    dual_left = np.array(cons_left[0].dual_value)
    for i in range(n):
        lambda_left[i][0] = dual_left[i][0]
    dual_left = np.array(cons_left[1].dual_value)
    for tau in range(tau_max):
        for i in range(n):
            lambda_left[i][1+tau] = dual_left[tau*n+i][0]

    return Vopt, Xopt, Yopt, end_resource, lambda_right, lambda_left, prob_right.status, prob_left.status


## solve the scheduling with sparse input
# param_job: a matrix with rows consisting by [M_row_num, M_col_num, M_coeff, W_coeff, tauX_coeff]; if there is no job demand, param_job = None
# param_rep: a matrix with rows consisting by [rep_row_num, rep_col_num, C_coeff, tauY_coeff]
def scheduling_mp_sparse(n, tau_max, R, Ru, param_job, param_rep, P, PLen):

    if param_job is None or param_job.shape[1] == 0:
        M_is_empty = True
    else:
        M_is_empty = False

    if not M_is_empty:
        M_row_num = param_job[0, :].flatten().astype(int)  # row indices of positive M elements
        M_col_num = param_job[1, :].flatten().astype(int)  # col indices of positive M elements
        M_coeff = param_job[2, :].flatten()  # positive M elements
        W_coeff = param_job[3, :].flatten()  # W elements corresponding to positive M elements
        tauX_coeff = param_job[4, :].flatten().astype(int)  # tau corresponding to positive M elements

        num_nonzero_M = len(M_coeff)

        # construct the coefficient matrix for X (need to find the entries that correpsond to each row/column)
        row_sum_matrix_X = sp.sparse.coo_matrix((np.ones(num_nonzero_M), (M_row_num, range(num_nonzero_M))), shape=(n, num_nonzero_M))

        col_sum_matrix_X = np.zeros((tau_max*n, num_nonzero_M))
        for i in range(num_nonzero_M):
            tau = tauX_coeff[i]
            if tau < 0 or tau > tau_max:
                print >> sys.stderr, 'Contradiction in definitions of M and tauX!'
                return -1
            else:
                if tau == 0:
                    tau = 1
                # for the column - summing vector, write it vertically separated by values of corresponding tau
                col_sum_matrix_X[(tau - 1)*n+M_col_num[i]][i] = 1

    rep_row_num = param_rep[0, :].flatten().astype(int)  # row indices of positive rep_matrix elements
    rep_col_num = param_rep[1, :].flatten().astype(int)  # col indices of positive rep_matrix elements
    C_coeff = param_rep[2, :].flatten()  # C elements corresponding to positive rep_matrix elements
    tauY_coeff = param_rep[3, :].flatten().astype(int)  # tau corresponding to positive rep_matrix elements

    num_nonzero_rep = len(C_coeff)

    # construct the coefficient matrix for Y (need to find the entries those correspond to each row/column
    row_sum_matrix_Y = coo_matrix((np.ones(num_nonzero_rep), (rep_row_num, range(num_nonzero_rep))), shape=(n, num_nonzero_rep))

    col_sum_matrix_Y = np.zeros((tau_max * n, num_nonzero_rep))
    for i in range(num_nonzero_rep):
        tau = tauY_coeff[i]
        if tau < 0 or tau > tau_max:
            print >> sys.stderr, 'Contradiction in definitions of rep_matrix and tauY!'
            return -1
        else:
            # for reposition, we have tau = 0 cases; just treat them as tau = 1 cases
            if tau == 0:
                tau = 1
            # for the column - summing vector, write it vertically separated by values of corresponding tau
            col_sum_matrix_Y[(tau - 1)*n+rep_col_num[i]][i] = 1

    # initialize results
    Xopt = np.zeros((n, n))
    Yopt = np.zeros((n, n))
    end_resource = np.zeros((n, tau_max))
    lambda_right = np.zeros((n, tau_max + 1))
    lambda_left = np.zeros((n, tau_max + 1))

    ## Generating optimal flow and right derivative
    R_right = R + 0.0001*np.ones((n,1))
    Ru_right = np.reshape(Ru, (tau_max*n, 1), order='F') + 0.0001 * np.ones((tau_max * n, 1))

    # Construct the problem.
    if not M_is_empty:
        X_right = cvx.Variable(num_nonzero_M, 1)
    Y_right = cvx.Variable(num_nonzero_rep, 1)
    Z_right = cvx.Variable(P.shape[0], tau_max*n)

    obj_right = - C_coeff*Y_right + cvx.sum_entries(cvx.mul_elemwise(P, Z_right))
    if not M_is_empty:
        obj_right += W_coeff * X_right

    if M_is_empty:
        cons_right = [R_right == row_sum_matrix_Y*Y_right,
                      cvx.sum_entries(Z_right, axis=0).T == col_sum_matrix_Y * Y_right + Ru_right,
                      0 <= Y_right, 0 <= Z_right, Z_right <= PLen]
    else:
        cons_right = [R_right == row_sum_matrix_X*X_right + row_sum_matrix_Y*Y_right,
                      cvx.sum_entries(Z_right, axis=0).T == col_sum_matrix_X * X_right + col_sum_matrix_Y * Y_right + Ru_right,
                      0 <= X_right, X_right <= M_coeff, 0 <= Y_right, 0 <= Z_right, Z_right <= PLen]

    prob_right = cvx.Problem(cvx.Maximize(obj_right), cons_right)

    # Solve with ECOS.
    #prob_right.solve(solver=cvx.ECOS_BB) #, mi_max_iters=100
    prob_right.solve(solver=cvx.ECOS)

    Vopt = prob_right.value
    if not M_is_empty:
        Xval = np.array(X_right.value)
    Yval = np.array(Y_right.value)
    Zval = np.array(Z_right.value)

    if not M_is_empty:
        if num_nonzero_M == 1:
            Xopt[M_row_num[0]][M_col_num[0]] = np.round(Xval)
        else:
            for i in range(num_nonzero_M):
                Xopt[M_row_num[i]][M_col_num[i]] = np.round(Xval[i][0])
    if num_nonzero_rep == 1:
        Yopt[rep_row_num[0]][rep_col_num[0]] = np.round(Yval)
    else:
        for i in range(num_nonzero_rep):
            Yopt[rep_row_num[i]][rep_col_num[i]] = np.round(Yval[i][0])
    dual_right = np.array(cons_right[0].dual_value)
    for i in range(n):
        lambda_right[i][0] = dual_right[i][0]
    dual_right = np.array(cons_right[1].dual_value)
    for tau in range(tau_max):
        for i in range(n):
            end_resource[i][tau] = np.sum(Zval[:, tau*n+i])
            lambda_right[i][1+tau] = dual_right[tau*n+i][0]

    ## Generating left derivative
    small = 1.0e-6
    R_left = np.maximum(R - 0.0001 * np.ones((n, 1)), small*np.ones((n, 1))) # R_left should be positive
    Ru_left = np.maximum(np.reshape(Ru, (tau_max*n, 1), order='F') - 0.0001 * np.ones((tau_max*n, 1)), small*np.ones((tau_max*n, 1)))

    # Construct the problem.
    if not M_is_empty:
        X_left = cvx.Variable(num_nonzero_M, 1)
    Y_left = cvx.Variable(num_nonzero_rep, 1)
    Z_left = cvx.Variable(P.shape[0], tau_max*n)

    obj_left =  - C_coeff*Y_left + cvx.sum_entries(cvx.mul_elemwise(P, Z_left))
    if not M_is_empty:
        obj_left += W_coeff*X_left

    if M_is_empty:
        cons_left = [R_left == row_sum_matrix_Y*Y_left,
                      cvx.sum_entries(Z_left, axis=0).T == col_sum_matrix_Y * Y_left + Ru_left,
                      0 <= Y_left, 0 <= Z_left, Z_left <= PLen]
    else:
        cons_left = [R_left == row_sum_matrix_X*X_left + row_sum_matrix_Y*Y_left,
                      cvx.sum_entries(Z_left, axis=0).T == col_sum_matrix_X * X_left + col_sum_matrix_Y * Y_left + Ru_left,
                      0 <= X_left, X_left <= M_coeff, 0 <= Y_left, 0 <= Z_left, Z_left <= PLen]

    prob_left = cvx.Problem(cvx.Maximize(obj_left), cons_left)

    # Solve with ECOS.
    # prob_left.solve(solver=cvx.ECOS_BB) #, mi_max_iters=100
    prob_left.solve(solver=cvx.ECOS)

    dual_left = np.array(cons_left[0].dual_value)
    for i in range(n):
        lambda_left[i][0] = dual_left[i][0]
    dual_left = np.array(cons_left[1].dual_value)
    for tau in range(tau_max):
        for i in range(n):
            lambda_left[i][1+tau] = dual_left[tau*n+i][0]

    return Vopt, Xopt, Yopt, end_resource, lambda_right, lambda_left, prob_right.status, prob_left.status

