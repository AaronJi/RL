## the CAVE algorithm
# Aaron Ji

import numpy as np
import sys

## This function performs the CAVE operation for one step. The input and output are
## Inputs:
# A: a matrix containing all the breakpoints and gradient information
# A = [0 s_1
#      u_1 s_2
#        ...
#      u_n, s_{n+1}]
# which means that the piecewise linear function has derivative s_1 from 0
# to u_1, derivative s_2 from u_1 to u_2, ... and derivative s_{n+1} from
# u_n to infinity. It should satisfy 0 < u_1 < u_2 < ... < u_n and s_1 >= s_2
# >= ... >= s_{n+1}. (All u_i's must be integers.)
# newBreakpoint = [w, s-, s+] is the incoming new breakpoint to be added.
# Particularly, the new breakpoint is w (must be an integer) with left derivative being s- and right derivative being s+.
# alpha: the step size
# comparing with the paper of Godfrey, Powell, 2002, the smoothing interval epsilon_minus and epsilon_plus have been set to 1 in this function.
## Outputs: new A after CAVE update
def CAVE(A, newBreakPoint, alpha):

    numInterval = A.shape[0]

    w = int(newBreakPoint[0])
    v_minus = np.max((newBreakPoint[1], 0))
    v_plus = np.max((newBreakPoint[2], 0))

    insert_w = False

    small = 1.0e-6

    # Get the updated A for the range [w - 1, w + 1]: update the derivative to be the weighted
    # average between the old derivative and new derivative; otherwise keep the original derivative
    # Note that A may contain repetitive  breakpoints, and may have nonconcavity
    if A[0, 0] > small or v_minus > small:  # to save time, we do not update zero derivatives by zero derivatives, even when w is nonzero
        for i in range(1, numInterval):

            if w == int(A[i][0]):
                insert_w = True

                A = np.vstack((A, np.zeros((2, 2))))
                if i < numInterval-1:
                    for j in range(numInterval-1, i, -1):
                        A[j+2][0] = A[j][0]
                        A[j+2][1] = A[j][1]

                numInterval += 2

                vi = A[i-1][1]
                vi_1 = A[i][1]

                A[i][0] = w-1
                A[i][1] = alpha * v_minus + (1-alpha) * vi
                A[i+1][0] = w
                A[i+1][1] = alpha * v_plus + (1-alpha) * vi_1
                A[i+2][0] = w+1
                A[i+2][1] = vi_1

                break

            elif w > int(A[i-1][0]) and w < int(A[i][0]):
                insert_w = True

                A = np.vstack((A, np.zeros((3, 2))))
                for j in range(numInterval-1, i-1, -1):
                    A[j+3][0] = A[j][0]
                    A[j+3][1] = A[j][1]

                numInterval += 3

                vi = A[i - 1][1]

                A[i][0] = w - 1
                A[i][1] = alpha * v_minus + (1 - alpha) * vi
                A[i + 1][0] = w
                A[i + 1][1] = alpha * v_plus + (1 - alpha) * vi
                A[i + 2][0] = w + 1
                A[i + 2][1] = vi

                break

    if insert_w:
        # Eliminating repetitive entries backwards. If there are multiple entries at the same breakpoint,
        # take the last one (because the last one is the one for the next range by construction)
        if numInterval > 1:
            newRowIndexList = [numInterval - 1]  # Record the row indexes to be kept; note it is revers

            for i in range(numInterval-2, -1, -1):
                if int(A[i][0]) < int(A[i+1][0]):
                    newRowIndexList.append(i)
                elif int(A[i][0]) == int(A[i+1][0]):
                    pass
                else:
                    print >> sys.stderr, 'Wrong value of A!'

            # remove the repeative elements
            newNumInterval = len(newRowIndexList)
            Anew = np.zeros((newNumInterval, 2))
            i = 0
            for index in reversed(newRowIndexList):
                Anew[i][0] = A[index][0]
                Anew[i][1] = A[index][1]
                i += 1

        # Eliminating nonconcavity by requiring those breakpoints to the left of w to take
        # larger values (sort from right to left), an those that are to the right of w to take smaller values.
        iw = -1
        for i in range(newNumInterval):
            if w == int(Anew[i][0]):
                iw = i
                break
        assert 0 <= iw < newNumInterval

        for i in range(iw, 0, -1):
            if Anew[i - 1][1] < Anew[i][1]:
                Anew[i - 1][1] = Anew[i][1]
        for i in range(iw+1, newNumInterval):
            if Anew[i][1] > Anew[i-1][1]:
                Anew[i][1] = Anew[i - 1][1]

        # continue to shrink breakpoints of A with derivatives too similar
        Anew1 = Anew[0, :]
        for i in range(1, Anew.shape[0]-1):
            if abs(Anew[i,1] - Anew[i-1,1]) > small:
                Anew1 = np.vstack((Anew1, Anew[i, :]))

        Anew1 = np.vstack((Anew1, Anew[-1, :]))

        return Anew1

    else:
        return A

## the function which transfers the v and vLen to A, i.e., the CAVE definition style
## Inputs
# v: an 1-by-N vector characterizing the coefficients of the piecewise
# linear function at the future end node, with v(1, k) = V(k) - V(k - 1)
# vLen: an 1-by-N vector matrix characterizing the interval length of the piecewise
# linear function at the future end node, with vLen(1, k) = u(k) - u(k - 1)
# N: the number of interval in v and vLen
# A: a matrix containing all the breakpoints and gradient information
# A = [0 s_1
#      u_1 s_2
#        ...
#      u_n, s_{n+1}]
# which means that the piecewise linear function has derivative s_1 from 0
# to u_1, derivative s_2 from u_1 to u_2, ... and derivative s_{n+1} from
# u_n to infinity. It should satisfy 0 < u_1 < u_2 < ... < u_n and s_1 >= s_2
# >= ... >= s_{n+1}. (All u_i's must be integers.)
def v2A(v, vLen, N):

    A = np.zeros((N+1, 2))

    u = 0
    for k in range(N):
        A[k][0] = u
        A[k][1] = v[k]
        u += int(vLen[k])

    A[N][0] = u
    A[N][1] = 0.0

    return A

## the function which transfers A to v and vLen, i.e., the v3 definition style
def A2v(A):
    numInterval = A.shape[0]
    N = numInterval - 1
    v = np.zeros(N)
    vLen = np.zeros(N)

    for k in range(N):
        vLen[k] = int(A[k+1][0] - A[k][0])
        v[k] = A[k][1]

    return v, vLen, N


