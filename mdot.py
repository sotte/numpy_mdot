from __future__ import print_function

import numpy as np
import mytimer


#@mytimer.timeit
def chain_order_rec(args):
    """
    cost[i, k ] = min([cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
                       for k in range(i, j)])
    m[i, k ] = min([m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1]
                    for k in range(i, j)])

    """
    # p is the list of the row length of all matrices plus the column of the
    # last matrix
    # example
    # A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    # The cost for multipying AB is then: 10 * 100 * 5
    p = [arg.shape[0] for arg in args]
    p.append(args[-1].shape[1])

    # determine the order of the multiplication using DP
    n = len(p) - 1
    # costs for subproblems
    m = np.zeros((n, n))
    # helper to actually multiply optimal solution
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            cost, k = min((m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1], k)
                          for k in range(i, j))
            m[i, j] = cost
            s[i, j] = k

    return m, s


#@mytimer.timeit
def multiply_r(args, s, i, j):
    if i == j:
        return args[int(i)]
    else:
        return np.dot(multiply_r(args, s, i, s[i, j]),
                      multiply_r(args, s, s[i, j] + 1, j))


def _print_parens(args, s, i, j, names=None):
    if i == j:
        if names:
            print(names[int(i)], end="")
        else:
            str_ = "M_{}".format(int(i))
            print(str_, end="")
    else:
        print("np.dot(", end="")
        _print_parens(args, s, i, s[i, j], names)
        print(", ", end="")
        _print_parens(args, s, s[i, j] + 1, j, names)
        print(")", end="")


def print_optimal(*args, **kwargs):
    """Print the optimal chain of multiplications that minimizes the total
    number of multiplications.

    """
    names = kwargs.get("names", None)
    m, s = chain_order_rec(args)
    _print_parens(args, s, 0, len(args) - 1, names=names)


def mdot(*args, **kwargs):
    """Multiply the given arrays.

    `optimize` = True

    TODO extend and document.

    Minimize the number of required scalar multiplications for the given
    matrices.

    Example for the costs:
    A_{10x100}, B_{100x5}, C_{5x50}

    cost((AB)C) = 5000 + 2500 = 7500
    cost(A(BC)) = 50000 + 25000 = 75000

    """
    if len(args) == 1:
        return args[0]

    optimize = kwargs.get("optimize", True)

    if optimize:
        m, s = chain_order_rec(args)
        return multiply_r(args, s, 0, len(args) - 1)
    else:
        return reduce(np.dot, args)
