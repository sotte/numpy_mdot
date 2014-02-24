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
    n = len(args)
    # costs for subproblems
    m = np.zeros((n, n), dtype=np.int)
    # helper to actually multiply optimal solution
    s = np.zeros((n, n), dtype=np.int)
    for i in range(n):
        for j in range(i+1, n):
            cost, k = min((m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1], k)
                          for k in range(i, j))
            m[i, j] = cost
            s[i, j] = k

    return m, s


def chain_order_for_three(A, B, C, evaluate=True):
    """Determine the optimal parenthesizations for three arrays.

    Doing in manually instead is approximately 15 times faster.

    """
    # cost1 = cost((AB)C)
    cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
             A.shape[0] * B.shape[1] * C.shape[1])   # TMP C
    # cost2 = cost((AB)C)
    cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  # (BC)
             A.shape[0] * A.shape[1] * C.shape[1])   # A TMP

    if evaluate is True:
        if cost1 < cost2:
            return np.dot(np.dot(A, B), C)
        else:
            return np.dot(A, np.dot(B, C))


#@mytimer.timeit
def multiply_r(args, s, i, j):
    if i == j:
        return args[i]
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
    """Dot product of multiple arrays.

    `mdot` chains `numpy.dot` and uses an optimal parenthesizations of
    the matrices [1]_ [2]_. Depending on the shape of the matrices this can
    speed up the multiplication a lot.

    Think of `mdot` as::

        def mdot(*args): return reduce(numpy.dot, args)


    Parameters
    __________
    *args : multiple arrays

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays

    See Also
    --------
    dot : dot multiplication with two arguments.

    References
    ----------
    TODO extend and format

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] http://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Examples
    --------
    `mdot` allows you to write
    >>> import numpy as np
    >>> A = np.random.random(10000, 100)
    >>> B = np.random.random(100, 1000)
    >>> C = np.random.random(1000, 5)
    >>> D = np.random.random(5, 333)
    >>> # the actual dot multiplication
    >>> mdot(A, B, C, D)

    instead of
    >>> np.dot(np.dot(np.dot(A, B), C), D)
    >>> # or
    >>> A.dot(B).dot(C).dot(D)


    Example: cost of different parenthesizations
    --------------------------------------------
    The cost for a matrix multiplication can be calculated with the
    following function::

        def cost(A, B): return A.shape[0] * A.shape[1] * B.shape[1]

    Let's assume we have three matrices
    :math:`A_{10x100}, B_{100x5}, C_{5x50}$`.

    The costs for the two different parenthesizations are as follows::

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
