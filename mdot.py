from __future__ import print_function

import numpy as np


###############################################################################
# Logic
def mdot(*args, **kwargs):
    """
    Dot product of multiple arrays with `ndim >= 2`.

    `mdot` chains `numpy.dot` and uses an optimal parenthesizations of
    the matrices [1]_ [2]_. Depending on the shape of the matrices this can
    speed up the multiplication a lot.

    Think of `mdot` as::

        def mdot(*args): return reduce(numpy.dot, args)


    Parameters
    __________
    *args : multiple arrays

    optimize : bool (default: True)
        Use optimization if True, otherwise use `reduce(np.dot, args)`

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
    `mdot` allows you to write::

    >>> import numpy as np
    >>> # Prepare some data
    >>> A = np.random.random(10000, 100)
    >>> B = np.random.random(100, 1000)
    >>> C = np.random.random(1000, 5)
    >>> D = np.random.random(5, 333)
    >>> # the actual dot multiplication
    >>> mdot(A, B, C, D)

    instead of::

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
    for array in args:
        if array.ndim != 2:
            raise ValueError("ndim for each argument must be 2.")

    optimize = kwargs.get("optimize", True)

    n = len(args)
    # optimization only makes sense for len(args) > 2
    if n == 1:
        return args[0]
    elif n == 2:
        return np.dot(args[0], args[1])

    if optimize:
        # _mdot_three is much faster than _optimum_order
        if n == 3:
            return _mdot_three(args[0], args[1], args[2])
        else:
            order = _optimum_order(args)
            return _mdot(args, order, 0, n - 1)
    else:
        return reduce(np.dot, args)


def print_optimal_chain_order(*args, **kwargs):
    """
    Print the optimal chain of multiplications that minimizes the total
    number of multiplications.

    This is just a temporary helper function that will be deleted eventually.

    """
    names = kwargs.get("names", None)
    order = _optimum_order(args)
    print(_order_to_str(args, order, 0, len(args) - 1, names=names))


###############################################################################
# Internal stuff
def _mdot_three(A, B, C):
    """
    mdot for three arrays.

    Doing in manually instead of using dynamic programing is approximately 15
    times faster due to the overhead.

    """
    # cost1 = cost((AB)C)
    cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
             A.shape[0] * B.shape[1] * C.shape[1])   # (--)C
    # cost2 = cost((AB)C)
    cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  #  (BC)
             A.shape[0] * A.shape[1] * C.shape[1])   # A(--)

    if cost1 < cost2:
        return np.dot(np.dot(A, B), C)
    else:
        return np.dot(A, np.dot(B, C))


def _optimum_order(args):
    """
    Return a np.array which encodes the opimal order of mutiplications.

    This follows Cormen.

    cost[i, k ] = min([cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
                       for k in range(i, j)])
    m[i, k ] = min([m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1]
                    for k in range(i, j)])

    """
    # p is the list of the row length of all matrices plus the column of the
    # last matrix
    #
    # Example:
    # A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    # The cost for multiplying AB is then: 10 * 100 * 5
    p = [arg.shape[0] for arg in args]
    p.append(args[-1].shape[1])

    # determine the order of the multiplication using dynamic programing
    n = len(args)
    # costs for subproblems
    m = np.zeros((n, n), dtype=np.int)
    # helper to actually multiply optimal solution
    order = np.zeros((n, n), dtype=np.int)
    for i in range(n):
        for j in range(i+1, n):
            cost, k = min((m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1], k)
                          for k in range(i, j))
            m[i, j] = cost
            order[i, j] = k

    return order


def _mdot(args, order, i, j):
    if i == j:
        return args[i]
    else:
        return np.dot(_mdot(args, order, i, order[i, j]),
                      _mdot(args, order, order[i, j] + 1, j))


def _order_to_str(args, order, i, j, names=None):
    """This is just a helper function to print the parens."""
    if i == j:
        if names:
            return names[int(i)]
        else:
            return "M_{}".format(int(i))
    else:
        return "np.dot({}, {})".format(
            _order_to_str(args, order, i, order[i, j], names),
            _order_to_str(args, order, order[i, j] + 1, j, names)
        )
