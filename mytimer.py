import time


def timeit(func=None, loops=1, verbose=False):
    if func is not None:
        def inner(*args, **kwargs):

            sums = 0.0
            mins = 1.7976931348623157e+308
            maxs = 0.0
            print '====%s Timing====' % func.__name__
            for i in range(0, loops):
                t0 = time.time()
                result = func(*args, **kwargs)
                dt = time.time() - t0
                mins = dt if dt < mins else mins
                maxs = dt if dt > maxs else maxs
                sums += dt
                if verbose is True:
                    print "\t%r ran in %2.9f sec on run %s" % (
                        func.__name__, dt, i)
            print "%r min run time was %2.9f sec" % (func.__name__, mins)
            print "%r max run time was %2.9f sec" % (func.__name__, maxs)
            print "%r avg run time was %2.9f sec in %s runs" % (
                func.__name__, sums/loops, loops)
            print "==== end ===="
            return result

        return inner
    else:
        def partial_inner(func):
            return timeit(func, loops, verbose)
        return partial_inner
