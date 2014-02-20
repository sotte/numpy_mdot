# Tired of numpy.dot?

Note: this is pre alpha!

TODO change that!

TODO Ideally this thould be integrated in numpy.

`mdot` chains multiplication calls and allows you to write
```python
mdot(A, B, C, D)
```
instead of
```python
np.dot(np.dot(np.dot(A, B), C), D)
A.dot(B).dot(C).dot(D)
```

Did I mention that it automatically speeds up the multiplication by setting the
parens in an optimal fashion:
```python
>>> %timeit np.dot(np.dot(np.dot(A, B), C), D)
1 loops, best of 3: 694 ms per loop
>>> %timeit mdot(A, B, C, D)
100 loops, best of 3: 5.18 ms per loop
```

Still, not satisfied? Get red rid of the overhead for calculating the optimal
parens once and then use the expression:
```python
>>> print_optimal(D, A, B, C, names=list("DABC"))
"np.dot(np.dot(D, np.dot(A, B)), C)"
```
