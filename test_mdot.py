from mdot import mdot
import numpy as np


##############################################################################
def test_unoptimized_one_parameter():
    I = np.eye(3, 3)
    assert (mdot(I, optimize=False) == I).all()


def test_unoptimized_multiple_parameters():
    I = np.eye(3, 3)
    assert (mdot(I, I, optimize=False) == I).all()
    assert (mdot(I, I, I, optimize=False) == I).all()


def test_unoptimized_fancy():
    A = np.random.random((3, 3))
    B = np.linalg.inv(A)
    I = np.eye(3)
    assert np.allclose(mdot(A, B, optimize=False), I)


##############################################################################
def test_optimized_general():
    I = np.eye(3, 3)
    assert np.allclose(mdot(I, I, I, I, I, I, I, I, optimize=True), I)


def test_optimized_fancy():
    A = np.random.random((3, 3))
    B = np.linalg.inv(A)
    I = np.eye(3)
    assert np.allclose(mdot(A, B, I), I)
    print mdot(A, B, I, optimize=True).shape
