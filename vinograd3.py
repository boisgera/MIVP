#!/usr/bin/env python

# Python Standard Library
pass

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Local Library
import mivp

# ------------------------------------------------------------------------------

# Vector field
def fun(t, xy):
    x, y = xy
    q = x**2 + y**2 * (1 + (x**2 + y**2) ** 2)
    dx = (x**2 * (y - x) + y**5) / q
    dy = y**2 * (y - 2 * x) / q
    return [dx, dy]


# Time span & frame rate
t_span = (0.0, 20.0)

df = 60.0
dt = 1.0 / df
t = np.arange(t_span[0], t_span[1], dt)
t = np.r_[t, t_span[1]]

# Initial set boundary
y0 = [0.0, 0.0]
radius = 0.5
n = 10
xc, yc = y0


def vectorize(fun):
    return np.vectorize(fun, signature="()->(n)")

@vectorize
def cube(s):
    if 0 <= s < 0.25:
        return np.array([4 * s, 1])
    elif 0.25 <= s < 0.5:
        return np.array([1.0, 1 - 4 * (s - 0.25)])
    elif 0.5 <= s < 0.75:
        return np.array([1.0 - 4 * (s - 0.5), 0])
    else:
        return np.array([0.0, 4 * (s - 0.75)])


boundary = lambda s: 1.0 * cube(s) + np.array([-0.0001, -0.5])


# Precision
rtol = 1e-9  # default: 1e-3
atol = 1e-12  # default: 1e-6

# ------------------------------------------------------------------------------
data = mivp.solve(
    fun=fun,
    t_eval=t,
    boundary=boundary,
    boundary_rtol=0.0,
    boundary_atol=0.05,
    rtol=rtol,
    atol=atol,
    method="LSODA",
)

mivp.generate_movie(data, filename="vinograd3.mp4", fps=df)
