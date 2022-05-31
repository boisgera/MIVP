#!/usr/bin/env python

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Local Library
import mivp

# ------------------------------------------------------------------------------

# Vector field
def fun(t, xy):
    x, y = xy
    r = np.sqrt(x * x + y * y)
    dx = x + x * y - (x + y) * r
    dy = y - x * x + (x - y) * r
    return [dx, dy]


# Time span & frame rate
t_span = (0.0, 10.0)

df = 60.0
dt = 1.0 / df
t = np.arange(t_span[0], t_span[1], dt)
t = np.r_[t, t_span[1]]

# Initial set boundary
y0 = [1.0, 0.0]
radius = 0.5
n = 10000
xc, yc = y0
# y0s = np.array(
#     [
#         [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
#         for theta in np.linspace(0, 2 * np.pi, n)
#     ]
# )


def boundary(t):  # we assume that t is a 1-dim array
    return np.array(
        [
            [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
            for theta in 2 * np.pi * t
        ]
    )


# Precision
rtol = 1e-6  # default: 1e-3
atol = 1e-12  # default: 1e-6

# ------------------------------------------------------------------------------

# results = mivp.solve(
#     fun=fun,
#     t_span=t_span,
#     y0s=y0s,
#     rtol=rtol,
#     atol=atol,
#     method="LSODA",
# )
# data = mivp.get_data(results, t)

data = mivp.solve(
    fun=fun,
    t_eval=t,
    boundary=boundary,
    boundary_rtol=0.0,
    boundary_atol=0.1,
    rtol=rtol,
    atol=atol,
    method="LSODA",
)
mivp.generate_movie(data, filename="main.mp4", fps=df)
