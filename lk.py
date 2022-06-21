#!/usr/bin/env python

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Local Library
import mivp

# ------------------------------------------------------------------------------

# Vector field
alpha = 2 / 3
beta = 4 / 3
delta = gamma = 1.0


def fun(t, xy):
    x, y = xy
    u = alpha * x - beta * x * y
    v = delta * x * y - gamma * y
    return np.array([u, v])


# Time span & frame rate
t_span = (0.0, 20.0)

df = 60.0
dt = 1.0 / df
t = np.arange(t_span[0], t_span[1], dt)
t = np.r_[t, t_span[1]]

# Initial set boundary
y0 = [1.0, 0.5]
radius = 0.25
n = 100
xc, yc = y0
# y0s = np.array(
#     [
#         [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
#         for theta in np.linspace(0, 2 * np.pi, n)
#     ]
# )


def vectorize(fun):
    return np.vectorize(fun, signature="()->(n)")

@vectorize
def boundary(s):
    theta = 2 * np.pi * s
    return np.array([
        xc + radius * np.cos(theta), 
        yc + radius * np.sin(theta)
    ])


# Precision
rtol = 1e-9  # default: 1e-3
atol = 1e-15  # default: 1e-6

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
    boundary_sampling=3,
    boundary_rtol=0.0,
    boundary_atol=0.01,
    rtol=rtol,
    atol=atol,
    method="LSODA",
)

mivp.generate_movie(data, filename="lk.mp4", fps=df)
