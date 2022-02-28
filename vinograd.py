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
    q = x**2 + y**2 * (1 + (x**2 + y**2) ** 2)
    dx = (x**2 * (y - x) + y**5) / q
    dy = y**2 * (y - 2 * x) / q
    return [dx, dy]


# Time span & frame rate
t_span = (0.0, 10.0)

df = 60.0
dt = 1.0 / df
t = np.arange(t_span[0], t_span[1], dt)
t = np.r_[t, t_span[1]]

# Initial set boundary
y0 = [0.0, 0.0]
radius = 0.5
n = 50000
xc, yc = y0
y0s = np.array(
    [
        [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
        for theta in np.linspace(0, 2 * np.pi, n)
    ]
)

# Precision
rtol = 1e-9  # default: 1e-3
atol = 1e-12  # default: 1e-6

# ------------------------------------------------------------------------------

results = mivp.solve(
    fun=fun,
    t_span=t_span,
    y0s=y0s,
    rtol=rtol,
    atol=atol,
)
data = mivp.get_data(results, t)
mivp.generate_movie(data, filename="vinograd.mp4", fps=df)
