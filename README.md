# ODE flow & sets of initial values

[Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):

$$ 
\begin{array}{lll}
\dot{x} &=& \alpha x - \beta x y \\
\dot{y} &=& \delta x y -\gamma y
\end{array}
$$


with $\alpha = 2 /3, \beta = 4 / 3, \delta =1, \gamma = 1.0$.


https://user-images.githubusercontent.com/1217694/174568980-8ebd31f5-6cf1-4359-82e0-61d8a6a37d02.mp4

https://user-images.githubusercontent.com/1217694/174569344-3081bebc-68b3-47a2-9682-cb9a0658a8d7.mp4

```python
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
y0s = np.array(
    [
        [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
        for theta in np.linspace(0, 2 * np.pi, n)
    ]
)

# Precision
rtol = 1e-6  # default: 1e-3
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
mivp.generate_movie(data, filename="movie.mp4", fps=df)
```
