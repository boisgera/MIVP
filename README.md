# ODE flow & sets of initial values

Flow of the [Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations),
governed by

$$ 
\begin{array}{rcl}
\dot{x} &=& \alpha x - \beta x y \\
\dot{y} &=& \delta x y -\gamma y
\end{array}
$$

Here with $\alpha = 2 /3, \beta = 4 / 3, \delta =1, \gamma = 1.0$.


https://user-images.githubusercontent.com/1217694/174568980-8ebd31f5-6cf1-4359-82e0-61d8a6a37d02.mp4


```python
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


def boundary(t):  # we assume that t is a 1-dim array
    return np.array(
        [
            [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
            for theta in 2 * np.pi * t
        ]
    )


# Precision
rtol = 1e-9  # default: 1e-3
atol = 1e-15  # default: 1e-6

# ------------------------------------------------------------------------------
data = mivp.solve(
    fun=fun,
    t_eval=t,
    boundary=boundary,
    boundary_rtol=0.0,
    boundary_atol=0.01,
    rtol=rtol,
    atol=atol,
    method="LSODA",
)

mivp.generate_movie(data, filename="lk.mp4", fps=df)
```


Flow of the [Vinograd system](http://www.mathnet.ru/php/getFT.phtml?jrnid=dan&paperid=21930&what=fullt&option_lang=eng), governed by

$$
\begin{array}{rcl}
\dot{x} &=& (x^2 (y - x) + y^5) / q(x, y) \\
\dot{y} &=& y^2 (y - 2 x) / q(x, y)
\end{array}
\\; \mbox{ with } \\;
q(x, y) = x^2 + y^2 + (1 + (x^2 + y^2)^2)
$$

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


def boundary(t):  # we assume that t is a 1-dim array
    return np.array(
        [
            [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
            for theta in 2 * np.pi * t
        ]
    )


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

mivp.generate_movie(data, filename="vinograd.mp4", fps=df)
```
