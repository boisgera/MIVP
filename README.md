# ODE flow & sets of initial values

<!--
<div style="display:flex;gap:1em;flex: 0 1 auto;">
<div style="box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;">
<a href="https://github.com/boisgera/MIVP/raw/gh-pages/movie.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/movie.png">
</img>
</a>
</div>
<div style="box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;">
<a href="https://github.com/boisgera/MIVP/raw/gh-pages/lk.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/lk.png">
</img>
</a>
</div>
<div style="box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;">
<a href="https://github.com/boisgera/MIVP/raw/gh-pages/vinograd.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/vinograd.png">
</img>
</a>
</div>
</div>
-->

https://raw.githubusercontent.com/boisgera/MIVP/gh-pages/movie.mp4

https://raw.githubusercontent.com/boisgera/MIVP/gh-pages/lk.mp4

https://raw.githubusercontent.com/boisgera/MIVP/gh-pages/vinograd.mp4


<a href="https://github.com/boisgera/MIVP/raw/gh-pages/movie.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/movie.png"
style="width:100%">
</img>
</a>

<a href="https://github.com/boisgera/MIVP/raw/gh-pages/lk.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/lk.png"
style="width:100%">
</img>
</a>

<a href="https://github.com/boisgera/MIVP/raw/gh-pages/vinograd.mp4">
<img
src="https://github.com/boisgera/MIVP/raw/gh-pages/images/vinograd.png"
style="width:100%">
</img>
</a>

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
