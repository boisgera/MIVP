import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy
import scipy.integrate

def fun(t, xy): # Vinograd
    _ = t
    x, y = xy
    q = x**2 + y**2 * (1 + (x**2 + y**2)**2)
    dx = (x**2 * (y - x) + y**5) / q
    dy = y**2 * (y - 2*x) / q
    return np.array([dx, dy])

def forward(fun, t_span, y0, **options):
    yf = np.zeros_like(y0)
    tf = t_span[1]
    for i, y0_ in enumerate(y0):
        r = scipy.integrate.solve_ivp(fun, t_span, y0_, dense_output=True, **options)
        yf[i] = r.sol(tf)
    return np.array(yf)

def resample(xy):
    n = len(xy)
    t = np.linspace(0.0, 1.0, n)
    x, y = xy.T
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    d2x = np.gradient(dx, t)
    d2y= np.gradient(dy, t)
    ds2 = dx * dx + dy * dy
    ds = np.sqrt(ds2)
    _c = np.abs(dx * d2y - dy * d2x) / ds ** 3 # curvature

    # Adaptive sampling density
    density = ds # 1 / (1 + 10 * curvature)  # C = 10 controls adaptation
    cum_density = np.cumsum(density)
    cum_density /= cum_density[-1]

    # Resample at new adaptive points
    cs = CubicSpline(t,  xy, bc_type="periodic")
    t_new = np.interp(t, cum_density, t)
    xy_new = cs(t_new)
    return xy_new


n = 100

t = np.linspace(0.0, 1.0, n)
x0 = np.cos(2 * np.pi * t)
y0 = np.sin(2 * np.pi * t)
# Patch for exact periodicity
x0[-1] = x0[0]
y0[-1] = y0[0]
xy0 = np.array([x0, y0])

# xy1 = forward(fun, t_span, xy0.T).T
# xy2 = resample(xy1.T).T


xy = xy0
axes = plt.gca()
tfs = np.linspace(0.0, 10.0, 100)
tf_prev = tfs[0]
for tf in tfs[1:]:
    xy = forward(fun, [tf_prev, tf], xy.T).T
    tf_prev = tf
    xy = resample(xy.T).T
    x, y = xy
    axes.plot(x, y, color="C0", alpha= (tf - tfs[0]) / (tfs[-1] - tfs[0]))
plt.axis("equal")
plt.show()

