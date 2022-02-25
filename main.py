# Python Standard Library
import time

# Third-Party Libraries
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as ani


# Parameters
# ------------------------------------------------------------------------------
def fun(t, xy):
    x, y = xy
    r = np.sqrt(x * x + y * y)
    dx = x + x * y - (x + y) * r
    dy = y - x * x + (x - y) * r
    return [dx, dy]


t_span = (0.0, 10.0)
y0 = [1.0, 0.0]

df = 60.0
dt = 1.0 / df
t = np.arange(t_span[0], t_span[1], dt)
t = np.r_[t, t_span[1]]

radius = 0.5
n = 10000
xc, yc = y0
y0s = np.array(
    [
        [xc + radius * np.cos(theta), yc + radius * np.sin(theta)]
        for theta in np.linspace(0, 2 * np.pi, n)
    ]
)


def solve(**kwargs):
    kwargs = kwargs.copy()
    kwargs["dense_outputs"] = True
    results = []
    for y0 in y0s:
        kwargs["y0"] = y0
        result = sci.solve_ivp(**kwargs)
        results.append(result)
    return results

def get_data(results, t):
    n = len(results)
    data = np.zeros((len(t), 2, n))
    for i, r in enumerate(results):
        sol_t = r.sol(t)
        data[:, :, i] = sol_t.T
        return data

def generate_movie(data, filename="movie.mp4"):
    fig = plt.figure(figsize=(16, 9))
    axes = fig.subplots()
    axes.axis("equal")
    ratio = 16 / 9
    ym = 1.2
    xm = ym * ratio
    print([-xm, xm, -ym, ym])
    axes.axis([-xm, xm, -ym, ym])
    fig.subplots_adjust(0, 0, 1, 1)
    # axes.axis('off')

    polygon = None

    def update(i):
        nonlocal polygon
        x, y = data[i]
        if polygon:
            polygon.remove()
        polygon = plt.fill(x, y, color="k")[0]

    writer = ani.FFMpegWriter(fps=df)
    animation = ani.FuncAnimation(fig, func=update, frames=len(data))
    animation.save("filename", writer=writer, dpi=100)

results = solve(
    fun=fun, 
    t_span=t_span, 
    y0s=y0s, 
    rtol=1e-9, 
    atol=0,
)
data = get_data(results, t)
generate_movie(data)