import time
from numpy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation


# def fun(t, xy):
#     x, y = xy
#     return array([-y - 0.5 * x, +x - 0.1 * y])

def fun(t, xy):
    x, y = xy
    r = sqrt(x*x + y*y)
    dx = x + x * y - (x + y) * r
    dy = y - x * x + (x - y) * r
    return [dx, dy]

t_span = (0.0, 10.0)
y0 = [1.0, 0.0]

df = 60.0
dt = 1.0 / df

r = solve_ivp(fun, t_span, y0, dense_output=True)
t = linspace(t_span[0], t_span[1], 1000)
td = arange(t_span[0], t_span[1], dt)
x_t, y_t = r.sol(t)
x_td, y_td = r.sol(td)

if False:
    plt.figure()
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.axis("equal")
    plt.plot(x_t, y_t, "--", color="grey")
    plt.plot(x_td, y_td, "k.")
    plt.grid(True)
    plt.show()

if True:

    radius = 0.5
    n = 10000
    xc, yc = y0
    y0s = array([
        [xc + radius * cos(theta), yc + radius * sin(theta)]
        for theta in linspace(0, 2 * pi, n)
    ])

    t_ = time.time()
    print(0.0)

    rs = []
    for y0 in y0s:
        r = solve_ivp(fun, t_span, y0, dense_output=True, rtol=1e-9, atol=1e-100)
        rs.append(r)

    print(time.time() - t_)

    if False:
        plt.figure()
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.axis("equal")
        plt.plot(x_t, y_t, "--", color="grey")

    data = zeros((len(td), 2, n))
    for i, r in enumerate(rs):
        sol_td = r.sol(td)
        data[:, :, i] = sol_td.T

    # for x, y in data:
    #     plt.fill(x, y, color="k", alpha=0.1)
    # plt.grid(True)
    # print(time.time() - t_)
    # plt.show()

    # x_td = []
    # y_td = []
    # for td_ in td:
    #     x_td.append([])
    #     y_td.append([])
    #     for r in rs:
    #         x_td[-1].append(r.sol(td_)[0])
    #         y_td[-1].append(r.sol(td_)[1])

    # for x, y in zip(x_td, y_td):
    #     plt.fill(x, y, color="k")
    # plt.grid(True)
    # print(time.time() - t_)
    # plt.show()

fig = plt.figure(figsize=(16, 9))
axes = fig.subplots()
axes.axis("equal")
ratio = 16/9
ym = 1.2
xm = ym * ratio
print([-xm, xm, -ym, ym])
axes.axis([-xm, xm, -ym, ym])
print(axes.axis())
fig.subplots_adjust(0, 0, 1, 1)
#axes.axis('off')

polygon = None

def func(i):
    global polygon
    x, y = data[i]
    if polygon:
        polygon.remove()
    polygon = plt.fill(x, y, color="k")[0]

writer = matplotlib.animation.FFMpegWriter(fps=df)
anim = matplotlib.animation.FuncAnimation(fig, func, len(td))
anim.save("movie.mp4", writer=writer, dpi=100)

#plt.show()