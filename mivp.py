# Third-Party Libraries
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def solve(**kwargs):
    kwargs = kwargs.copy()
    kwargs["dense_output"] = True
    y0s = kwargs["y0s"]
    del kwargs["y0s"]
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


def generate_movie(data, filename, fps, axes=None):
    fig = None
    if axes:
        fig = axes.get_figure()
    if not fig:
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplots()
        axes.axis("equal")
        ratio = 16 / 9
        x_max = np.amax(data[:, 0, :]) * 1.2
        x_min = np.amin(data[:, 0, :]) * 1.2
        y_max = np.amax(data[:, 1, :]) * 1.2
        y_min = np.amin(data[:, 1, :]) * 1.2
        x_c, y_c = 0.5 * (x_max + x_min), 0.5 * (y_max + y_min)
        width, height = x_max - x_min, y_max - y_min
        if width / height <= ratio: # adjust width
            width = height * ratio
            x_min, x_max = x_c - 0.5 * width, x_c + 0.5 * width
        else: # adjust height
            height = width / ratio
            y_min, y_max = y_c - 0.5 * height, y_c + 0.5 * height
        axes.axis([x_min, x_max, y_min, y_max])
        fig.subplots_adjust(0, 0, 1, 1)
        axes.axis("off")

    polygon = None

    def update(i):
        nonlocal polygon
        x, y = data[i]
        if polygon:
            polygon.remove()
        polygon = axes.fill(x, y, color="k")[0]

    writer = ani.FFMpegWriter(fps=fps)
    animation = ani.FuncAnimation(fig, func=update, frames=len(data))
    animation.save(filename, writer=writer, dpi=100)
