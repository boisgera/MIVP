# Third-Party Libraries
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tqdm

# TODO: branding: something using "ink" (I see ink that spreads). 
#       inkblot is still available in Pypi. Inkflow too, etc.

# TODO: measure the error from last to first too (boundary is a closed path)
#       OR, ask for a path that repeats the initial point. OR deal with an
#       explicit topology concept (nodes + edges). Would allow shallow as
#       well as filled topologies, holes, etc. I like the idea, since it
#       does allow for a clean API for multiple disconnected initial blobs.
#       BUT we lose the principle of curve parametrized by t for example.
#
#       WARNING: the initial "shape data" has to be GENERATIVE of graphs
#       (if graphs are to be adopted). We should be able to spot a fault edge
#       and to introduce an extra node in the middle (with coords).
#       But this process should introduced extra nodes with very controlled
#       coords (think: valid on the circle), so the generative concept should
#       be well thought.
#
#       Nota: "faulty edge" is not the proper concept in we are not propagating
#       1D shapes. If we propagate a grid, it is "fauly quad", etc. This is
#       (graph-)topology dependent. Mmm complex ... Some simple cases are
#       ok though (lines, triangles, quads, etc.). Arf ... depending on the
#       topology, the new nodes should be able to connect differently with
#       different neighbors.
#
#       NOTA: this approach would also work in N-dim, not mereley in 2D, which
#       could make for some really cool demos.
#
#       Well, this is complex. Start with 1D only, then maybe 2D dims,
#       and simplest topologies (lines, rect patch), then see what can
#       be done afterwards.


# TODO: multiple boundaries management is not pretty, solve this issue.
#       maybe regress the API.

# TODO: tqdm & asymptotics (convergence speed in theory & practice ?)

# TODO: proper abstraction for the results (wrt spatial index and or time).
#       So that the user doesn't have the messy reordering of the raw data
#       todo?

# TODO: diagnoses (real-time?) with "heatmap"?

# TODO: better data / plot decoupling. Don't presume too much of the usage.

def solve(**kwargs):
    kwargs = kwargs.copy()

    boundary = kwargs["boundary"]
    del kwargs["boundary"]

    boundary_atol = kwargs.get("boundary_atol", 0.01)
    del kwargs["boundary_atol"]
    boundary_rtol = kwargs.get("boundary_rtol", 0.1)
    del kwargs["boundary_rtol"]
    t_eval = kwargs["t_eval"]
    kwargs["t_span"] = (t_eval[0], t_eval[-1])

    data = [np.zeros((2, len(t_eval)), dtype=np.float64) for _ in range(4)]

    s = list(np.linspace(0.0, 1.0, 4))
    y0s = boundary(np.array(s))
    for i, y0 in enumerate(y0s):
        kwargs["y0"] = y0
        result = sci.solve_ivp(**kwargs)
        data[i] = result.y

    while True:
        data_array = np.array(data)
        x, y = data_array[:, 0], data_array[:, 1]
        d = np.sqrt(x * x + y * y)[:, :-1]
        error = boundary_atol + boundary_rtol * d
        # compute max and index that corresponds ?
        dxdy = np.diff(data, axis=0)
        dx, dy = dxdy[:, 0], dxdy[:, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        if np.all(np.amax(dd) <= error):
            break
        index_flat = np.argmax(dd)
        i, j = divmod(index_flat, np.shape(dd)[1])
        assert np.amax(dd) == dd[i, j]  # may fail when nan/infs?
        # with vinograd, np.amax(dd) may be nan if we include the origin.
        # Investigate !
        print(f"{len(data)=} {(i, j)=}", f"{np.amax(dd)=}")

        s.insert(i + 1, 0.5 * (s[i] + s[i + 1]))
        y0 = boundary(np.array([s[i + 1]]))[0]
        kwargs["y0"] = y0
        result = sci.solve_ivp(**kwargs)
        data.insert(i + 1, result.y)

    reshaped_data = np.einsum("kji", data)
    # reshaped_data is a (time_index, dim_state_space, num_points)-shaped array
    return reshaped_data


def generate_movie(data, filename, fps, dpi=300, axes=None, hook=None):
    fig = None
    if axes:
        fig = axes.get_figure()
    if not fig:
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplots()
        axes.axis("equal")
        ratio = 16 / 9
        x_max = np.amax(data[:, 0, :])
        x_min = np.amin(data[:, 0, :])
        y_max = np.amax(data[:, 1, :])
        y_min = np.amin(data[:, 1, :])

        # Create a margin
        x_c, y_c = 0.5 * (x_max + x_min), 0.5 * (y_max + y_min)
        width, height = x_max - x_min, y_max - y_min
        x_min = x_min - 0.1 * width
        x_max = x_max + 0.1 * width
        y_min = y_min - 0.1 * width
        y_max = y_max + 0.1 * width
        width, height = x_max - x_min, y_max - y_min

        if width / height <= ratio:  # adjust width
            width = height * ratio
            x_min, x_max = x_c - 0.5 * width, x_c + 0.5 * width
        else:  # adjust height
            height = width / ratio
            y_min, y_max = y_c - 0.5 * height, y_c + 0.5 * height
        axes.axis([x_min, x_max, y_min, y_max])
        fig.subplots_adjust(0, 0, 1, 1)
        axes.axis("off")

    polygon = None

    def update(i):
        nonlocal polygon
        try:
            polygon.remove()
        except:
            pass

        # print(f"{d = }", f"{np.shape(d)  = }")
        # print(f"{i = }")
        x = data[i][0]
        y = data[i][1]
        polygon = axes.fill(x, y, color="k", zorder=1000)[0]

        if hook:
            hook(i, axes)

    writer = ani.FFMpegWriter(fps=fps)
    animation = ani.FuncAnimation(fig, func=update, frames=len(data))
    bar = tqdm.tqdm(total=len(data))
    animation.save(filename, writer=writer, dpi=dpi, progress_callback=lambda i, n: bar.update(1))
    bar.close()