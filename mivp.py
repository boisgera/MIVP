
# Python Standard Library
import multiprocessing as mp

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
#       Short-term : 1D (boundary of 2D), with start point = end point


# TODO: multiple boundaries management is not pretty, solve this issue.
#       maybe regress the API.
#       Yes; go back to simple boundary ATM.

# TODO: tqdm & asymptotics (convergence speed in theory & practice ?)

# TODO: proper abstraction for the results (wrt spatial index and or time).
#       So that the user doesn't have the messy reordering of the raw data
#       todo?

# TODO: diagnoses (real-time?) with "heatmap"? Or max curves ; also plot
#       (final) density distributions, etc. Read about von mises (circular)
#       kernel density estimation a bit (or first plot very rough density
#       approx, assuming some non-random spacing).

# TODO: try multi-processing for speed? (on a decoupled example first?)

# TODO: better data / plot decoupling. Don't presume too much of the usage.
#
# TODO: custom "plots" (/viz) : movie, movement with shadow of the past,
#       snapshot with shadow, "vignettes" (multiple snapshots), 3D graph
#       (time along z), etc.

# TODO: ATM we are t_eval-based. Shall we support dense output two and how?
#       We do not pretend that we have a "result" structure anymore now,
#       so I am not sure ...

# TODO: investigate proper form for boundaries that work as expected with scalar
#       and vector inputs (not as simple as applying a numpy.vectorize decorator)
#       ATM we require only a scalar form that works and manually vectorize the
#       stuff.

def solve(**kwargs):
    # Arguments Handling
    kwargs = kwargs.copy()

    # Get t_eval (mandatory), infer t_span if needed
    try:
        t_eval = kwargs["t_eval"]
    except KeyError:
        raise TypeError("t_eval argument is mandatory")
    kwargs.setdefault("t_span", (t_eval[0], t_eval[-1]))

    # Parameters specific to mivp: pick them & clean-up kwargs afterwards.
    boundary = kwargs["boundary"]
    del kwargs["boundary"]
    boundary_atol = kwargs.get("boundary_atol", 0.01)
    try:
        del kwargs["boundary_atol"]
    except KeyError:
        pass
    boundary_rtol = kwargs.get("boundary_rtol", 0.0)
    try:
        del kwargs["boundary_rtol"]
    except KeyError:
        pass
    # 🚧 TODO: enforce boundary_sampling >= 3 via a ValueError
    boundary_sampling = kwargs.get("boundary_sampling", 3)
    try:
        del kwargs["boundary_sampling"]
    except KeyError:
        pass
    upsampling_margin = kwargs.get("upsampling_margin", 0.10)  # i.e. 10%
    try:
        del kwargs["upsampling_margin"]
    except KeyError:
        pass

    # Compute the trajectories for the initial boundary sampling
    s = list(np.linspace(0.0, 1.0, boundary_sampling, endpoint=True))
    assert len(s) == boundary_sampling
    y0s = [boundary(s_) for s_ in s]

    data = []
    for i, y0 in enumerate(y0s):
        kwargs["y0"] = y0
        result = sci.solve_ivp(**kwargs)
        data.append(result.y)
    assert np.shape(data) == (len(s), 2, len(t_eval))

    while True:
        data_array = np.array(data)
        x, y = data_array[:, 0], data_array[:, 1]
        assert np.shape(x) == (len(s), len(t_eval))
        assert np.shape(y) == (len(s), len(t_eval))
        v = np.sqrt(x * x + y * y)
        d = 0.5 * (v[:-1] + v[1:])
        threshold = boundary_atol + boundary_rtol * d
        assert np.shape(threshold) == (len(s) - 1, len(t_eval))
        dxdy = np.diff(data, axis=0)
        assert np.shape(dxdy) == (len(s) - 1, 2, len(t_eval))
        dx, dy = dxdy[:, 0], dxdy[:, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        assert np.shape(dd) == (len(s) - 1, len(t_eval))
        r = 0.5 * (np.array(s[:-1]) + np.array(s[1:]))
        r_init = r.copy()

        if np.all(dd <= threshold):  # 👍
            break
        density_init = 1 / np.diff(s)
        density_increase_factor = np.amax(dd / threshold, axis=1)

        # 🥼 Linear (integer) upsampling
        #print(f"{density_increase_factor =}")
        density_increase_factor *= 1.0 + upsampling_margin
        density_increase_factor_rounded_up = np.ceil(density_increase_factor).astype(np.int64)
        density_increase_factor = np.maximum(density_increase_factor, 1)
        #print(f"{density_increase_factor_rounded_up =}")

        s_upsampled = []
        density_increase_factor_upsampled = []
        density_upsampled = []
        for i, s_ in enumerate(s[:-1]):
            s_next = s[i + 1]
            #print(f"{len(np.linspace(s_, s_next, density_increase_factor_rounded_up[i]))}")
            s_upsampled.extend(
                np.linspace(s_, s_next, density_increase_factor_rounded_up[i], endpoint=False)
            )
            z = density_increase_factor_rounded_up[i]
            density_increase_factor_upsampled.extend([z] * z)
            density_upsampled.extend([density_init[i]] * z)
        s_upsampled.append(1.0)
        s_upsampled = np.array(s_upsampled)
        density_upsampled = np.array(density_upsampled)
        density_increase_factor_upsampled = np.array(density_increase_factor_upsampled)
        assert len(s_upsampled) == len(density_upsampled) +1

        
        # 🚧 🧠 try/adapt for multi-processing & benchmark
        for i, s_ in enumerate(s_upsampled):
            if s_ not in s: # ⚠️ avoid useless recomputations
                kwargs["y0"] = boundary(s_)
                data.insert(i, sci.solve_ivp(**kwargs).y)

        s = s_upsampled

    reshaped_data = np.einsum("kji", data)
    assert np.shape(reshaped_data) == (len(t_eval), 2, len(s))
    # reshaped_data is a (time_index, dim_state_space, num_points)-shaped array

    # 📈 Density Graph
    # --------------------------------------------------------------------------
    if False:
        _, ax1 = plt.subplots(nrows=1)#, sharex=True)
        ax1.set_xlim(0.0, 1.0)
        s = np.array(s)
        density = 1.0 / np.diff(s)
        #r = 0.5 * (s[:-1] + s[1:])
        ss = []
        for s_ in s:
            ss.extend([s_, s_])
        ss = ss[1:-1]
        dd = []
        for d in density:
            dd.extend([d, d])
        
        ax1.semilogy(ss, dd, color="orange", label="density")
        ax1.legend()
        plt.show()

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
        #polygon = axes.plot(x, y, "k.")

        if hook:
            hook(i, axes)

    writer = ani.FFMpegWriter(fps=fps)
    animation = ani.FuncAnimation(fig, func=update, frames=len(data))
    bar = tqdm.tqdm(total=len(data))
    animation.save(
        filename, writer=writer, dpi=dpi, progress_callback=lambda i, n: bar.update(1)
    )
    bar.close()
