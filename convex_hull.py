from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def circ(r):
    x = r * np.cos(np.arange(0, 2 * np.pi, 0.001))
    y = r * np.sin(np.arange(0, 2 * np.pi, 0.001))

    return np.vstack([x, y]).T


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def base_2d(num_vertex, radius):
    if num_vertex < 2:
        raise ValueError("Two few vertices!")
    if radius < 0.2:
        raise ValueError("Radius too small")

    sample_base = circ(radius)
    plt.plot(sample_base[:, 0], sample_base[:, 1])
    # print(sample_base)

    base_points = []
    ind_ref = []

    while len(base_points) < num_vertex:
        ind = np.random.randint(0, len(sample_base))
        new_point = sample_base[ind]
        for pt in base_points:
            if euclidean_dist(pt, new_point) < radius / 10:
                break
        base_points.append(new_point)
        ind_ref.append(ind)

    sort_help = list(zip(base_points, ind_ref))
    sort_help.sort(key=lambda x: x[1])
    base_points = list(x[0] for x in sort_help)

    # visual
    base_points.append(base_points[0])

    return np.array(base_points)


def hull(num_vertex, radius, height, shift=False):

    base_polygon = base_2d(num_vertex, radius)
    base = np.column_stack([base_polygon, np.zeros(len(base_polygon))])
    top = np.column_stack([base_polygon, height * np.ones(len(base_polygon))])

    sides = []
    prev = None
    for top_pt, bot_pt in zip(top, base):
        if not prev:
            prev = (top_pt, bot_pt)
            continue
        curr_face = np.array([prev[0], prev[1], top_pt, bot_pt])
        sides.append(curr_face)
        prev = (top_pt, bot_pt)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(
        base[:, 0],
        base[:, 1],
        base[:, 2],
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.plot_trisurf(
        top[:, 0],
        top[:, 1],
        top[:, 2],
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    for side in sides:
        print(side)
        ax.plot(
            side[:, 0],
            side[:, 1],
            side[:, 2],
            linewidth=1,
            antialiased=False,
        )
    plt.show()


hull(5, 5, 10)
# plt.show()
