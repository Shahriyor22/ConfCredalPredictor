import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

def dist_to_xy(dist):
    """Maps a 3D probability distribution to 2D simplex coordinates."""
    dist = np.asarray(dist)

    return (dist @ corners).ravel()

def plot_simplex(pred, true_dist, noisy_dist, weights, class_labels = None,
                 cred_set_kl = None, cred_set_chi2 = None, cred_set_so = None):
    def annotate_corners(ax, class_labels, offset = 0.05):
        """Writes class names at the corners."""
        if not class_labels or len(class_labels) != 3:
            return

        t = class_labels[0]  # entailment
        l = class_labels[1]  # neutral
        r = class_labels[2]  # contradiction

        def place(text, corner_idx, angle):
            p = corners[corner_idx]
            v = p - corners.mean(axis = 0)
            u = v / (np.linalg.norm(v) + 1e-9)
            pos = p + offset * u

            ax.text(pos[0], pos[1], text, ha = "center", va = "center", rotation = angle)

        place(t, 2, 0)
        place(l, 0, -60)
        place(r, 1, 60)

    def draw_fo(ax, cred_set, title = None, leg_show = False):
        """Visualizes a first-order credal set on a simplex."""
        xs = [corners[0, 0], corners[1, 0], corners[2, 0], corners[0, 0]]
        ys = [corners[0, 1], corners[1, 1], corners[2, 1], corners[0, 1]]
        ax.plot(xs, ys, 'k-')
        annotate_corners(ax, class_labels)

        # credal set
        if cred_set is not None and len(cred_set) > 0:
            xy = np.array([dist_to_xy(dist) for dist in cred_set])
            ax.scatter(xy[:, 0], xy[:, 1], c = "green", alpha = 0.2, s = 5, label = "Credal set")

        # prediction, true and noisy distributions
        px, py = dist_to_xy(pred)
        tx, ty = dist_to_xy(true_dist)
        nx, ny = dist_to_xy(noisy_dist)

        ax.scatter(px, py, c = "black", marker = "o", label = "Prediction")
        ax.scatter(tx, ty, c = "orange", marker = "s", label = "True distribution")
        ax.scatter(nx, ny, c = "red", marker = "s", label = "Noisy distribution")

        if leg_show: ax.legend(loc = "upper left")
        ax.axis("equal")
        ax.axis("off")
        if title: ax.set_title(title)

    def draw_so(ax, cred_set, weights):
        """Visualizes a first-order credal set and a second-order contour on a simplex."""
        xs = [corners[0, 0], corners[1, 0], corners[2, 0], corners[0, 0]]
        ys = [corners[0, 1], corners[1, 1], corners[2, 1], corners[0, 1]]
        ax.plot(xs, ys, 'k-')
        annotate_corners(ax, class_labels)

        # credal set
        xy = np.array([dist_to_xy(dist) for dist in cred_set])
        x, y = xy[:, 0], xy[:, 1]

        w = np.asarray(weights, float).ravel()
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)

        ax.tricontourf(mtri.Triangulation(x, y), w, cmap = "Greens")

        # true and noisy distributions
        tx, ty = dist_to_xy(true_dist)
        nx, ny = dist_to_xy(noisy_dist)

        ax.scatter(tx, ty, c = "orange", marker = "s", label = "True distribution")
        ax.scatter(nx, ny, c = "red", marker = "s", label = "Noisy distribution")

        ax.axis('equal')
        ax.axis('off')
        ax.set_title("SO")

    fig, axes = plt.subplots(1, 3, figsize = (12, 4))

    draw_fo(axes[0], cred_set_kl, title = "KL", leg_show = False)
    draw_fo(axes[1], cred_set_chi2, title = "CHI2")
    draw_so(axes[2], cred_set_so, weights)

    fig.tight_layout()
    plt.show()
