from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from ..complex import SimplicialComplex


Shape = Literal["random", "circle", "two_circles", "sphere", "torus"]
PointCloud = NDArray[np.float64]


def generate_point_cloud(
    n_points: int,
    shape: Shape = "circle",
    noise: float = 0.03,
    seed: int | None = None,
) -> PointCloud:
    """Generate a point cloud in R^2 or R^3.

    Parameters
    ----------
    n_points:
        Number of points.
    shape:
        "random", "circle", "two_circles", "sphere", or "torus".
    noise:
        Standard deviation of Gaussian noise added to manifold samples.
    seed:
        Seed for reproducible generation.
    """
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)

    if shape == "random":
        # Uniform random cloud in the square [-1, 1]^2.
        return rng.uniform(-1.0, 1.0, size=(n_points, 2))

    if shape == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        points = np.column_stack((np.cos(theta), np.sin(theta)))

    elif shape == "two_circles":
        n_left = n_points // 2
        n_right = n_points - n_left

        theta_left = rng.uniform(0.0, 2.0 * np.pi, n_left)
        theta_right = rng.uniform(0.0, 2.0 * np.pi, n_right)

        left = np.column_stack((np.cos(theta_left) - 1.1, np.sin(theta_left)))
        right = np.column_stack((np.cos(theta_right) + 1.1, np.sin(theta_right)))
        points = np.vstack((left, right))

    elif shape == "sphere":
        # Normalizing Gaussian vectors gives uniform directions on S^2.
        points = rng.normal(size=(n_points, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    elif shape == "torus":
        major_radius = 0.7
        minor_radius = 0.7
        u = rng.uniform(0.0, 2.0 * np.pi, n_points)
        v = rng.uniform(0.0, 2.0 * np.pi, n_points)

        points = np.column_stack(
            (
                (major_radius + minor_radius * np.cos(v)) * np.cos(u),
                (major_radius + minor_radius * np.cos(v)) * np.sin(u),
                minor_radius * np.sin(v),
            )
        )

    else:
        raise ValueError(f"Unknown shape: {shape!r}")

    if noise > 0:
        points = points + rng.normal(scale=noise, size=points.shape)

    return points.astype(float)


def pairwise_distances(points: PointCloud) -> PointCloud:
    """Return the Euclidean pairwise-distance matrix."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n_points, dimension)")

    differences = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(differences, axis=2)


def choose_epsilon(
    points: PointCloud,
    k: int = 3,
    factor: float = 1.15,
) -> float:
    """Choose a heuristic Rips scale from k-nearest-neighbour distances.

    For mathematical experiments, it is usually better to specify epsilon
    manually. This function is convenient when only n_points is supplied.
    """
    points = np.asarray(points, dtype=float)
    n_points = len(points)

    if n_points == 0:
        raise ValueError("The point cloud is empty")
    if n_points == 1:
        return 0.0
    if k < 1:
        raise ValueError("k must be at least 1")
    if factor <= 0:
        raise ValueError("factor must be positive")

    k = min(k, n_points - 1)
    distances = pairwise_distances(points)
    np.fill_diagonal(distances, np.inf)

    kth_distances = np.partition(distances, k - 1, axis=1)[:, k - 1]
    return float(factor * np.median(kth_distances))


def rips_complex(
    points: PointCloud,
    epsilon: float,
    *,
    max_dimension: int = 2,
    vertex_start: int = 1,
) -> SimplicialComplex:
    """Construct a Vietoris–Rips complex by filling cliques up to max_dimension."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite array of shape (n_points, dimension).")
    if not np.isfinite(epsilon) or epsilon < 0 or max_dimension < 0:
        raise ValueError("Use a finite nonnegative radius and a nonnegative dimension.")
    adjacent = pairwise_distances(points) <= epsilon + 1e-12
    n_points = len(points)
    simplices = [(vertex,) for vertex in range(n_points)]
    for size in range(2, min(max_dimension + 1, n_points) + 1):
        simplices.extend(
            simplex for simplex in combinations(range(n_points), size)
            if all(adjacent[i, j] for i, j in combinations(simplex, 2))
        )
    return SimplicialComplex(tuple(vertex + vertex_start for vertex in simplex) for simplex in simplices)


def generate_rips_data(
    n_points: int,
    shape: Shape = "circle",
    epsilon: float | None = None,
    max_dimension: int = 2,
    noise: float = 0.03,
    seed: int | None = None,
    vertex_start: int = 1,
) -> tuple[PointCloud, SimplicialComplex, float]:
    """Generate points and return ``(points, complex_rips, epsilon)``.

    This version keeps the coordinates so that the point cloud, Rips complex,
    and a computed categorical cover can later be visualized together.
    """
    points = generate_point_cloud(
        n_points=n_points,
        shape=shape,
        noise=noise,
        seed=seed,
    )
    if epsilon is None:
        epsilon = choose_epsilon(points)

    complex_rips = rips_complex(
        points=points,
        epsilon=epsilon,
        max_dimension=max_dimension,
        vertex_start=vertex_start,
    )

    return points, complex_rips, float(epsilon)


