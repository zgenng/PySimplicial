import math
from collections import defaultdict

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


DEFAULT_PALETTE = (
    "#E63946",  # red
    "#2563EB",  # blue
    "#059669",  # green
    "#F59E0B",  # amber
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # magenta
    "#92400E",  # brown
    "#65A30D",  # lime
    "#4F46E5",  # indigo
)

DEFAULT_THEME = {
    "background": "#FFFFFF",
    "panel_background": "#FFFFFF",
    "base_edge": "#94A3B8",
    "base_triangle": "#CBD5E1",
    "base_triangle_edge": "#94A3B8",
    "node_face": "#FFFFFF",
    "node_edge": "#0F172A",
    "label": "#0F172A",
    "title": "#0F172A",
    "subtitle": "#475569",
    "outline": "#FFFFFF",
}


def _vertex_sort_key(vertex):
    cls = type(vertex)
    return cls.__module__, cls.__qualname__, repr(vertex)


def _simplex_sort_key(simplex):
    return len(simplex), tuple(
        sorted((_vertex_sort_key(vertex) for vertex in simplex))
    )


def _edge_key(edge):

    edge = frozenset(edge)
    if len(edge) != 2:
        raise ValueError(f"Expected an edge with two endpoints, received {edge}.")
    return edge


def _ordered_edge(edge):
    return tuple(sorted(edge, key=_vertex_sort_key))


def _build_graph(K):
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    edges = sorted(
        (
            _edge_key(simplex)
            for simplex in K.simplicial_complex
            if len(simplex) == 2
        ),
        key=lambda edge: tuple(
            _vertex_sort_key(vertex) for vertex in _ordered_edge(edge)
        ),
    )
    adjacency = {vertex: set() for vertex in vertices}
    for edge in edges:
        u, v = tuple(edge)
        adjacency[u].add(v)
        adjacency[v].add(u)
    return vertices, edges, adjacency


def _circular_positions(vertices):
    count = len(vertices)
    if count == 1:
        return {vertices[0]: np.array([0.0, 0.0])}
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    # Starting at the top gives labelled examples a more natural orientation.
    angles += np.pi / 2.0
    return {
        vertex: np.array([math.cos(angle), math.sin(angle)])
        for vertex, angle in zip(vertices, angles)
    }


def _spring_positions(vertices, edges, seed, iterations=350):
    count = len(vertices)
    if count <= 2:
        return _circular_positions(vertices)

    index = {vertex: i for i, vertex in enumerate(vertices)}
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    angles += np.pi / 2.0
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    points += rng.normal(scale=0.035, size=points.shape)

    edge_indices = [
        (index[u], index[v])
        for u, v in (_ordered_edge(edge) for edge in edges)
    ]
    ideal = 1.55 / math.sqrt(count)
    temperature = 0.22

    for step in range(iterations):
        delta = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, 1.0)
        distance = np.maximum(distance, 1e-4)

        # Repulsion between every pair of vertices.
        displacement = (
            delta * (ideal * ideal / (distance * distance))[:, :, None]
        ).sum(axis=1)

        # Attraction along graph edges.
        for i, j in edge_indices:
            vector = points[i] - points[j]
            length = max(float(np.linalg.norm(vector)), 1e-4)
            force = vector * (length / ideal)
            displacement[i] -= force
            displacement[j] += force

        norms = np.linalg.norm(displacement, axis=1)
        norms = np.maximum(norms, 1e-9)
        points += displacement / norms[:, None] * np.minimum(norms, temperature)[:, None]
        points -= points.mean(axis=0)
        temperature = 0.22 * (1.0 - (step + 1) / iterations) ** 1.4

    return {
        vertex: point
        for vertex, point in zip(vertices, points)
    }


def _bipartite_parts(vertices, adjacency):
    colors = {}
    components = []

    for start in sorted(vertices, key=_vertex_sort_key):
        if start in colors:
            continue

        colors[start] = 0
        stack = [start]
        component = []

        while stack:
            vertex = stack.pop()
            component.append(vertex)

            for neighbour in sorted(adjacency[vertex], key=_vertex_sort_key):
                if neighbour not in colors:
                    colors[neighbour] = 1 - colors[vertex]
                    stack.append(neighbour)
                elif colors[neighbour] == colors[vertex]:
                    u, v = sorted((vertex, neighbour), key=_vertex_sort_key)
                    raise ValueError(
                        "layout='partite' requires a bipartite graph; "
                        f"the edge ({u!r}, {v!r}) joins vertices in the same part."
                    )

        first = sorted(
            (vertex for vertex in component if colors[vertex] == 0),
            key=_vertex_sort_key,
        )
        second = sorted(
            (vertex for vertex in component if colors[vertex] == 1),
            key=_vertex_sort_key,
        )
        components.append((first, second))

    left = []
    right = []
    for first, second in components:
        keep_imbalance = abs(
            (len(left) + len(first)) - (len(right) + len(second))
        )
        swap_imbalance = abs(
            (len(left) + len(second)) - (len(right) + len(first))
        )
        if swap_imbalance < keep_imbalance:
            first, second = second, first
        left.extend(first)
        right.extend(second)

    return (
        sorted(left, key=_vertex_sort_key),
        sorted(right, key=_vertex_sort_key),
    )


def _partite_positions(vertices, adjacency):
    left, right = _bipartite_parts(vertices, adjacency)
    largest_part = max(len(left), len(right), 1)
    vertical_span = max(1.6, 0.34 * (largest_part - 1))

    def side_positions(part, x_coordinate):
        if not part:
            return {}
        if len(part) == 1:
            y_coordinates = np.array([0.0])
        else:
            y_coordinates = np.linspace(
                vertical_span / 2.0,
                -vertical_span / 2.0,
                len(part),
            )
        return {
            vertex: np.array([x_coordinate, y], dtype=float)
            for vertex, y in zip(part, y_coordinates)
        }

    positions = side_positions(left, -1.0)
    positions.update(side_positions(right, 1.0))
    return positions


def _connected_vertex_components(vertices, adjacency):
    unseen = set(vertices)
    components = []

    while unseen:
        start = min(unseen, key=_vertex_sort_key)
        unseen.remove(start)
        component = {start}
        stack = [start]

        while stack:
            vertex = stack.pop()
            for neighbour in sorted(adjacency[vertex], key=_vertex_sort_key):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)

        components.append(component)

    return components


def _biconnected_edge_blocks(vertices, adjacency):
    discovery = {}
    low = {}
    parent = {}
    edge_stack = []
    blocks = []
    clock = 0

    def visit(vertex):
        nonlocal clock
        clock += 1
        discovery[vertex] = clock
        low[vertex] = clock

        for neighbour in sorted(adjacency[vertex], key=_vertex_sort_key):
            edge = _edge_key((vertex, neighbour))

            if neighbour not in discovery:
                parent[neighbour] = vertex
                edge_stack.append(edge)
                visit(neighbour)
                low[vertex] = min(low[vertex], low[neighbour])

                if low[neighbour] >= discovery[vertex]:
                    block = set()
                    while edge_stack:
                        stacked_edge = edge_stack.pop()
                        block.add(stacked_edge)
                        if stacked_edge == edge:
                            break
                    if block:
                        blocks.append(frozenset(block))

            elif (
                parent.get(vertex) != neighbour
                and discovery[neighbour] < discovery[vertex]
            ):
                edge_stack.append(edge)
                low[vertex] = min(low[vertex], discovery[neighbour])

    for start in sorted(vertices, key=_vertex_sort_key):
        if start in discovery:
            continue
        visit(start)
        if edge_stack:
            blocks.append(frozenset(edge_stack))
            edge_stack.clear()

    return blocks


def _block_vertices(block):
    return {vertex for edge in block for vertex in edge}


def _is_cycle_block(block):
    vertices = _block_vertices(block)
    if len(vertices) < 3 or len(block) != len(vertices):
        return False

    degree = {vertex: 0 for vertex in vertices}
    for edge in block:
        u, v = tuple(edge)
        degree[u] += 1
        degree[v] += 1
    return all(value == 2 for value in degree.values())


def _cycle_order(block):
    if not _is_cycle_block(block):
        raise ValueError("The supplied block is not a simple cycle.")

    vertices = _block_vertices(block)
    adjacency = {vertex: set() for vertex in vertices}
    for edge in block:
        u, v = tuple(edge)
        adjacency[u].add(v)
        adjacency[v].add(u)

    start = min(vertices, key=_vertex_sort_key)
    first = min(adjacency[start], key=_vertex_sort_key)
    order = [start]
    previous = start
    current = first

    while current != start:
        if current in order:
            raise RuntimeError("Cycle traversal encountered a repeated vertex.")
        order.append(current)
        candidates = adjacency[current] - {previous}
        if not candidates:
            raise RuntimeError("Cycle traversal reached an open path.")
        next_vertex = min(candidates, key=_vertex_sort_key)
        previous, current = current, next_vertex

    if len(order) != len(vertices):
        raise RuntimeError("Cycle traversal did not visit every block vertex.")
    return order


def _rotate_vector(vector, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y = vector
    return np.array(
        [cosine * x - sine * y, sine * x + cosine * y],
        dtype=float,
    )


def _unit_vector(vector, fallback=(1.0, 0.0)):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        vector = np.asarray(fallback, dtype=float)
        length = max(float(np.linalg.norm(vector)), 1e-9)
    return vector / length


def _median_edge_length(pos, edges):
    lengths = [
        float(np.linalg.norm(pos[u] - pos[v]))
        for u, v in (_ordered_edge(edge) for edge in edges)
        if float(np.linalg.norm(pos[u] - pos[v])) > 1e-9
    ]
    return float(np.median(lengths)) if lengths else 1.0


def _regular_cycle_radius(vertex_count, side_length):
    return side_length / (2.0 * math.sin(math.pi / vertex_count))


def _fit_regular_cycle(order, initial, side_length):
    points = np.asarray([initial[vertex] for vertex in order], dtype=float)
    center = points.mean(axis=0)
    radius = _regular_cycle_radius(len(order), side_length)
    best_positions = None
    best_error = math.inf

    for direction in (1.0, -1.0):
        base_angles = direction * np.linspace(
            0.0,
            2.0 * np.pi,
            len(order),
            endpoint=False,
        )
        template = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        centered = points - center
        correlation = np.sum(
            (centered[:, 0] + 1j * centered[:, 1])
            * np.conjugate(template[:, 0] + 1j * template[:, 1])
        )
        rotation = float(np.angle(correlation)) if abs(correlation) > 1e-12 else 0.0
        angles = base_angles + rotation
        candidate = center + radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        error = float(np.sum((candidate - points) ** 2))
        if error < best_error:
            best_error = error
            best_positions = candidate

    return {
        vertex: point
        for vertex, point in zip(order, best_positions)
    }


def _anchored_regular_cycle(
    block,
    anchor,
    anchor_position,
    outward,
    side_length,
    initial,
    occupied_positions,
):
    order = _cycle_order(block)
    anchor_index = order.index(anchor)
    order = order[anchor_index:] + order[:anchor_index]
    radius = _regular_cycle_radius(len(order), side_length)
    outward = _unit_vector(outward)
    center = np.asarray(anchor_position, dtype=float) + outward * radius
    anchor_angle = math.atan2(-outward[1], -outward[0])

    candidates = []
    for direction in (1.0, -1.0):
        angles = anchor_angle + direction * np.linspace(
            0.0,
            2.0 * np.pi,
            len(order),
            endpoint=False,
        )
        points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        points[0] = anchor_position
        candidate = {
            vertex: point
            for vertex, point in zip(order, points)
        }

        movable = [vertex for vertex in order if vertex != anchor]
        if occupied_positions and movable:
            clearance = min(
                float(np.linalg.norm(candidate[vertex] - occupied))
                for vertex in movable
                for occupied in occupied_positions
            )
        else:
            clearance = math.inf

        initial_vectors = np.asarray(
            [initial[vertex] - initial[anchor] for vertex in movable],
            dtype=float,
        )
        candidate_vectors = np.asarray(
            [candidate[vertex] - candidate[anchor] for vertex in movable],
            dtype=float,
        )
        if movable:
            initial_scale = max(float(np.linalg.norm(initial_vectors)), 1e-9)
            candidate_scale = max(float(np.linalg.norm(candidate_vectors)), 1e-9)
            shape_error = float(np.sum(
                (
                    initial_vectors / initial_scale
                    - candidate_vectors / candidate_scale
                ) ** 2
            ))
        else:
            shape_error = 0.0

        candidates.append((clearance, -shape_error, candidate))

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _transform_anchored_block(
    block,
    anchor,
    anchor_position,
    outward,
    side_length,
    initial,
):
    vertices = _block_vertices(block)
    if len(block) == 1:
        other = next(vertex for vertex in vertices if vertex != anchor)
        return {
            anchor: np.asarray(anchor_position, dtype=float),
            other: np.asarray(anchor_position, dtype=float)
            + _unit_vector(outward) * side_length,
        }

    local = {
        vertex: np.asarray(initial[vertex] - initial[anchor], dtype=float)
        for vertex in vertices
    }
    centroid = np.mean([local[vertex] for vertex in vertices], axis=0)
    source_direction = _unit_vector(centroid)
    target_direction = _unit_vector(outward)
    rotation = math.atan2(target_direction[1], target_direction[0]) - math.atan2(
        source_direction[1],
        source_direction[0],
    )
    current_length = _median_edge_length(initial, block)
    scale = side_length / max(current_length, 1e-9)

    return {
        vertex: np.asarray(anchor_position, dtype=float)
        + scale * _rotate_vector(local[vertex], rotation)
        for vertex in vertices
    }


def _pack_connected_components(pos, components, gap):
    if len(components) <= 1:
        return pos

    components = sorted(
        components,
        key=lambda component: min(
            (_vertex_sort_key(vertex) for vertex in component),
            default=("", "", ""),
        ),
    )
    boxes = []
    for component in components:
        points = np.asarray([pos[vertex] for vertex in component], dtype=float)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        boxes.append((minimum, maximum))

    cell_width = max(float(maximum[0] - minimum[0]) for minimum, maximum in boxes)
    cell_height = max(float(maximum[1] - minimum[1]) for minimum, maximum in boxes)
    cell_width = max(cell_width, gap) + 2.2 * gap
    cell_height = max(cell_height, gap) + 2.2 * gap
    columns = math.ceil(math.sqrt(len(components)))

    packed = dict(pos)
    for index, (component, (minimum, maximum)) in enumerate(zip(components, boxes)):
        row, column = divmod(index, columns)
        target_center = np.array(
            [column * cell_width, -row * cell_height],
            dtype=float,
        )
        shift = target_center - (minimum + maximum) / 2.0
        for vertex in component:
            packed[vertex] = np.asarray(pos[vertex], dtype=float) + shift

    return packed


def _orient_graph_horizontally(pos, components, gap):
    if not pos:
        return {}

    components = sorted(
        components,
        key=lambda component: min(
            (_vertex_sort_key(vertex) for vertex in component),
            default=("", "", ""),
        ),
    )
    oriented_components = []

    for component in components:
        ordered = sorted(component, key=_vertex_sort_key)
        points = np.asarray([pos[vertex] for vertex in ordered], dtype=float)
        center = points.mean(axis=0)

        if len(ordered) <= 1:
            rotated = points - center
        else:
            best_pair = (0, 1)
            best_distance_squared = -1.0

            for first in range(len(ordered)):
                for second in range(first + 1, len(ordered)):
                    vector = points[second] - points[first]
                    distance_squared = float(np.dot(vector, vector))
                    if distance_squared > best_distance_squared + 1e-12:
                        best_distance_squared = distance_squared
                        best_pair = (first, second)

            first, second = best_pair
            horizontal = _unit_vector(points[second] - points[first])
            vertical = np.array([-horizontal[1], horizontal[0]], dtype=float)
            centered = points - center
            rotated = np.column_stack(
                (centered @ horizontal, centered @ vertical)
            )

        oriented_components.append((ordered, rotated))

    packed = {}
    cursor = 0.0
    separation = max(float(gap), 1e-6) * 2.2

    for index, (ordered, points) in enumerate(oriented_components):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        width = float(maximum[0] - minimum[0])
        vertical_center = float((minimum[1] + maximum[1]) / 2.0)

        if index == 0:
            left = 0.0
        else:
            left = cursor + separation

        shift = np.array([left - minimum[0], -vertical_center], dtype=float)
        for vertex, point in zip(ordered, points):
            packed[vertex] = np.asarray(point, dtype=float) + shift
        cursor = left + width

    all_points = np.asarray(list(packed.values()), dtype=float)
    global_center = all_points.mean(axis=0)
    return {
        vertex: point - global_center
        for vertex, point in packed.items()
    }


def _polygonal_graph_positions(vertices, edges, adjacency, seed):
    initial = _spring_positions(vertices, edges, seed=seed)
    blocks = _biconnected_edge_blocks(vertices, adjacency)
    cycle_blocks = {index for index, block in enumerate(blocks) if _is_cycle_block(block)}
    if not cycle_blocks:
        return initial

    side_length = _median_edge_length(initial, edges)
    positions = {
        vertex: np.asarray(point, dtype=float).copy()
        for vertex, point in initial.items()
    }
    block_vertices = [_block_vertices(block) for block in blocks]
    memberships = defaultdict(list)
    for block_index, vertices_in_block in enumerate(block_vertices):
        for vertex in vertices_in_block:
            memberships[vertex].append(block_index)

    visited_blocks = set()
    fixed_vertices = set()

    def block_center(block_index):
        return np.mean(
            [positions[vertex] for vertex in block_vertices[block_index]],
            axis=0,
        )

    def place_children(block_index, parent_articulation=None):
        current_center = block_center(block_index)
        articulation_vertices = sorted(
            (
                vertex
                for vertex in block_vertices[block_index]
                if len(memberships[vertex]) > 1
                and vertex != parent_articulation
            ),
            key=_vertex_sort_key,
        )

        for articulation in articulation_vertices:
            children = [
                child
                for child in memberships[articulation]
                if child not in visited_blocks
            ]
            children.sort(
                key=lambda child: (
                    child not in cycle_blocks,
                    -len(blocks[child]),
                    child,
                )
            )
            if not children:
                continue

            base_outward = positions[articulation] - current_center
            if float(np.linalg.norm(base_outward)) <= 1e-9:
                raw_center = np.mean(
                    [initial[vertex] for vertex in block_vertices[children[0]]],
                    axis=0,
                )
                base_outward = raw_center - initial[articulation]
            base_outward = _unit_vector(base_outward)

            if len(children) == 1:
                fan_angles = [0.0]
            else:
                fan_width = min(math.radians(150.0), math.radians(42.0 * len(children)))
                fan_angles = np.linspace(
                    -fan_width / 2.0,
                    fan_width / 2.0,
                    len(children),
                )

            for child, fan_angle in zip(children, fan_angles):
                outward = _rotate_vector(base_outward, float(fan_angle))
                occupied = [
                    positions[vertex]
                    for vertex in fixed_vertices
                    if vertex != articulation
                    and vertex not in block_vertices[child]
                ]

                if child in cycle_blocks:
                    child_positions = _anchored_regular_cycle(
                        blocks[child],
                        articulation,
                        positions[articulation],
                        outward,
                        side_length,
                        initial,
                        occupied,
                    )
                else:
                    child_positions = _transform_anchored_block(
                        blocks[child],
                        articulation,
                        positions[articulation],
                        outward,
                        side_length,
                        initial,
                    )

                positions.update(child_positions)
                fixed_vertices.update(block_vertices[child])
                visited_blocks.add(child)
                place_children(child, parent_articulation=articulation)

    while len(visited_blocks) < len(blocks):
        candidates = [index for index in range(len(blocks)) if index not in visited_blocks]
        root = max(
            candidates,
            key=lambda index: (
                index in cycle_blocks,
                len(blocks[index]),
                len(block_vertices[index]),
                -index,
            ),
        )

        if root in cycle_blocks:
            root_order = _cycle_order(blocks[root])
            positions.update(_fit_regular_cycle(root_order, initial, side_length))

        visited_blocks.add(root)
        fixed_vertices.update(block_vertices[root])
        place_children(root)

    components = _connected_vertex_components(vertices, adjacency)
    return _pack_connected_components(positions, components, side_length)


def _normalize_positions(pos):
    if not pos:
        return {}

    vertices = list(pos)
    points = np.asarray([pos[vertex] for vertex in vertices], dtype=float)
    points -= points.mean(axis=0)
    span = np.ptp(points, axis=0)
    scale = max(float(span.max()), 1.0)
    points = 2.0 * points / scale
    return {
        vertex: np.asarray(point, dtype=float)
        for vertex, point in zip(vertices, points)
    }


def _get_positions(K, seed=42, layout="auto"):
    vertices, edges, adjacency = _build_graph(K)
    node_count = len(vertices)
    is_graph = not any(
        len(simplex) >= 3
        for simplex in K.simplicial_complex
    )

    supported_layouts = {"auto", "spring", "polygonal", "circular", "partite"}
    if layout not in supported_layouts:
        expected = ", ".join(repr(name) for name in sorted(supported_layouts))
        raise ValueError(f"Unknown layout={layout!r}; expected one of {expected}.")

    if node_count == 0:
        return {}
    if node_count == 1:
        return {vertices[0]: np.array([0.0, 0.0])}
    if layout == "partite":
        positions = _partite_positions(vertices, adjacency)
    elif not edges:
        positions = _circular_positions(vertices)
    else:
        if layout == "auto" and is_graph:
            positions = _polygonal_graph_positions(
                vertices,
                edges,
                adjacency,
                seed=seed,
            )
        elif layout in ("auto", "spring"):
            positions = _spring_positions(vertices, edges, seed=seed)
        elif layout == "polygonal":
            positions = _polygonal_graph_positions(
                vertices,
                edges,
                adjacency,
                seed=seed,
            )
        else:
            positions = _circular_positions(vertices)

    if is_graph and layout != "partite":
        components = _connected_vertex_components(vertices, adjacency)
        gap = _median_edge_length(positions, edges) if edges else 1.0
        positions = _orient_graph_horizontally(
            positions,
            components,
            gap,
        )

    return _normalize_positions(positions)


def _resolve_positions(K, pos, seed, layout):
    if pos is None:
        return _get_positions(K, seed=seed, layout=layout)

    missing = set(K.vertices) - set(pos)
    if missing:
        raise ValueError(f"The supplied positions omit vertices: {missing}.")
    return {
        vertex: np.asarray(pos[vertex], dtype=float)
        for vertex in K.vertices
    }


def _offset_segment(p1, p2, offset):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return p1, p2

    normal_x = -dy / length
    normal_y = dx / length
    return (
        (x1 + offset * normal_x, y1 + offset * normal_y),
        (x2 + offset * normal_x, y2 + offset * normal_y),
    )


def _node_radius_points(node_size, gap=0.0):
    return max(math.sqrt(max(float(node_size), 0.0) / math.pi) + gap, 0.0)


def _draw_trimmed_edge(
    ax,
    p1,
    p2,
    color,
    *,
    width,
    alpha,
    zorder,
    node_size,
    node_gap=2.0,
    outline=None,
    outline_width=0.0,
):
    patch = FancyArrowPatch(
        tuple(np.asarray(p1, dtype=float)),
        tuple(np.asarray(p2, dtype=float)),
        arrowstyle="-",
        shrinkA=_node_radius_points(node_size, node_gap),
        shrinkB=_node_radius_points(node_size, node_gap),
        linewidth=width,
        color=color,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        mutation_scale=1.0,
        zorder=zorder,
    )
    if outline is not None and outline_width > 0.0:
        patch.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=width + outline_width,
                    foreground=outline,
                    alpha=0.96,
                ),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)
    return patch


def _draw_offset_edge(
    ax,
    pos,
    edge,
    color,
    *,
    offset=0.0,
    width=4.2,
    alpha=0.96,
    zorder=8,
    outline="#FFFFFF",
    node_size=600,
    node_gap=3.2,
):
    u, v = _ordered_edge(edge)
    p1, p2 = _offset_segment(pos[u], pos[v], offset)
    return _draw_trimmed_edge(
        ax,
        p1,
        p2,
        color=color,
        width=width,
        alpha=alpha,
        zorder=zorder,
        node_size=node_size,
        node_gap=node_gap,
        outline=outline,
        outline_width=2.8,
    )


def _ordered_triangle_points(simplex, pos, scale=1.0):
    points = np.asarray([pos[vertex] for vertex in simplex], dtype=float)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return center + scale * (points - center)


def _draw_triangle(
    ax,
    simplex,
    pos,
    *,
    facecolor,
    edgecolor,
    alpha,
    linewidth,
    zorder,
    scale=1.0,
):
    points = _ordered_triangle_points(simplex, pos, scale=scale)
    patch = Polygon(
        points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _draw_top_nodes(ax, pos, theme, node_size=620):
    if not pos:
        return None

    ordered = sorted(pos, key=_vertex_sort_key)
    collection = ax.scatter(
        [pos[vertex][0] for vertex in ordered],
        [pos[vertex][1] for vertex in ordered],
        s=node_size,
        c=theme["node_face"],
        edgecolors=theme["node_edge"],
        linewidths=1.8,
        zorder=20,
    )
    collection.set_path_effects(
        [
            path_effects.SimplePatchShadow(
                offset=(1.2, -1.2),
                alpha=0.18,
                shadow_rgbFace="#0F172A",
            ),
            path_effects.Normal(),
        ]
    )
    return collection


def _draw_top_labels(ax, pos, theme, font_size=10):
    for vertex in sorted(pos, key=_vertex_sort_key):
        x, y = pos[vertex]
        ax.text(
            x,
            y,
            str(vertex),
            fontsize=font_size,
            fontweight="semibold",
            ha="center",
            va="center",
            color=theme["label"],
            zorder=21,
        )


def _draw_base_structure(
    ax,
    K,
    pos,
    theme,
    *,
    triangle_alpha=0.18,
    node_size=620,
):
    triangles = sorted(
        (simplex for simplex in K.simplicial_complex if len(simplex) == 3),
        key=_simplex_sort_key,
    )
    for simplex in triangles:
        _draw_triangle(
            ax,
            simplex,
            pos,
            facecolor=theme["base_triangle"],
            edgecolor=theme["base_triangle_edge"],
            alpha=triangle_alpha,
            linewidth=1.3,
            zorder=1,
        )

    _, edges, _ = _build_graph(K)
    for edge in edges:
        u, v = _ordered_edge(edge)
        _draw_trimmed_edge(
            ax,
            pos[u],
            pos[v],
            color=theme["base_edge"],
            width=1.7,
            alpha=0.58,
            zorder=2,
            node_size=node_size,
            node_gap=1.5,
        )


def _style_axes(ax, pos, title, theme):
    ax.set_facecolor(theme["panel_background"])
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=theme["title"],
        pad=16,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if pos:
        xs = np.asarray([point[0] for point in pos.values()])
        ys = np.asarray([point[1] for point in pos.values()])
        x_span = max(float(np.ptp(xs)), 1.0)
        y_span = max(float(np.ptp(ys)), 1.0)
        ax.set_xlim(xs.min() - 0.18 * x_span, xs.max() + 0.18 * x_span)
        ax.set_ylim(ys.min() - 0.18 * y_span, ys.max() + 0.18 * y_span)


def _make_gscat_title(
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    default_title="Categorical cover",
):
    pieces = []
    if gscat_value is not None:
        pieces.append(f"gscat({complex_name}) = {gscat_value}")
    if arboricity_value is not None:
        pieces.append(f"a({complex_name}) = {arboricity_value}")
    return "   В·   ".join(pieces) if pieces else default_title


def _cover_memberships(cover):
    edge_memberships = defaultdict(list)
    vertex_memberships = defaultdict(list)
    triangle_memberships = defaultdict(list)

    for cover_index, subcomplex in enumerate(cover):
        for simplex in subcomplex:
            simplex = frozenset(simplex)
            if len(simplex) == 1:
                vertex = next(iter(simplex))
                vertex_memberships[vertex].append(cover_index)
            elif len(simplex) == 2:
                edge_memberships[_edge_key(simplex)].append(cover_index)
            elif len(simplex) == 3:
                triangle_memberships[simplex].append(cover_index)

    return vertex_memberships, edge_memberships, triangle_memberships


def _validate_cover(K, cover):
    target = {frozenset(simplex) for simplex in K.simplicial_complex}
    elements = [
        {frozenset(simplex) for simplex in subcomplex}
        for subcomplex in cover
    ]
    union = set().union(*elements) if elements else set()
    if union != target:
        missing = target - union
        extra = union - target
        raise ValueError(
            "cover does not coincide with the complex: "
            f"missing={missing}, extra={extra}."
        )


def _draw_vertex_membership_rings(
    ax,
    pos,
    vertex_memberships,
    colors,
    *,
    zorder=15,
):
    if not pos:
        return
    xs = np.asarray([point[0] for point in pos.values()])
    ys = np.asarray([point[1] for point in pos.values()])
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
    base_radius = 0.044 * span
    ring_step = 0.016 * span

    for vertex, members in vertex_memberships.items():
        for local_index, cover_index in enumerate(members):
            ring = Circle(
                pos[vertex],
                radius=base_radius + local_index * ring_step,
                facecolor="none",
                edgecolor=colors[cover_index % len(colors)],
                linewidth=2.5,
                alpha=0.78,
                zorder=zorder - local_index * 0.01,
            )
            ax.add_patch(ring)


def _cover_legend(ax, cover, colors, labels=None):
    if not cover:
        return None
    if labels is not None and len(labels) != len(cover):
        raise ValueError("cover_labels must have the same length as cover.")

    handles = []
    for index, subcomplex in enumerate(cover):
        edge_count = sum(len(simplex) == 2 for simplex in subcomplex)
        triangle_count = sum(len(simplex) == 3 for simplex in subcomplex)
        label = labels[index] if labels is not None else f"$U_{{{index + 1}}}$"
        details = []
        if edge_count:
            noun = "edge" if edge_count == 1 else "edges"
            details.append(f"{edge_count} {noun}")
        if triangle_count:
            noun = "face" if triangle_count == 1 else "faces"
            details.append(f"{triangle_count} {noun}")
        if details:
            label = f"{label}  В·  {', '.join(details)}"

        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[index % len(colors)],
                linewidth=5,
                solid_capstyle="round",
                label=label,
            )
        )

    legend = ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        fontsize=9,
        borderpad=0.8,
        labelspacing=0.65,
        handlelength=2.4,
    )
    legend.set_zorder(30)
    return legend


def draw_complex(
    K,
    ax=None,
    title="Simplicial complex",
    seed=42,
    show=True,
    *,
    pos=None,
    layout="auto",
    node_size=620,
    theme=None,
):
    theme = {**DEFAULT_THEME, **(theme or {})}
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=theme["background"])
        figure.patch.set_facecolor(theme["background"])

    pos = _resolve_positions(K, pos, seed, layout)
    _draw_base_structure(
        ax,
        K,
        pos,
        theme,
        triangle_alpha=0.22,
        node_size=node_size,
    )
    _draw_top_nodes(ax, pos, theme, node_size=node_size)
    _draw_top_labels(ax, pos, theme, font_size=10)
    _style_axes(ax, pos, title, theme)

    if show:
        plt.show()
    return ax, pos


def draw_cover_on_complex(
    K,
    cover,
    ax=None,
    title=None,
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    seed=42,
    show=True,
    offset_shared_edges=True,
    *,
    pos=None,
    layout="auto",
    legend=False,
    cover_labels=None,
    validate=True,
    node_size=600,
    palette=None,
    theme=None,
):
    cover = tuple(cover)
    if validate:
        _validate_cover(K, cover)

    theme = {**DEFAULT_THEME, **(theme or {})}
    colors = tuple(palette or DEFAULT_PALETTE)
    if not colors:
        raise ValueError("palette must contain at least one color.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=theme["background"])
        figure.patch.set_facecolor(theme["background"])

    pos = _resolve_positions(K, pos, seed, layout)
    if title is None:
        title = _make_gscat_title(
            complex_name=complex_name,
            gscat_value=gscat_value,
            arboricity_value=arboricity_value,
        )

    vertex_memberships, edge_memberships, triangle_memberships = (
        _cover_memberships(cover)
    )

    _draw_base_structure(
        ax,
        K,
        pos,
        theme,
        triangle_alpha=0.09,
        node_size=node_size,
    )

    # Shared triangles are nested slightly, so every membership remains visible.
    for simplex in sorted(triangle_memberships, key=_simplex_sort_key):
        members = triangle_memberships[simplex]
        for local_index, cover_index in enumerate(members):
            color = colors[cover_index % len(colors)]
            _draw_triangle(
                ax,
                simplex,
                pos,
                facecolor=color,
                edgecolor=color,
                alpha=0.22 if len(members) == 1 else 0.18,
                linewidth=2.2,
                zorder=4 + local_index * 0.1,
                scale=max(0.68, 1.0 - 0.085 * local_index),
            )

    for edge in sorted(edge_memberships, key=lambda item: tuple(
        _vertex_sort_key(vertex) for vertex in _ordered_edge(item)
    )):
        members = edge_memberships[edge]
        count = len(members)
        for local_index, cover_index in enumerate(members):
            offset = 0.0
            if offset_shared_edges and count > 1:
                offset = (local_index - (count - 1) / 2.0) * 0.045
            _draw_offset_edge(
                ax,
                pos,
                edge,
                colors[cover_index % len(colors)],
                offset=offset,
                width=4.3,
                outline=theme["outline"],
                node_size=node_size,
            )

    _draw_vertex_membership_rings(
        ax,
        pos,
        vertex_memberships,
        colors,
    )
    _draw_top_nodes(ax, pos, theme, node_size=node_size)
    _draw_top_labels(ax, pos, theme, font_size=10)
    _style_axes(ax, pos, title, theme)

    if legend:
        _cover_legend(ax, cover, colors, labels=cover_labels)

    if show:
        plt.show()
    return ax, pos


def draw_complex_and_cover(
    K,
    cover,
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    seed=42,
    *,
    title=None,
    layout="auto",
    show=True,
    save_path=None,
    dpi=180,
    legend=False,
    cover_labels=None,
    validate=True,
    palette=None,
    theme=None,
):
    theme = {**DEFAULT_THEME, **(theme or {})}
    positions = _get_positions(K, seed=seed, layout=layout)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.0),
        facecolor=theme["background"],
        constrained_layout=True,
    )
    figure.patch.set_facecolor(theme["background"])

    draw_complex(
        K,
        ax=axes[0],
        title=f"Original complex В· {complex_name}",
        pos=positions,
        seed=seed,
        layout=layout,
        show=False,
        theme=theme,
    )
    draw_cover_on_complex(
        K,
        cover,
        ax=axes[1],
        title=title,
        complex_name=complex_name,
        gscat_value=gscat_value,
        arboricity_value=arboricity_value,
        pos=positions,
        seed=seed,
        layout=layout,
        show=False,
        legend=legend,
        cover_labels=cover_labels,
        validate=validate,
        palette=palette,
        theme=theme,
    )

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    if show:
        plt.show()
    return figure, axes