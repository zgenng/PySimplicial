from functools import lru_cache
from importlib.resources import files
import json

from ..complex import SimplicialComplex


@lru_cache(maxsize=None)
def _data(filename):
    return json.loads(files(__package__).joinpath(filename).read_text(encoding="utf-8"))


def triangulation_names() -> tuple[str, ...]:
    return tuple(_data("triangulations.json"))


def triangulation(name: str) -> SimplicialComplex:
    return SimplicialComplex(_data("triangulations.json")[name])


def tetrahedron_boundary() -> SimplicialComplex:
    return triangulation("S2")


def torus() -> SimplicialComplex:
    return triangulation("T2")


def projective_plane() -> SimplicialComplex:
    return triangulation("RP2")


def orientable_surface(genus: int) -> SimplicialComplex:
    if not 0 <= genus <= 6:
        raise ValueError("The catalog includes orientable genera 0 through 6.")
    return triangulation(("S2", "T2")[genus] if genus < 2 else f"M{genus}")


def nonorientable_surface(genus: int) -> SimplicialComplex:
    if not 1 <= genus <= 9:
        raise ValueError("The catalog includes nonorientable genera 1 through 9.")
    return projective_plane() if genus == 1 else triangulation(f"NoS{genus}")


