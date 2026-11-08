"""Finite simplicial complexes, contiguity and simplicial category."""

from ._errors import CudaUnavailableError, SearchLimitExceeded
from .category import (
    CategoryResult,
    GSCatResult,
    contiguity_distance,
    cuda_status,
    gscat,
    is_categorical,
    scat,
    verify_gscat_cover,
)
from .collapse import core, dominators, is_strongly_collapsible, strong_core
from .complex import SimplicialComplex
from .contiguity import contiguity_chain, contiguity_class, is_contiguous, same_contiguity_class
from .maps import SimplicialMap
from .operations import categorical_product, connected_components, disjoint_union, graph_join, join, link, relabel, wedge

__all__ = [
    "CategoryResult", "CudaUnavailableError", "GSCatResult", "SearchLimitExceeded",
    "SimplicialComplex", "SimplicialMap", "categorical_product", "connected_components",
    "contiguity_chain", "contiguity_class", "contiguity_distance", "core", "cuda_status",
    "disjoint_union", "dominators", "graph_join", "gscat", "is_categorical",
    "is_contiguous", "is_strongly_collapsible", "join", "link", "relabel",
    "same_contiguity_class", "scat", "strong_core", "verify_gscat_cover", "wedge",
]
