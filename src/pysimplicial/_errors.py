class SearchLimitExceeded(RuntimeError):
    """A search budget expired before minimality or reachability was decided."""

    def __init__(self, message, *, gpu_used=False, cover_lower_bound=None, cover_masks=()):
        super().__init__(message)
        self.gpu_used = gpu_used
        self.cover_lower_bound = cover_lower_bound
        self.cover_masks = cover_masks
        self.cover_witnesses = ()


class CudaUnavailableError(RuntimeError):
    """The explicitly requested CUDA backend is unavailable."""
