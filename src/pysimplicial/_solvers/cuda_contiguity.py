"""Batch the single-vertex moves in a contiguity search on CUDA."""

import numpy as np

from .cuda_restarts import cuda


MAX_VERTICES = 64
_DENSE_TABLE_VERTICES = 12
_THREADS_PER_BLOCK = 128
_AUTO_MIN_JOBS = 256


if cuda is not None:
    from numba import uint64

    @cuda.jit
    def _move_masks(states, faces, incident, target_facets, extensions, full, output):
        job = cuda.grid(1)
        width = states.shape[1]
        row = job // width
        if row >= states.shape[0]:
            return
        vertex = job % width
        allowed = full
        for incident_slot in range(incident.shape[1]):
            face = incident[vertex, incident_slot]
            if face < 0:
                break
            image = uint64(0)
            for slot in range(faces.shape[1]):
                index = faces[face, slot]
                if index < 0:
                    break
                image |= uint64(1) << states[row, index]
            if extensions.shape[0] != 0:
                allowed &= extensions[image]
            else:
                expanded = uint64(0)
                for i in range(target_facets.shape[0]):
                    facet = target_facets[i]
                    if image & facet == image:
                        expanded |= facet
                allowed &= expanded
            if allowed == 0:
                break
        output[row, vertex] = allowed & ~(uint64(1) << states[row, vertex])


def _padded(rows):
    result = np.full((len(rows), max(map(len, rows))), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        result[i, :len(row)] = row
    return result


class _CudaMoves:
    def __init__(self, moves, *, batch_size, auto):
        self.moves = moves
        self.batch_size = batch_size
        self.auto = auto
        self._target_data = None

    def _device_target(self):
        if self._target_data is None:
            target_facets = np.asarray(self.moves.facets, dtype=np.uint64)
            n = len(self.moves.vertices)
            size = 1 << n if n <= _DENSE_TABLE_VERTICES else 0
            extensions = np.fromiter(
                (self.moves.extensions(image) for image in range(size)), dtype=np.uint64, count=size,
            )
            self._target_data = cuda.to_device(target_facets), cuda.to_device(extensions)
        return self._target_data

    def for_domain(self, faces, incident):
        device_domain = None

        def batch(states):
            nonlocal device_domain
            jobs = len(states) * len(incident)
            if self.auto and jobs < _AUTO_MIN_JOBS:
                return None
            if device_domain is None:
                device_domain = cuda.to_device(_padded(faces)), cuda.to_device(_padded(incident))
            target_facets, extensions = self._device_target()
            encoded = cuda.to_device(np.asarray(states, dtype=np.uint8))
            output = cuda.device_array(encoded.shape, dtype=np.uint64)
            blocks = (jobs + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            _move_masks[blocks, _THREADS_PER_BLOCK](
                encoded, *device_domain, target_facets, extensions, np.uint64(self.moves.full), output,
            )
            return output.copy_to_host()

        return batch
