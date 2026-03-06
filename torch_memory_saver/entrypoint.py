import atexit
import ctypes

import numpy as np
import logging
import os
from contextlib import contextmanager
from typing import Optional
import paddle
import paddle.device.cuda as paddle_cuda

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"


class TorchMemorySaver:
    def __init__(self):
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        self._ensure_initialized()
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
    ):
        """Similar to `paddle.device.cuda.graphs.CUDAGraph`, but ensures memory in it to be pauseable."""
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
        ):
            yield

    @contextmanager
    def disable(self):
        self._ensure_initialized()
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.resume(tag=tag)

    # for compatibility
    @property
    def enabled(self):
        return True

    @property
    def memory_margin_bytes(self):
        raise NotImplementedError("Only setter is supported")

    @memory_margin_bytes.setter
    def memory_margin_bytes(self, value: int):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    def get_cpu_backup(self, x, zero_copy: bool = False):
        self._ensure_initialized()
        return self._impl.get_cpu_backup(x, zero_copy=zero_copy)

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        self._impl = _TorchMemorySaverImpl()


class _TorchMemorySaverImpl:
    def __init__(self):
        self._hook_util = HookUtilBase.create()
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
            yield

    @contextmanager
    def cuda_graph(self, cuda_graph, tag: str, enable_cpu_backup: bool):
        from paddle.device.cuda.graphs import CUDAGraph
        assert isinstance(cuda_graph, CUDAGraph), f"Expected paddle CUDAGraph, got {type(cuda_graph)}"
        cuda_graph.capture_begin()
        try:
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield
        finally:
            cuda_graph.capture_end()

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool):
        assert not self._binary_wrapper.cdll.tms_get_interesting_region()
        original_enable_cpu_backup = self._binary_wrapper.cdll.tms_get_enable_cpu_backup()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
        try:
            yield
        finally:
            assert self._binary_wrapper.cdll.tms_get_interesting_region()
            self._binary_wrapper.set_config(tag=_TAG_DEFAULT, interesting_region=False, enable_cpu_backup=original_enable_cpu_backup)

    @contextmanager
    def disable(self):
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        # Flush Paddle's allocator pool before entering the disabled region.
        # This ensures that any cached TMS-managed addresses (which may have been
        # paused/unmapped) are properly freed via TMS's free() before we allocate
        # new memory in the disabled region. Without this, Paddle might return a
        # cached TMS address that has no physical backing.
        paddle_cuda.empty_cache()
        try:
            yield
            # Also flush at exit to clean up any allocations made in the disabled region
            paddle_cuda.empty_cache()
        finally:
            self._binary_wrapper.cdll.tms_set_interesting_region(True)

    def pause(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)

    def get_cpu_backup(self, x, zero_copy: bool = False):
        assert x.place.is_gpu_place(), f"{x.place=}"
        assert x.is_contiguous(), f"{x.shape=} {x.strides()=} {x.dtype=}"

        nbytes = x.numel() * x.element_size()
        gpu_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
        cpu_ptr = self._binary_wrapper.cdll.tms_get_cpu_backup_pointer(gpu_ptr, nbytes)
        if not cpu_ptr:
            return None

        np_untyped = np.ctypeslib.as_array(cpu_ptr, shape=(nbytes,))
        assert np_untyped.dtype == np.uint8, f"{np_untyped.dtype=} {np_untyped.shape=}"

        import paddle
        ans_untyped = paddle.to_tensor(np_untyped, place='cpu')
        # view as original dtype and shape
        ans = ans_untyped.view(dtype=_paddle_dtype_to_str(x.dtype)).reshape(x.shape)

        # For simplicity and safety
        if not zero_copy:
            ans = ans.clone()

        assert ans.place.is_cpu_place(), f"{ans.place=}"
        assert ans.dtype == x.dtype, f"{ans.dtype=} {x.dtype=}"
        assert list(ans.shape) == list(x.shape), f"{ans.shape=} {x.shape=}"
        return ans


def _paddle_dtype_to_str(dtype):
    """Convert paddle dtype to numpy dtype string for view operations."""
    import paddle
    dtype_map = {
        paddle.float32: 'float32',
        paddle.float64: 'float64',
        paddle.float16: 'float16',
        paddle.bfloat16: 'bfloat16',
        paddle.int32: 'int32',
        paddle.int64: 'int64',
        paddle.int16: 'int16',
        paddle.int8: 'int8',
        paddle.uint8: 'uint8',
        paddle.bool: 'bool',
    }
    return dtype_map.get(dtype, str(dtype))
