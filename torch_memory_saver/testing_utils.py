"""Not to be used by end users, but only for tests of the package itself."""

import ctypes
import paddle
import paddle.device.cuda as paddle_cuda


def _cuda_mem_get_info_for_device(gpu_id=0):
    """Get (free, total) GPU memory in bytes via cudaMemGetInfo for a specific device."""
    libcudart = ctypes.CDLL('libcudart.so')

    # Save current device, switch to target, query, then restore
    orig_dev = ctypes.c_int()
    libcudart.cudaGetDevice(ctypes.byref(orig_dev))
    libcudart.cudaSetDevice(ctypes.c_int(gpu_id))
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    ret = libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    assert ret == 0, f"cudaMemGetInfo failed with code {ret}"
    libcudart.cudaSetDevice(orig_dev)
    return free.value, total.value


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message.

    Returns the amount of GPU memory *in use* (total - free), which reflects
    actual physical GPU memory consumption including TMS-managed VMM pages.
    paddle_cuda.memory_reserved() only tracks Paddle's internal allocator pool
    and does NOT decrease when TMS unmaps physical pages via cuMemUnmap.
    """
    free, total = _cuda_mem_get_info_for_device(gpu_id)
    used = total - free
    print(f"GPU {gpu_id} memory: {used / 1024 ** 3:.2f} GB ({message})")
    return used
