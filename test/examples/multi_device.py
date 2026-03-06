import logging
import sys

import paddle

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Force CUDA context initialization on both GPUs before taking baseline measurements.
    paddle.device.set_device('gpu:0')
    paddle.zeros([1], dtype='uint8').cuda()
    paddle.device.set_device('gpu:1')
    paddle.zeros([1], dtype='uint8').cuda()

    mem_initial = get_and_print_gpu_memory("Initial", gpu_id=0)

    # Allocate on GPU 0: must set active device to GPU 0 first so TMS
    # uses the correct CUDA context (cuCtxGetDevice returns active device)
    paddle.device.set_device('gpu:0')
    with torch_memory_saver.region():
        dev0_a = paddle.full([100_000_000], 10, dtype='uint8').cuda()

    mem_after_dev0 = get_and_print_gpu_memory("alloc dev0_a", gpu_id=0)
    assert mem_after_dev0 >= mem_initial + 80_000_000, \
        f"Expected memory increase after dev0_a alloc, got {mem_after_dev0 - mem_initial} bytes"
    assert dev0_a.place.is_gpu_place(), f"dev0_a should be on GPU, got {dev0_a.place}"

    # Allocate on GPU 1: must set active device to GPU 1 first
    paddle.device.set_device('gpu:1')
    with torch_memory_saver.region():
        dev1_a = paddle.full([100_000_000], 10, dtype='uint8').cuda()

    get_and_print_gpu_memory("alloc dev1_a", gpu_id=1)
    assert dev1_a.place.is_gpu_place(), f"dev1_a should be on GPU, got {dev1_a.place}"

    with torch_memory_saver.region():
        dev1_b = paddle.full([100_000_000], 10, dtype='uint8').cuda()

    assert dev1_b.place.is_gpu_place(), f"dev1_b should be on GPU, got {dev1_b.place}"
    get_and_print_gpu_memory("alloc dev1_b", gpu_id=1)

    # Verify tensor values before pause
    paddle.device.set_device('gpu:0')
    assert dev0_a.cast('int32').mean().item() == 10, "dev0_a values incorrect"
    paddle.device.set_device('gpu:1')
    assert dev1_a.cast('int32').mean().item() == 10, "dev1_a values incorrect"
    assert dev1_b.cast('int32').mean().item() == 10, "dev1_b values incorrect"

    mem_before_pause_0 = get_and_print_gpu_memory("Before pause", gpu_id=0)
    mem_before_pause_1 = get_and_print_gpu_memory("Before pause", gpu_id=1)
    torch_memory_saver.pause()
    mem_after_pause_0 = get_and_print_gpu_memory("After pause", gpu_id=0)
    mem_after_pause_1 = get_and_print_gpu_memory("After pause", gpu_id=1)
    # Pausing should free physical memory for all TMS allocations (across both GPUs)
    total_freed = (mem_before_pause_0 - mem_after_pause_0) + (mem_before_pause_1 - mem_after_pause_1)
    assert total_freed > 2 * 80_000_000, \
        f"Expected >160MB total freed after pause, got {total_freed} bytes"

    torch_memory_saver.resume()
    mem_after_resume_0 = get_and_print_gpu_memory("After resume", gpu_id=0)
    mem_after_resume_1 = get_and_print_gpu_memory("After resume", gpu_id=1)
    # After resume, memory should be restored (across both GPUs)
    total_restored = (mem_after_resume_0 - mem_after_pause_0) + (mem_after_resume_1 - mem_after_pause_1)
    assert total_restored > 2 * 80_000_000, \
        f"Expected >160MB total restored after resume, got {total_restored} bytes"

    # Verify tensor values after resume
    # Note: without enable_cpu_backup, data is lost on pause (pages are freshly allocated)
    # Just verify the tensors are accessible (no CUDA error) and addresses are stable
    paddle.device.set_device('gpu:0')
    _ = dev0_a[0]  # should not raise CUDA error
    paddle.device.set_device('gpu:1')
    _ = dev1_a[0]  # should not raise CUDA error
    _ = dev1_b[0]  # should not raise CUDA error
    print("Tensor access after resume: OK")

    get_and_print_gpu_memory("End", gpu_id=0)


if __name__ == '__main__':
    run()
