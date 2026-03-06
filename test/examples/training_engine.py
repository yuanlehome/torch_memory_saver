import logging
import os
import sys
from functools import reduce
from typing import List

import paddle
import paddle.device.cuda as paddle_cuda
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run():
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    initial_tensor = paddle.full([1_000_000], 42, dtype='uint8').cuda()
    mem_initial = get_and_print_gpu_memory("Init")

    model_weights = [
        paddle.full([size], 42, dtype='uint8').cuda()
        for size in [1024 ** 3, 1024 ** 2, 1024 ** 1, 42]
    ]
    mem_after_model_weights = get_and_print_gpu_memory("After model weights")

    _execute_forward_pass_and_assert(model_weights)
    mem_after_forward_pass = get_and_print_gpu_memory("After forward pass")
    assert mem_after_forward_pass > mem_initial + 5 * 1024 ** 3

    torch_memory_saver.pause()
    mem_after_pause = get_and_print_gpu_memory("After pause")
    assert mem_after_pause < mem_initial + 200 * 1024 ** 2

    with torch_memory_saver.disable():
        mem_after_disable = get_and_print_gpu_memory("After disable")
        assert mem_after_disable <= mem_after_pause + 10 * 1024 ** 2, \
            f"mem_after_disable={mem_after_disable} should be close to mem_after_pause={mem_after_pause}"

        # Can still execute code in disabled region
        tensor_in_disabled_region = paddle.full([1024 ** 3], 53, dtype='uint8').cuda()
        out = tensor_in_disabled_region.cast('float32').mean().item()
        assert out == 53, f"{out=}"

        mem_after_exec_in_disable = get_and_print_gpu_memory("After exec in disable")
        assert mem_after_exec_in_disable > mem_after_disable + 4 * 1024 ** 3

        del tensor_in_disabled_region

    # should do cleanup
    mem_after_exit_disable = get_and_print_gpu_memory("After exiting disable")
    assert mem_after_exit_disable <= mem_after_pause + 10 * 1024 ** 2

    torch_memory_saver.resume()
    mem_after_resume = get_and_print_gpu_memory("After resume")
    # Model weights should be restored: delta should be ~(mem_after_model_weights - mem_initial)
    delta_expect = mem_after_model_weights - mem_initial
    delta_actual = mem_after_resume - mem_after_exit_disable
    assert delta_expect - 50 * 1024 ** 2 < delta_actual < delta_expect + 50 * 1024 ** 2

    _execute_forward_pass_and_assert(model_weights)
    get_and_print_gpu_memory("After second forward pass")

    # simulate cache eviction
    # Get total GPU memory
    props = paddle_cuda.get_device_properties(0)
    total_mem_raw = props.total_memory  # may be int (bytes) or str like '80994MB'
    if isinstance(total_mem_raw, str):
        total_mem = int(total_mem_raw.replace('MB', '')) * 1024 * 1024
    else:
        total_mem = int(total_mem_raw)
    print(f"Total memory: {total_mem / 1e9}GB")

    a = paddle.full([int(total_mem * 0.3)], 42, dtype='uint8').cuda()
    get_and_print_gpu_memory("[cache-eviction-test] after alloc a")
    b = paddle.full([int(total_mem * 0.3)], 42, dtype='uint8').cuda()
    get_and_print_gpu_memory("[cache-eviction-test] after alloc b")
    del a, b
    get_and_print_gpu_memory("[cache-eviction-test] after del a,b")
    c = paddle.full([int(total_mem * 0.6)], 42, dtype='uint8').cuda()
    get_and_print_gpu_memory("[cache-eviction-test] after alloc c")
    initial_tensor += 1
    assert (initial_tensor.cast('int32').max() == 43) and (initial_tensor.cast('int32').min() == 43)
    get_and_print_gpu_memory("[cache-eviction-test] after using other tensors")

    del initial_tensor


def _execute_forward_pass_and_assert(weights: List[paddle.Tensor]):
    # simulate large activations during forward pass
    ones = paddle.ones([1024 ** 3], dtype='float32').cuda()
    sum_avg_weights = reduce(lambda a, b: (a + b) / 2, [w.cast('float32').mean() for w in weights])
    outs = ones * sum_avg_weights
    out = outs.mean().item()
    assert out == 42, f"{out=}"


if __name__ == '__main__':
    run()
