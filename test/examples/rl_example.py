"""
This example demonstrates the core functionalities of torch_memory_saver with Paddle backend.
"""

import logging
import sys
from typing import Callable

import paddle
import paddle.device.cuda as paddle_cuda
from paddle.device.cuda.graphs import CUDAGraph
import time
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory

# KV cache: 5 * 100,000,000 * 4 bytes = 2GB
dummy_tensor_size = [5, 100_000_000]
# Model weights: 4096 * 4096 * 4 bytes = 64MB (small enough to avoid OOM)
MODEL_DIM = 4096


def _ptr(x):
    assert isinstance(x, paddle.Tensor)
    return hex(x.data_ptr())


class Model:
    def __init__(self, dim=MODEL_DIM):
        self.dim = dim
        self.create_weights()

    def create_weights(self):
        with torch_memory_saver.region(tag="model_weights"):
            # Use a 1D weight vector instead of Linear layer to avoid matmul
            # (Paddle CUDA graphs have known issues with matmul when input changes)
            self.weight = paddle.full([self.dim], 1.0, dtype='float32').cuda()
        print(f'Model weights created: {_ptr(self.weight)}')

    def forward(self, x):
        # Elementwise multiply then mean (avoids matmul)
        return (x * self.weight).mean()

    def clear_weights(self):
        del self.weight


class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        with torch_memory_saver.region(tag="kv_cache"):
            self.kv_buffer = paddle.full(dummy_tensor_size, value, dtype='float32').cuda()
        print(f'KV cache created: {_ptr(self.kv_buffer)}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg):
        return (arg + self.kv_buffer.mean(axis=1)).mean()


def create_cuda_graph(fn: Callable):
    place = paddle.CUDAPlace(0)

    s = paddle_cuda.Stream()
    cs = paddle_cuda.current_stream()
    s.wait_stream(cs)
    with paddle_cuda.stream_guard(s):
        fn()
    cs.wait_stream(s)

    g = CUDAGraph(place=place, mode='thread_local')
    g.capture_begin()
    fn()
    g.capture_end()

    return g


def run():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    cache = KVCache()
    model = Model()
    original_kv_cache_ptr = _ptr(cache.kv_buffer)
    original_model_weights_ptr = _ptr(model.weight)
    print(f'Original addresses - KV cache: {original_kv_cache_ptr}, Model weights: {original_model_weights_ptr}')

    static_input = paddle.ones([MODEL_DIM], dtype='float32').cuda()

    # Use tensor assignment approach (like cuda_graph.py) instead of pre-allocated buffers
    # Paddle CUDA graphs work by capturing tensor computations; the output tensor
    # from the captured fn is the stable reference for subsequent replays.
    class GraphOutputs:
        static_output = None

    def fn():
        kv_out = cache.execute(static_input[:5])
        model_out = model.forward(static_input)
        GraphOutputs.static_output = kv_out + model_out

    g = create_cuda_graph(fn)

    static_input[...] = 100
    g.replay()
    result = float(GraphOutputs.static_output.item())
    print(f'First execution result: {result}')
    # kv_output = (100 + 1).mean() = 101
    # model_output = (weight * input).mean() = 1 * 100 = 100
    expected = 101.0 + 100.0
    if abs(result - expected) > 0.1:
        raise ValueError(f'Expected {expected}, got {result}')
    print("CUDA graph first execution passed!")

    time.sleep(1)

    print('\n=== Pausing memory regions ===')
    get_and_print_gpu_memory("model_weights: allocated, kv_cache: allocated")
    torch_memory_saver.pause("kv_cache")
    get_and_print_gpu_memory("model_weights: allocated, kv_cache: released")
    torch_memory_saver.pause("model_weights")
    get_and_print_gpu_memory("model_weights: released, kv_cache: released")

    time.sleep(1)

    print('\n=== Resuming memory regions ===')
    torch_memory_saver.resume("model_weights")
    get_and_print_gpu_memory("model_weights: resumed, kv_cache: released")
    torch_memory_saver.resume("kv_cache")
    get_and_print_gpu_memory("model_weights: resumed, kv_cache: resumed")

    time.sleep(1)

    print('\n=== Virtual Address Verification ===')
    kv_cache_ptr_after_resume = _ptr(cache.kv_buffer)
    model_weights_ptr_after_resume = _ptr(model.weight)

    kv_address_unchanged = kv_cache_ptr_after_resume == original_kv_cache_ptr
    model_address_unchanged = model_weights_ptr_after_resume == original_model_weights_ptr

    print(f'KV cache address unchanged: {kv_address_unchanged}')
    print(f'Model weights address unchanged: {model_address_unchanged}')

    assert kv_address_unchanged, f"KV cache virtual address changed"
    assert model_address_unchanged, f"Model weights virtual address changed"
    print("Virtual addresses verification passed!")

    time.sleep(1)

    print('\n=== Testing functionality after resume ===')
    cache.kv_buffer[...] = 2
    model.weight[...] = 2

    static_input[...] = 200
    g.replay()
    result2 = float(GraphOutputs.static_output.item())
    print(f'Second execution result: {result2}')
    # kv_output = (200 + 2).mean() = 202
    # model_output = (weight * input).mean() = 2 * 200 = 400
    expected2 = 202.0 + 400.0
    if abs(result2 - expected2) > 0.1:
        raise ValueError(f'Expected {expected2}, got {result2}')
    print("CUDA graph second execution passed!")

    time.sleep(1)

    print('\n=== Testing selective pause/resume ===')
    torch_memory_saver.pause("kv_cache")
    get_and_print_gpu_memory("model_weights: resumed, kv_cache: released")

    try:
        _ = model.weight[0]
        print("Model weights access successful")
    except Exception:
        print("Model weights access failed")
        raise

    torch_memory_saver.resume("kv_cache")
    get_and_print_gpu_memory("model_weights: resumed, kv_cache: resumed")

    print("Selective pause/resume test passed!")
    print("\nAll tests passed! torch_memory_saver is working correctly.")


if __name__ == '__main__':
    run()
