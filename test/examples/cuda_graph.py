import logging
import sys
import time
from typing import Callable

import paddle
import paddle.device.cuda as paddle_cuda
from paddle.device.cuda.graphs import CUDAGraph

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

dummy_tensor_size = [5, 100_000_000]
graph_static_buffer_size = [1_000_000_000]  # 1 GB static buffer simulating graph memory


def _ptr(x):
    assert isinstance(x, paddle.Tensor)
    return hex(x.data_ptr())


class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        with torch_memory_saver.region(tag="kv_cache"):
            self.kv_buffer = paddle.full(dummy_tensor_size, value, dtype='float32').cuda()
        print(f'create_buffers kv_buffer={_ptr(self.kv_buffer)}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg):
        return (arg + self.kv_buffer.mean(axis=1)).mean()


def create_cuda_graph(fn: Callable):
    place = paddle.CUDAPlace(0)

    # warmup
    s = paddle_cuda.Stream()
    cs = paddle_cuda.current_stream()
    s.wait_stream(cs)
    with paddle_cuda.stream_guard(s):
        print('warmup: execute fn')
        fn()
    cs.wait_stream(s)

    # capture: no TMS region active, so intermediate tensors are regular Paddle allocations
    g = CUDAGraph(place=place, mode='thread_local')
    g.capture_begin()
    print('capture: execute fn')
    fn()
    g.capture_end()

    return g


def run():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    cache = KVCache()
    static_input = paddle.zeros([5], dtype='float32').cuda()
    static_output = paddle.zeros([5], dtype='float32').cuda()
    print(f'static_input={_ptr(static_input)} static_output={_ptr(static_output)}')

    def fn():
        nonlocal static_output
        static_output = cache.execute(static_input)

    g = create_cuda_graph(fn)

    # Allocate a large static buffer in "graph" region to test pause/resume of graph region
    # This simulates graph-captured static memory (e.g. activations, buffers)
    with torch_memory_saver.region(tag="graph"):
        graph_static_buffer = paddle.zeros(graph_static_buffer_size, dtype='uint8').cuda()
    print(f'graph_static_buffer={_ptr(graph_static_buffer)}')

    print('replay #1')
    static_input[...] = 100
    g.replay()
    print(f'static_output={static_output}')
    assert float(static_output.item()) == 101, f'static_output={static_output}'

    print('paddle_cuda.empty_cache()')
    paddle_cuda.empty_cache()

    print('sleep...')
    time.sleep(1)

    mem_before_pause = get_and_print_gpu_memory("Before pause")

    print('call memory_saver.pause("kv_cache")')
    torch_memory_saver.pause("kv_cache")
    print('sleep...')
    time.sleep(1)

    mem_after_pause_kv_cache = get_and_print_gpu_memory("After pause kv_cache")
    assert mem_before_pause - mem_after_pause_kv_cache > 400_000_000

    print('call memory_saver.pause("graph")')
    torch_memory_saver.pause("graph")
    print('sleep...')
    time.sleep(1)

    mem_after_pause_graph = get_and_print_gpu_memory("After pause graph")
    assert mem_after_pause_kv_cache - mem_after_pause_graph > 800_000_000

    print('when kv cache is released, we can allocate *other* big tensors')
    other_big_tensor = paddle.zeros([2500_000_000], dtype='uint8').cuda()
    print('sleep...')
    time.sleep(1)
    print(f'other_big_tensor={other_big_tensor}')
    del other_big_tensor
    paddle_cuda.empty_cache()
    print('sleep...')
    time.sleep(1)

    print('call memory_saver.resume("graph")')
    torch_memory_saver.resume("graph")
    print('call memory_saver.resume("kv_cache")')
    torch_memory_saver.resume("kv_cache")

    dummy = paddle.zeros([3], dtype='float32').cuda()
    print(f'dummy={_ptr(dummy)}')

    cache.kv_buffer[...] = 2

    print('replay #2')
    static_input[...] = 200
    g.replay()
    print(f'static_output={static_output}')
    assert float(static_output.item()) == 202, f'static_output={static_output}'

    print('sleep...')
    time.sleep(1)

    print(f'dummy={dummy}')


if __name__ == '__main__':
    run()
