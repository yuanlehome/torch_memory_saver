import pytest
from contextlib import nullcontext

import multiprocessing
import traceback
import torch_memory_saver
from torch_memory_saver.utils import change_env

from examples import simple, cuda_graph, cpu_backup, rl_example, multi_device, training_engine


def test_simple():
    _test_core(simple.run)


def test_cuda_graph():
    _test_core(cuda_graph.run)


def test_cpu_backup():
    _test_core(cpu_backup.run)


def test_multi_device():
    _test_core(multi_device.run)


def test_rl_example():
    _test_core(rl_example.run)


def test_training_engine():
    with (
        change_env("TMS_INIT_ENABLE", "1"),
        change_env("TMS_INIT_ENABLE_CPU_BACKUP", "1")
    ):
        _test_core(training_engine.run)


def _test_core(fn):
    with torch_memory_saver.configure_subprocess():
        _run_in_subprocess(fn)


def _run_in_subprocess(fn):
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_fn_wrapper, args=(fn, output_queue))
    proc.start()
    proc.join()
    success = proc.exitcode == 0
    assert success


def _subprocess_fn_wrapper(fn, output_queue):
    try:
        print(f"Subprocess execution start")
        fn()
        print(f"Subprocess execution end (may see error messages when CUDA exit which is normal)", flush=True)
        output_queue.put(True)
    except Exception as e:
        print(f"Subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
