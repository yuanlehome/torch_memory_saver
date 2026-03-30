import torch
from torch_memory_saver import torch_memory_saver

# 创建 fixed VA tensor（通过 TMS 集成的 API，直接返回 tensor）
handle, a = torch_memory_saver.create_fixed_va_tensor(50000, torch.float32, torch.device('cuda', 0))

print("a.data_ptr =", hex(a.data_ptr()))
print(f"{a=}")

# 在 region 内分配 b
with torch_memory_saver.region():
    b = torch.full((50000,), 100, dtype=torch.float32, device='cuda')

print(f"b.data_ptr = {hex(b.data_ptr())}")
print(f"{b=}")

# remap：TMS 内部查找 b 的 allocation，直接用原始 handle 做 cuMemMap（零拷贝）
torch_memory_saver.remap_fixed_va_tensor(handle, b)

# remap 前后，a.data_ptr() 应该不变
print("a.data_ptr after remap =", hex(a.data_ptr()))
print(f"{a=}")

torch_memory_saver.destroy_fixed_va_tensor(handle)

# LD_PRELOAD=./torch_memory_saver_hook_mode_preload.abi3.so python test_fixed_va_ext.py
