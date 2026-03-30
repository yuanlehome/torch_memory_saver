# 将 fixed_va_ext 集成到 torch_memory_saver (TMS)

## Context

当前 `fixed_va_ext` 是一个独立的 pybind11 扩展，使用 `cuMemRetainAllocationHandle` 获取 src tensor 的 VMM handle 进行 remap。但 PyTorch caching allocator 会一次调用 `cudaMalloc(~20MB)`，TMS hook 创建一个 20MB 的 VMM 分配，tensor `b` 只是其中的 sub-allocation。`cuMemRetainAllocationHandle` 返回的是 20MB 的 handle，无法 map 到 2MB 的 fixed VA。

由于 `cuMemMap` 的 `offset` 参数必须 granularity 对齐（2MB），无法对任意 sub-allocation 做 partial remap。

**解决方案**：将 fixed VA 功能集成到 TMS 内部，在 `TorchMemorySaver::malloc` 中为 fixed VA 场景做**独立的、大小精确匹配的 VMM 分配**，绕过 caching allocator 的 block 合并问题。具体做法是让 remap 时 TMS 内部查找包含 src_ptr 的 allocation，直接使用其原始 handle + offset=0 做 cuMemMap（当 src_ptr 恰好在 block 起始位置时），或在一般情况下新建一个与 fixed VA 大小相同的独立 VMM 分配，从 TMS 管理的内存中 unmap 对应区域并 remap 到 fixed VA。

**最终采用的核心思路**：remap 时在 TMS 内部直接操作 —— 通过查找 `allocation_metadata_` 获取包含 src_ptr 的 block 的 `allocHandle`，如果 src_ptr 恰好等于 block base 且 block size == fixed VA size，则直接 remap（零拷贝）；否则做一个新的 `cuMemCreate` + `cuMemMap` 到 fixed VA，然后 `cudaMemcpyAsync(DeviceToDevice)` 拷贝 src 数据到 fixed VA（这个场景用户说了不能用 memcpy，所以需要确保场景一成立——通过让 b 独占一个 TMS 分配来保证）。

**关键约束**：要实现真正的零拷贝 remap，src tensor 必须独占一个完整的 TMS VMM 分配块（offset=0, size 精确匹配）。TMS 可以通过一个专用 API 来分配这类 tensor，不走 PyTorch caching allocator。

## 修改文件

1. **`csrc/core.h`** — 添加 FixedVaTensor 结构体和相关方法声明
2. **`csrc/core.cpp`** — 实现 create/remap/destroy_fixed_va 和 lookup_allocation
3. **`csrc/entrypoint.cpp`** — 添加 extern "C" 导出函数
4. **`torch_memory_saver/binary_wrapper.py`** — 添加新函数的 ctypes 签名
5. **`torch_memory_saver/entrypoint.py`** — 添加 Python API
6. **`fixed_va_ext/test_fixed_va_ext.py`** — 更新测试使用集成后的 API

## 详细设计

### Step 1: core.h — 添加 FixedVaTensor 结构体和新方法

```cpp
struct FixedVaTensorMetadata {
    CUdeviceptr fixed_va;
    size_t bytes;         // granularity 对齐后的大小
    size_t granularity;
    CUdevice device;
    CUmemAllocationProp prop;
    CUmemAccessDesc access;
    CUmemGenericAllocationHandle current_handle;
    bool mapped;
};
```

在 `TorchMemorySaver` 类中添加：
```cpp
// Fixed VA 管理
int64_t create_fixed_va(size_t bytes, int device_index);
void remap_fixed_va(int64_t fixed_va_handle, void* src_ptr, size_t src_size);
void destroy_fixed_va(int64_t fixed_va_handle);
void* get_fixed_va_ptr(int64_t fixed_va_handle);

// 查找包含 ptr 的 TMS 分配，返回 handle 和 offset
struct AllocationLookupResult {
    CUmemGenericAllocationHandle allocHandle;
    size_t offset;
    size_t block_size;
    CUdevice device;
};
std::optional<AllocationLookupResult> lookup_allocation(void* ptr, size_t size);

// 新成员
std::mutex fixed_va_mutex_;
int64_t next_fixed_va_id_ = 1;
std::unordered_map<int64_t, FixedVaTensorMetadata> fixed_va_registry_;
```

### Step 2: core.cpp — 实现核心逻辑

**`lookup_allocation`**：复用 `get_cpu_backup_pointer` 的 range-containment 扫描模式，返回 `{allocHandle, offset, block_size, device}`。

**`create_fixed_va`**：
1. 设置 `CUmemAllocationProp`（与 TMS 的 `cu_mem_create` 保持一致的 prop）
2. `cuMemGetAllocationGranularity` 获取 granularity
3. `align_up(bytes, granularity)` 对齐
4. `cuMemAddressReserve` 预留固定 VA
5. `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` 创建初始 backing
6. 注册到 `fixed_va_registry_`
7. 返回 handle ID

**`remap_fixed_va`**：
1. `lookup_allocation(src_ptr, src_size)` 查找包含 src 的 TMS 分配
2. 验证 offset == 0 且 block_size == fixed_va_bytes（零拷贝条件）
3. Unmap fixed VA 当前的 backing
4. Release fixed VA 当前的 handle
5. **直接使用 TMS 分配的 allocHandle**：`cuMemMap(fixed_va, bytes, 0, lookup_result.allocHandle, 0)`
6. `cuMemSetAccess` 设置权限
7. 更新 fixed_va 的 current_handle（注意：不 release 这个 handle，因为它由 TMS allocation_metadata_ 管理）

**关键**：remap 后 fixed VA 与 src tensor **共享同一块物理内存**（dual-map），修改 fixed VA 的内容等于修改 src 的内容，反之亦然。fixed VA 的 current_handle 不应 release，因为所有权在 TMS allocation_metadata_。析构 fixed VA 时只需 unmap + free VA，不 release handle。

**`destroy_fixed_va`**：
1. 如果 mapped，`cuMemUnmap`
2. 如果 current_handle 是 create_fixed_va 自己创建的初始 handle，则 release；如果是 remap 来的，则不 release
3. `cuMemAddressFree`

### Step 3: entrypoint.cpp — 添加 extern "C" 导出

```cpp
extern "C" {
int64_t tms_create_fixed_va(size_t bytes, int device_index) {
    return TorchMemorySaver::instance().create_fixed_va(bytes, device_index);
}

void* tms_get_fixed_va_ptr(int64_t handle) {
    return TorchMemorySaver::instance().get_fixed_va_ptr(handle);
}

void tms_remap_fixed_va(int64_t handle, void* src_ptr, size_t src_size) {
    TorchMemorySaver::instance().remap_fixed_va(handle, src_ptr, src_size);
}

void tms_destroy_fixed_va(int64_t handle) {
    TorchMemorySaver::instance().destroy_fixed_va(handle);
}
}
```

### Step 4: binary_wrapper.py — 添加 ctypes 签名

```python
cdll.tms_create_fixed_va.argtypes = [ctypes.c_size_t, ctypes.c_int]
cdll.tms_create_fixed_va.restype = ctypes.c_int64

cdll.tms_get_fixed_va_ptr.argtypes = [ctypes.c_int64]
cdll.tms_get_fixed_va_ptr.restype = ctypes.c_void_p

cdll.tms_remap_fixed_va.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t]

cdll.tms_destroy_fixed_va.argtypes = [ctypes.c_int64]
```

### Step 5: entrypoint.py — Python API

在 `_TorchMemorySaverImpl` 中添加：
```python
def create_fixed_va_tensor(self, numel, dtype, device):
    element_size = torch.tensor([], dtype=dtype).element_size()
    bytes_needed = numel * element_size
    handle = self._binary_wrapper.cdll.tms_create_fixed_va(bytes_needed, device.index or 0)
    ptr = self._binary_wrapper.cdll.tms_get_fixed_va_ptr(handle)
    tensor = torch.from_blob(ptr, [numel], dtype=dtype, device=device)
    return handle, tensor

def remap_fixed_va_tensor(self, handle, src_tensor):
    self._binary_wrapper.cdll.tms_remap_fixed_va(
        handle,
        ctypes.c_void_p(src_tensor.data_ptr()),
        src_tensor.numel() * src_tensor.element_size()
    )

def destroy_fixed_va_tensor(self, handle):
    self._binary_wrapper.cdll.tms_destroy_fixed_va(handle)
```

在 `TorchMemorySaver`（公共 facade）上暴露这些方法。

### Step 6: 更新测试

```python
from torch_memory_saver import torch_memory_saver

# 创建 fixed VA tensor
handle, a = torch_memory_saver.create_fixed_va_tensor(5, torch.float32, torch.device('cuda', 0))
a = a.view(torch.bfloat16).reshape(10, 1)

# 用 TMS 的 malloc（绕过 caching allocator）直接分配 b，确保独占一个 VMM block
# 或者确保 b 是 region 中第一个且唯一的分配
with torch_memory_saver.region():
    b = torch.full((5,), 100, dtype=torch.float32, device='cuda')

# remap：TMS 内部查找 b 的 allocation，直接用原始 handle 做 cuMemMap
torch_memory_saver.remap_fixed_va_tensor(handle, b)

print(f"a.data_ptr after remap = {hex(a.data_ptr())}")
print(f"{a=}")

torch_memory_saver.destroy_fixed_va_tensor(handle)
```

## 零拷贝约束

`remap_fixed_va` 要求 src tensor 满足以下条件（不用 memcpy）：
- src 必须位于某个 TMS 分配的 **起始位置**（offset == 0）
- 该 TMS 分配的 size == fixed VA 的 size（granularity 对齐后）

如果条件不满足，`remap_fixed_va` 抛出错误并给出明确的提示信息。

## 验证

1. `cd /root/paddlejob/workspace/env_run/output/liuyuanle/learn/torch_memory_saver`
2. `rm -rf build/ *.so && python setup.py build_ext --inplace`（用 vllm_py310 的 python）
3. `cd fixed_va_ext && LD_PRELOAD=../torch_memory_saver_hook_mode_preload.abi3.so python test_fixed_va_ext.py`
4. 验证 remap 前后 `a.data_ptr()` 不变，且 `a` 的内容反映了 `b` 的数据

