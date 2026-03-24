#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <optional>
#include "utils.h"
#include "macro.h"

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle
    CUmemGenericAllocationHandle allocHandle;
#endif
};

struct FixedVaTensorMetadata {
    CUdeviceptr fixed_va;
    size_t bytes;         // granularity 对齐后的大小
    size_t granularity;
    CUdevice device;
    CUmemAllocationProp prop;
    CUmemAccessDesc access;
    CUmemGenericAllocationHandle current_handle;
    bool mapped;
    bool owns_handle;  // true = create_fixed_va 自己创建的, false = remap 来的
};

struct AllocationLookupResult {
    CUmemGenericAllocationHandle allocHandle;
    size_t offset;
    size_t block_size;
    CUdevice device;
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);

    // Fixed VA tensor 管理
    int64_t create_fixed_va(size_t bytes, int device_index);
    void remap_fixed_va(int64_t fixed_va_handle, void* src_ptr, size_t src_size);
    void destroy_fixed_va(int64_t fixed_va_handle);
    void* get_fixed_va_ptr(int64_t fixed_va_handle);

    // 查找包含 ptr 的 TMS 分配
    std::optional<AllocationLookupResult> lookup_allocation(void* ptr, size_t size);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;

    // Fixed VA registry
    std::mutex fixed_va_mutex_;
    int64_t next_fixed_va_id_ = 1;
    std::unordered_map<int64_t, FixedVaTensorMetadata> fixed_va_registry_;
};
