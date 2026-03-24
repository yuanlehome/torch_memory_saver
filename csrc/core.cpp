#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#include <cstring>

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));

            // TODO may provide a flag to choose whether to free immediately
            // (users may want to lazily free to reduce re-alloc time)
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                  << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}

// ---------------------------------------- Fixed VA Tensor ----------------------------------------

std::optional<AllocationLookupResult> TorchMemorySaver::lookup_allocation(void* query_ptr, size_t query_size) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    uint8_t* qptr = (uint8_t*) query_ptr;
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t* base = (uint8_t*) it->first;
        AllocationMetadata& metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.size;
#endif

        if ((base <= qptr) && (qptr + query_size <= base + total_size)) {
            size_t offset = qptr - base;
            return AllocationLookupResult{
                metadata.allocHandle,
                offset,
                total_size,
                metadata.device
            };
        }
    }
    return std::nullopt;
}

static size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

int64_t TorchMemorySaver::create_fixed_va(size_t bytes, int device_index) {
    SIMPLE_CHECK(bytes > 0, "create_fixed_va: bytes must be > 0");

    CURESULT_CHECK(cuInit(0));

    CUdevice dev;
    CURESULT_CHECK(cuDeviceGet(&dev, device_index));

    FixedVaTensorMetadata meta{};
    meta.device = dev;
    meta.mapped = false;
    meta.owns_handle = true;

    // 设置 prop，与 TMS 的 cu_mem_create 保持一致
    std::memset(&meta.prop, 0, sizeof(meta.prop));
    meta.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    meta.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    meta.prop.location.id = device_index;
    meta.prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    int rdma_flag = 0;
    cuDeviceGetAttribute(&rdma_flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev);
    if (rdma_flag) {
        meta.prop.allocFlags.gpuDirectRDMACapable = 1;
    }

    CURESULT_CHECK(cuMemGetAllocationGranularity(
        &meta.granularity, &meta.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    meta.bytes = align_up(bytes, meta.granularity);

    meta.access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    meta.access.location.id = device_index;
    meta.access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Reserve 固定 VA
    CURESULT_CHECK(cuMemAddressReserve(&meta.fixed_va, meta.bytes, meta.granularity, 0, 0));

    // 创建初始 backing
    CURESULT_CHECK(cuMemCreate(&meta.current_handle, meta.bytes, &meta.prop, 0));
    CURESULT_CHECK(cuMemMap(meta.fixed_va, meta.bytes, 0, meta.current_handle, 0));
    CURESULT_CHECK(cuMemSetAccess(meta.fixed_va, meta.bytes, &meta.access, 1));
    meta.mapped = true;

    int64_t id;
    {
        const std::lock_guard<std::mutex> lock(fixed_va_mutex_);
        id = next_fixed_va_id_++;
        fixed_va_registry_.emplace(id, meta);
    }

    return id;
}

void* TorchMemorySaver::get_fixed_va_ptr(int64_t fixed_va_handle) {
    const std::lock_guard<std::mutex> lock(fixed_va_mutex_);
    auto it = fixed_va_registry_.find(fixed_va_handle);
    SIMPLE_CHECK(it != fixed_va_registry_.end(), "get_fixed_va_ptr: invalid handle");
    return reinterpret_cast<void*>(it->second.fixed_va);
}

void TorchMemorySaver::remap_fixed_va(int64_t fixed_va_handle, void* src_ptr, size_t src_size) {
    FixedVaTensorMetadata* meta = nullptr;
    {
        const std::lock_guard<std::mutex> lock(fixed_va_mutex_);
        auto it = fixed_va_registry_.find(fixed_va_handle);
        SIMPLE_CHECK(it != fixed_va_registry_.end(), "remap_fixed_va: invalid handle");
        meta = &it->second;
    }

    // 查找包含 src_ptr 的 TMS 分配
    auto lookup = lookup_allocation(src_ptr, src_size);
    SIMPLE_CHECK(lookup.has_value(),
        "remap_fixed_va: src_ptr is not within any TMS-managed allocation. "
        "Make sure src tensor was allocated inside torch_memory_saver.region()");

    SIMPLE_CHECK(lookup->offset == 0,
        "remap_fixed_va: src_ptr must be at the start of a TMS allocation (offset must be 0). "
        "Ensure the src tensor is the first and only allocation in its region, "
        "or use a dedicated TMS malloc to avoid caching allocator sub-allocation");

    SIMPLE_CHECK(lookup->block_size == meta->bytes,
        "remap_fixed_va: TMS allocation size must match fixed VA size. "
        "TMS block_size=" + std::to_string(lookup->block_size) +
        " fixed_va_bytes=" + std::to_string(meta->bytes));

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Unmap fixed VA 当前的 backing
    if (meta->mapped) {
        CURESULT_CHECK(cuMemUnmap(meta->fixed_va, meta->bytes));
        meta->mapped = false;
    }

    // 只 release 自己创建的 handle，remap 来的 handle 所有权在 allocation_metadata_
    if (meta->owns_handle && meta->current_handle) {
        CURESULT_CHECK(cuMemRelease(meta->current_handle));
    }

    // 使用 TMS 分配的原始 handle 做 dual-map（零拷贝）
    CURESULT_CHECK(cuMemMap(meta->fixed_va, meta->bytes, 0, lookup->allocHandle, 0));
    CURESULT_CHECK(cuMemSetAccess(meta->fixed_va, meta->bytes, &meta->access, 1));

    meta->current_handle = lookup->allocHandle;
    meta->mapped = true;
    meta->owns_handle = false;  // handle 所有权在 TMS allocation_metadata_
}

void TorchMemorySaver::destroy_fixed_va(int64_t fixed_va_handle) {
    FixedVaTensorMetadata meta{};
    {
        const std::lock_guard<std::mutex> lock(fixed_va_mutex_);
        auto it = fixed_va_registry_.find(fixed_va_handle);
        if (it == fixed_va_registry_.end()) return;
        meta = it->second;
        fixed_va_registry_.erase(it);
    }

    if (meta.mapped && meta.fixed_va) {
        cuMemUnmap(meta.fixed_va, meta.bytes);
    }
    if (meta.owns_handle && meta.current_handle) {
        cuMemRelease(meta.current_handle);
    }
    if (meta.fixed_va) {
        cuMemAddressFree(meta.fixed_va, meta.bytes);
    }
}
