#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <acl/acl.h>

#include "paddle/device_ext.h"

#define ACL_CHECK(func)                                                       \
  do {                                                                        \
    auto acl_ret = func;                                                      \
    if (acl_ret != ACL_ERROR_NONE) {                                          \
      std::cerr << "Call " << #func << " failed : " << acl_ret << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;             \
      {                                                                       \
        const char *aclRecentErrMsg = nullptr;                                \
        aclRecentErrMsg = aclGetRecentErrMsg();                               \
        if (aclRecentErrMsg != nullptr) {                                     \
          printf("%s\n", aclRecentErrMsg);                                    \
        } else {                                                              \
          printf("Failed to get recent error message.\n");                    \
        }                                                                     \
      }                                                                       \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

class AlignnedAllocator {
 public:
  void *Alloc(size_t size, size_t align) {
    std::lock_guard<std::mutex> lock(mtx_);
    ProcessEvents();
    void *p = nullptr;
    ACL_CHECK(aclrtMallocHost(&p, size + align));
    void *ret =
        reinterpret_cast<void *>(reinterpret_cast<size_t>(p) + align -
                                 (reinterpret_cast<size_t>(p) & (align - 1)));
    recorded_events_[ret] = {p, nullptr};
    return ret;
  }

  ~AlignnedAllocator() {
    // ClearEvent();
  }

  void Record(void *p, aclrtEvent event) {
    std::lock_guard<std::mutex> lock(mtx_);
    recorded_events_[p].second = event;
  }

  void ClearEvent() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = recorded_events_.begin(); it != recorded_events_.end();) {
      aclrtEvent event = it->second.second;
      if (!event) continue;
      ACL_CHECK(aclrtSynchronizeEvent(event));
      void *ptr = it->second.first;
      ACL_CHECK(aclrtFreeHost(ptr));
      ACL_CHECK(aclrtDestroyEvent(event));
      recorded_events_.erase(it++);
    }
  }

  void ProcessEvents() {
    for (auto it = recorded_events_.begin(); it != recorded_events_.end();) {
      aclrtEvent event = it->second.second;
      if (!event) continue;
      aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
      ACL_CHECK(aclrtQueryEvent(event, &status));

      if (status == ACL_EVENT_STATUS_COMPLETE) {
        void *ptr = it->second.first;
        ACL_CHECK(aclrtFreeHost(ptr));
        recorded_events_.erase(it++);
        ACL_CHECK(aclrtDestroyEvent(event));
      } else {
        ++it;
      }
    }
  }

 private:
  std::unordered_map<void *, std::pair<void *, aclrtEvent>> recorded_events_;
  std::mutex mtx_;
};

class AlignnedAllocatorList {
 public:
  AlignnedAllocatorList(size_t device_count)
      : allocator_list(device_count, nullptr) {}

  void Init(size_t dev_id) { allocator_list[dev_id] = new AlignnedAllocator; }

  void Deinit(size_t dev_id) {
    delete allocator_list[dev_id];
    allocator_list[dev_id] = nullptr;
  }

  AlignnedAllocator *GetAllocator(size_t dev_id) {
    return allocator_list[dev_id];
  }

 private:
  std::vector<AlignnedAllocator *> allocator_list;
};

static AlignnedAllocatorList *global_allocator_list = nullptr;

inline size_t get_current_device_id() {
  int dev_id = 0;
  ACL_CHECK(aclrtGetDevice(&dev_id));
  return dev_id;
}

inline size_t get_devices_count() { return 1; }

C_Status Init() {
  // ACL_CHECK(aclInit(nullptr));
  size_t count = get_devices_count();
  if (count) {
    global_allocator_list = new AlignnedAllocatorList(count);
  }
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  ACL_CHECK(aclrtSetDevice(device->id));
  if (global_allocator_list) {
    global_allocator_list->Init(device->id);
  }
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  ACL_CHECK(aclrtSetDevice(device->id));
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = get_current_device_id();
  return C_SUCCESS;
}

C_Status ReleaseDevice(const C_Device device) {
  ACL_CHECK(aclrtSetDevice(device->id));
  if (global_allocator_list) {
    // global_allocator_list->GetAllocator(device->id)->ClearEvent();
    global_allocator_list->Deinit(device->id);
  }
  // ACL_CHECK(aclrtResetDevice(device->id));
  return C_SUCCESS;
}

C_Status Finalize() {
  if (global_allocator_list) {
    delete global_allocator_list;
    global_allocator_list = nullptr;
  }
  // ACL_CHECK(aclFinalize());
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  *count = get_devices_count();
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device, void *dst, const void *src,
                   size_t size) {
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device, void *dst, const void *src,
                   size_t size) {
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device, void *dst, const void *src,
                   size_t size) {
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
  return C_SUCCESS;
}

// https://support.huaweicloud.com/aclcppdevg-cann504alpha2infer/aclcppdevg_01_0061.html
C_Status AsyncMemCpyH2D(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size) {
  auto allocator = global_allocator_list->GetAllocator(get_current_device_id());
  void *tmp = allocator->Alloc(size, 64);
  aclrtEvent event;
  ACL_CHECK(aclrtCreateEvent(&event));
  memcpy(tmp, src, size);
  ACL_CHECK(aclrtMemcpyAsync(dst, size, tmp, size, ACL_MEMCPY_HOST_TO_DEVICE,
                             (aclrtStream)(stream)));
  ACL_CHECK(aclrtRecordEvent(event, (aclrtStream)(stream)));
  allocator->Record(tmp, event);
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size) {
  ACL_CHECK(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             (aclrtStream)(stream)));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size) {
  ACL_CHECK(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST,
                             (aclrtStream)(stream)));
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  ACL_CHECK(aclrtSetDevice(device->id));
  void *data;
  aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
  ACL_CHECK(aclrtSetDevice(device->id));
  void *data = nullptr;
  ACL_CHECK(aclrtMallocHost(&data, size));
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  ACL_CHECK(aclrtSetDevice(device->id));
  ACL_CHECK(aclrtFree(ptr));
  return C_SUCCESS;
}

C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
  ACL_CHECK(aclrtFreeHost(ptr));
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  ACL_CHECK(aclrtCreateStream(reinterpret_cast<aclrtStream *>(stream)));
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  ACL_CHECK(aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  ACL_CHECK(aclrtCreateEvent(reinterpret_cast<aclrtEvent *>(event)));
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  ACL_CHECK(aclrtRecordEvent(reinterpret_cast<aclrtEvent *>(event),
                             reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  ACL_CHECK(aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  ACL_CHECK(aclrtSynchronizeDevice());
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  ACL_CHECK(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  ACL_CHECK(aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device, C_Stream stream,
                         C_Event event) {
  ACL_CHECK(aclrtStreamWaitEvent(reinterpret_cast<aclrtStream>(stream),
                                 reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device, size_t *total_memory,
                        size_t *free_memory) {
  aclrtGetMemInfo(ACL_HBM_MEM, free_memory, total_memory);
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 512;
  return C_SUCCESS;
}

C_Status ExtraPaddingSize(const C_Device device, size_t *size) {
  *size = 32;
  return C_SUCCESS;
}

void InitPlugin(RuntimePluginParams *params) {
  if (params->size != sizeof(RuntimePluginParams) &&
      params->interface->size != sizeof(C_DeviceInterface)) {
    return;
  }

  params->device_type = "Ascend910";
  params->sub_device_type = "V100";
  params->version.major = PADDLE_DEVICE_PLUGIN_MAJOR_VERSION;
  params->version.minor = PADDLE_DEVICE_PLUGIN_MINOR_VERSION;
  params->version.patch = PADDLE_DEVICE_PLUGIN_PATCH_VERSION;

  memset((void *)params->interface, 0, sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = ReleaseDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = HostAllocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = HostDeallocate;

  params->interface->visible_devices_count = GetDevicesCount;
  params->interface->device_memory_stats = DeviceMemStats;

  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_extra_padding_size = ExtraPaddingSize;
}
