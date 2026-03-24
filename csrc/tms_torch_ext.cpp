#include <torch/extension.h>
#include <cstdint>

torch::Tensor wrap_ptr_as_tensor(int64_t ptr, int64_t numel, torch::Dtype dtype, int64_t device_index) {
  auto options = torch::TensorOptions()
      .device(torch::Device(torch::kCUDA, device_index))
      .dtype(dtype);
  return torch::from_blob(
      reinterpret_cast<void*>(ptr),
      {numel},
      [](void* /*p*/) {},
      options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap_ptr_as_tensor", &wrap_ptr_as_tensor,
        py::arg("ptr"),
        py::arg("numel"),
        py::arg("dtype"),
        py::arg("device_index") = 0);
}
