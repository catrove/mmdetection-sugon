#include <torch/torch.h>

#include <vector>

std::vector<torch::Tensor> pool_forward(
    torch::Tensor input
) {
    // Initialize output
    torch::Tensor output = torch::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    // Copy the last column
    torch::Tensor input_temp  = input.select(2, 0);
    torch::Tensor output_temp = output.select(2, 0);
    output_temp.copy_(input_temp);

    torch::Tensor max_temp;
    for (int64_t ind = 0; ind < height - 1; ++ind) {
        input_temp  = input.select(2, ind + 1);
        output_temp = output.select(2, ind);
        max_temp    = output.select(2, ind + 1);

        torch::max_out(max_temp, input_temp, output_temp);
    }

    return { 
        output
    };
}

std::vector<torch::Tensor> pool_backward(
    torch::Tensor input,
    torch::Tensor grad_output
) {
    auto output = torch::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    auto byte_options =
      torch::TensorOptions()
        .dtype(torch::kBool)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    auto long_options =
      torch::TensorOptions()
        .dtype(torch::kInt64)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    auto float_options =
      torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    auto max_val = torch::zeros({batch, channel, width},float_options);
    auto max_ind = torch::zeros({batch, channel, width},long_options);

    auto input_temp = input.select(2, 0);
    max_val.copy_(input_temp);

    max_ind.fill_(0);

    auto output_temp      = output.select(2, 0);
    auto grad_output_temp = grad_output.select(2, 0);
    output_temp.copy_(grad_output_temp);

    auto un_max_ind = max_ind.unsqueeze(2);
    auto gt_mask    = torch::zeros({batch, channel, width},byte_options);
    auto max_temp   = torch::zeros({batch, channel, width},float_options);
    for (int32_t ind = 0; ind < height - 1; ++ind) {
        input_temp = input.select(2, ind + 1);
        torch::gt_out(gt_mask, input_temp, max_val);

        torch::masked_select_out(max_temp, input_temp, gt_mask);
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, ind + 1);

        grad_output_temp = grad_output.select(2, ind + 1).unsqueeze(2);
        output.scatter_add_(2, un_max_ind, grad_output_temp);
    }

    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &pool_forward, "Bottom Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &pool_backward, "Bottom Pool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
