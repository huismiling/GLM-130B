import pkg_resources
import torch
import torch_mlu
import ctypes

from typing import List

RESOURCE_PACKAGE_NAME = __name__


class Kernel:
    def __init__(self, filename: str, function_names: List[str]):
        filename = filename + ".so"
        # if not pkg_resources.resource_exists(RESOURCE_PACKAGE_NAME, filename):
        #     raise RuntimeError("File `%s` not found in `%s`" % (filename, RESOURCE_PACKAGE_NAME))
        self.filename = filename
        self.lib = ctypes.cdll.LoadLibrary(filename)
        # self.code = pkg_resources.resource_string(RESOURCE_PACKAGE_NAME, filename)
        self._function_names = function_names
        # self._cmodule = LazyKernelCModule(self.code)

        for name in self._function_names:
            setattr(self, name, getattr(self.lib, name))

import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
kernels = Kernel(
    f"{cur_dir}/quantization",
    [
        "int4WeightCompression",
        "int4WeightExtractionFloat",
        "int4WeightExtractionHalf",
        "int8WeightExtractionFloat",
        "int8WeightExtractionHalf",
    ],
)


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with torch.mlu.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device="mlu")
        m = m * 2
        stream = torch.mlu.current_stream()
        # import pdb; pdb.set_trace()

        kernels.int4WeightCompression(
            ctypes.c_void_p(weight.data_ptr()), 
            ctypes.c_void_p(out.data_ptr()), 
            ctypes.c_void_p(stream.mlu_stream), 
            ctypes.c_int32(n), 
            ctypes.c_int32(m)
        )
        return out


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
        # out = weight.half() * scale_list[: , ]
        # return out
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, "Unsupported bit-width"

    with torch.mlu.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device="mlu")
        m = out.size(1)
        stream = torch.mlu.current_stream()

        func(
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(stream.mlu_stream), 
                ctypes.c_int32(n),
                ctypes.c_int32(m)
        )
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    test_shapes = [[12288, 8192], [12288, 8192], [12288, 8192], [8192, 12288]]
    # test_shapes = [[12288, 8192]]

    quantization_bit_width = 8

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU
                ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=7),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_mlu")
    )as p:

        for ishape in test_shapes:
            weight = torch.randn(ishape).mlu().half()
            scale = (weight.abs().max(dim=-1).values / ((2 ** (quantization_bit_width - 1)) - 1)).half()
            print(weight.shape)
            print(scale.shape)
            quant_weight = torch.round(weight / scale[:, None]).to(torch.int8)

            # print(weight)
            # b = compress_int4_weight(weight)
            # print(b)

            a = extract_weight_to_half(quant_weight, scale, source_bit_width=quantization_bit_width)
            # print(a)
            with torch.profiler.record_function("python"):
                dequant_weight = quant_weight.half() * scale[:, None]
            print(dequant_weight.shape)
            diff = a - dequant_weight
            p.step()

            print("weight :", weight.reshape(-1)[:10])
            print("scale :", scale.reshape(-1)[:10])
            print("quant_weight :", quant_weight.reshape(-1)[:10])
            print("dequant_weight :", dequant_weight.reshape(-1)[:10])
            print("a :", a.reshape(-1)[:10])
            print("diff :", diff.abs().sum())
