import torch

from kernels import extract_weight_to_half

use_mlu_ext = True
try:
    from mlu_ext.functions import cmm
except:
    use_mlu_ext = False

class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_shape = quant_w.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))

        if not use_mlu_ext:
            weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
            output = inp.mm(weight.t())
        else:
            a_max = inp.view(-1).abs().max().item() # *0.99
            a_quant_scale = 1024.0
            quant_A = torch.round((inp / a_max)*a_quant_scale).to(torch.int16)

            inp_shape = quant_A.shape
            output = cmm(quant_A.reshape(-1, inp_shape[-1]), quant_w.t(), a_quant_scale, 2**(weight_bit_width-1)-1)
            out_scale = scale_w.unsqueeze(0).mul((1.0) *a_max *(2**(weight_bit_width-1)-1))
            rank = torch.distributed.get_rank()
            output = output.mul_(out_scale).to(inp.dtype)

        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None
