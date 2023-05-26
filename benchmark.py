import torch
import torch_mlu
import time
from initialize import initialize, initialize_model_and_tokenizer

from functools import partial
from evaluation.model import batch_filling_sequence
from generate import get_masks_and_position_ids
from generation import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import update_mems

def test_decode(model, seq_len):
    end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
    strategy = BaseStrategy(
        batch_size=1, temperature=1.0, top_k=0, top_p=0.7, end_tokens=end_tokens
    )

    seqs = torch.ones(1, 1, device=torch.mlu.current_device(), dtype=torch.int64)

    context_lengths = torch.mlu.LongTensor([seqs.shape[-1]], device=args.device)
    max_memory_length=100000
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs, 0, seq_len-1, True)
    index = 0
    mems = None
    tokens = seqs
    for counter in range(seq_len):
        tokens = tokens.reshape(1, -1)
        mems = mems.reshape(mems.shape[0], 1, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        logits, *output_per_layers = model(
            tokens[..., index: counter+1],
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
        )
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        index = counter+1

if __name__ == "__main__":
    args = initialize(extra_args_provider=lambda parser: None)
    model, tokenizer = initialize_model_and_tokenizer(args)
    if torch.distributed.get_rank() == 0:
        print(model)

    repeate_times = 10
    for seq_len in [512, 1024, 2048]:
        torch.distributed.barrier()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _, *_ = model(
                    torch.ones(1, seq_len, device=torch.mlu.current_device(), dtype=torch.int64),
                    torch.arange(seq_len, device=torch.mlu.current_device(), dtype=torch.int64).view(1, -1),
                    torch.randn(1, 1, seq_len, seq_len, device=torch.mlu.current_device()) < 0.5,
                )
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"Encode {seq_len}: {(time.time() - start) * 1000 / repeate_times:.2f} ms")

        start = time.time()
        for _ in range(10):
            test_decode(model, seq_len)
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"Decode {seq_len}: {(time.time() - start) * 1000 / repeate_times:.2f} ms")
