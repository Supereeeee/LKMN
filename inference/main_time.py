import torch
import argparse
import time
from tqdm import tqdm
from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
from basicsr.archs.LKMN_arch import LKMN


def get_model(model_id=0, scale=4):
    if model_id == 1:
        model = LKMN(in_channels=3, channels=36, out_channels=3, upscale=4, num_block=8, large_kernel=31, split_group=4)
    else:
        assert False, "Model ID ERRO"
    return model


def main(args):
    clip = 100
    h, w = 2048, 2048
    model = get_model(args.model_id, args.scale)
    model = model.cpu().cuda()
    dummy_input = torch.randn(1, 3, h // args.scale, w // args.scale).cpu().cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    runtime = 0

    #  model.eval()
    with torch.no_grad():
        # print(model)
        for _ in tqdm(range(clip)):
            _ = model(dummy_input)

        for _ in tqdm(range(clip)):
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            runtime += start.elapsed_time(end)
        avg_time = runtime / clip
        max_memory = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

        print(model.__class__.__name__)
        print(f'{clip} Number Frames x{args.scale} SR Per Frame Time: {avg_time :.6f} ms')
        print(f' x{args.scale}SR FPS: {(1000 / avg_time):.6f} FPS')
        print(f' Max Memery {max_memory:.6f} [M]')
        # output = model(dummy_input)
        # print(output.shape)
        print(flop_count_table(FlopCountAnalysis(model, dummy_input),
                               activations=ActivationCountAnalysis(model, dummy_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flops")
    parser.add_argument("--model_id", default=1, type=int)
    parser.add_argument("--scale", default=4, type=int)
    args = parser.parse_args()
    main(args)

