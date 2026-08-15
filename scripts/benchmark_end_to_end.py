import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from framework.data import build_dataloader
from framework.models import build_model
from framework.utils.checkpoint import load_checkpoint


def main(args):
    device = torch.device(args.device)

    model = build_model(args.model)
    load_checkpoint(model, args.checkpoint)

    model.to(device)
    model.eval()

    loader = build_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_time = 0
    infer_time = 0
    save_time = 0

    total_start = time.perf_counter()

    prev = time.perf_counter()

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(loader)):

            now = time.perf_counter()
            load_time += now - prev

            images = batch["image"].to(device)

            torch.cuda.synchronize()

            t1 = time.perf_counter()

            outputs = model(images)

            torch.cuda.synchronize()

            t2 = time.perf_counter()

            infer_time += t2 - t1

            t3 = time.perf_counter()

            for i, img in enumerate(outputs):
                save_image(
                    img.cpu(),
                    out_dir / f"{batch_idx:05d}_{i}.png"
                )

            t4 = time.perf_counter()

            save_time += t4 - t3

            prev = time.perf_counter()

    total_time = time.perf_counter() - total_start

    print("\n========== END TO END BENCHMARK ==========")
    print(f"Images                : {len(loader.dataset)}")
    print(f"Batch Size            : {args.batch_size}")
    print(f"Data Loading          : {load_time:.3f} s")
    print(f"Model Inference       : {infer_time:.3f} s")
    print(f"Output Saving         : {save_time:.3f} s")
    print(f"Total Runtime         : {total_time:.3f} s")
    print(f"Average/Image         : {1000*total_time/len(loader.dataset):.2f} ms")
    print(f"Images/sec            : {len(loader.dataset)/total_time:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    main(args)