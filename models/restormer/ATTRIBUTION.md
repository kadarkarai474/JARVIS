# Restormer — Attribution & Important Note

**This is NOT the official repository's code.** Same situation as
`models/nafnet/ATTRIBUTION.md`: this is a clean-room reimplementation of
the architecture described in the published paper:

> Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., & Yang, M.H.
> (2022). *Restormer: Efficient Transformer for High-Resolution Image
> Restoration.* CVPR 2022.
> Official repo: https://github.com/swz30/Restormer (Apache 2.0 License)

## Why this isn't the vendored official code

Same two hard constraints as NAFNet: this sandbox has no network egress
(`git clone https://github.com/swz30/Restormer` was blocked by the network
allowlist), and Anthropic's copyright policy prohibits reproducing source
code verbatim from web search results into project files.

## What was reimplemented, from the paper's public description

- **MDTA (Multi-Dconv Head Transposed Attention):** self-attention computed
  across the *channel* dimension rather than the spatial dimension (an
  attention map over channels-per-head, not over pixels), making cost
  linear rather than quadratic in image resolution — the paper's key
  contribution for making transformers tractable on high-resolution images.
- **GDFN (Gated-Dconv Feed-Forward Network):** a feed-forward block with a
  GELU-gated parallel path, using depth-wise convolutions to inject local
  context.
- **4-level encoder-decoder** of Transformer blocks, `PixelUnshuffle`/
  `PixelShuffle` for down/upsampling (avoids checkerboard artifacts vs.
  strided/transposed convolutions), skip connections via channel-concat +
  1x1 reduction, and a final refinement stage at full resolution.

Default `dim`/`num_blocks`/`heads` here are set smaller than the paper's
originals, to fit this project's 4GB RTX3050 dev constraint — this is a
capacity choice, not an architectural deviation.

## Before treating this as your final submission

```bash
git clone https://github.com/swz30/Restormer third_party/Restormer
pip install -r third_party/Restormer/requirements.txt
```
Then wrap the official `Restormer` class instead of this reimplementation.

## Citation

```bibtex
@inproceedings{zamir2022restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
  booktitle={CVPR},
  year={2022}
}
```

## License note

The official Restormer repository is released under the Apache 2.0
License. No code was copied from the official repository into this
reimplementation.
