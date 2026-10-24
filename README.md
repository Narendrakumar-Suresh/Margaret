# Margaret

A small, open-source, offline English speech-to-text model that runs on your own device. No cloud, no account, no audio leaving your machine.

Named after [Margaret Hamilton](https://en.wikipedia.org/wiki/Margaret_Hamilton_(software_engineer)), who led the MIT team that wrote the onboard flight software for the Apollo missions and helped establish software engineering as a discipline. Her work made the moon landings possible, and it's a good standard to aim for: software that has to work, with no second chances.

> **Status: early development.** The data pipeline and model code are in place. Training has not been run yet, so there are no accuracy numbers.

---

## Goals

- **Small:** under ~100M parameters.
- **Private:** everything runs locally; audio never goes to a server.
- **Offline, not streaming:** whole audio in, full text out.
- **Runs on CPU and phones:** inference on a laptop CPU first, then smartphones, with custom kernels where they help.
- **English only.**
- **Simple, with no artificial limits** on how you use it.

## How it works

Margaret is a Whisper-style encoder-decoder transformer written in JAX and Flax NNX.

```
raw waveform (16 kHz)
      |
 conv frontend (downsamples time)
      |
 Encoder blocks x N     bidirectional self-attention + MLP
      |
      +---------------------------+
                                  | cross-attention
 text tokens -> embedding         |
      |                           |
 Decoder blocks x M     causal self-attention -> cross-attention -> MLP
      |
 LayerNorm -> Linear -> logits over vocabulary
```

- **Input:** raw audio samples, not a mel spectrogram. A strided-convolution frontend shortens the sequence before attention.
- **Blocks:** pre-norm with residual connections.
- **Attention:** RoPE positions in self-attention and grouped-query attention (fewer key/value heads than query heads).
- **Loss:** next-token cross-entropy with teacher forcing (not CTC).
- **Tokenizer:** BPE, vocabulary of 1024, trained on lowercase LibriSpeech transcripts.
- **Optimizer:** Muon (settings still being tuned).

## Data

Training uses LibriSpeech `train.100` + `train.360` (about 460 hours), converted into compact numpy shards so they upload to Kaggle easily and load quickly on TPU hosts.

```
librispeech_460/
├── shard_0000/
│   ├── audio.npy    all utterances in the shard, concatenated, int16
│   ├── lens.npy     number of samples in each utterance
│   └── text.txt     one lowercase transcript per line, same order as audio
├── shard_0001/
│   └── ...
└── shard_0026/
all_text.txt         every transcript, one per line, for tokenizer training
```

- 5000 utterances per shard.
- Audio is stored as `int16`, which is lossless for LibriSpeech and half the size of float32.
- Each shard's audio is one flat array. Utterance `i` is `audio[offsets[i]:offsets[i+1]]`, where `offsets = concatenate([[0], cumsum(lens)])`.
- Transcripts are lowercased with whitespace collapsed. Only `a-z`, the apostrophe and spaces remain.

### Build the dataset

```bash
python data/cleaner.py
```

This downloads LibriSpeech through Hugging Face `datasets`, decodes the FLAC audio, writes the shards in parallel, and builds `all_text.txt`. A finished shard is skipped on re-runs, so an interrupted run can be resumed.

### Read a shard

```python
import numpy as np

audio = np.load("librispeech_460/shard_0000/audio.npy", mmap_mode="r")
lens = np.load("librispeech_460/shard_0000/lens.npy")
text = open("librispeech_460/shard_0000/text.txt", encoding="utf-8").read().split("\n")

offsets = np.concatenate([[0], np.cumsum(lens)])
i = 3
wav = audio[offsets[i]:offsets[i + 1]].astype(np.float32) / 32768.0
print(text[i], wav.shape)
```

### Normalization

Text is lowercased everywhere it is used: tokenizer training, training labels, and inference. Use the same function in all three places.

```python
import re

def normalize(s: str) -> str:
    s = re.sub(r"[^a-z' ]", " ", s.lower())
    return " ".join(s.split())
```

## Tokenizer

A BPE tokenizer with 1024 tokens, trained on `all_text.txt`, saved as `tokenizer.json`.

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode(normalize("hello world")).ids
print(tok.decode(ids))
```

## Training

Training targets a Kaggle TPU v5e-8. Each shard is a few GB, so you can upload the shards as a Kaggle dataset and stream them with `mmap`.

Batches are padded to a fixed length, with masks passed to attention and the loss so padding never affects the math. Grouping utterances of similar length into buckets keeps the number of distinct shapes small, so XLA compiles only a few times.

## Roadmap

- [x] Dataset conversion to int16 shards
- [x] Text normalization and tokenizer
- [x] Encoder-decoder model in Flax NNX
- [ ] Real raw-audio frontend (~320x downsampling) to replace the test stub
- [ ] BOS, EOS and pad tokens in the tokenizer
- [ ] Data loader with length bucketing and masks
- [ ] Training loop, checkpointing, learning-rate schedule
- [ ] Evaluation (word error rate on LibriSpeech dev/test)
- [ ] CPU inference path and custom kernels
- [ ] Smartphone deployment

## Requirements

- Python 3.10+
- `jax`, `flax`, `optax`
- `datasets`, `soundfile`, `numpy`
- `tokenizers`

## License

To be decided.