import io
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, concatenate_datasets, load_dataset
from tokenizers import Tokenizer

SHARD = 5000
OUT = Path("librispeech_460")
ALL_TEXT = Path("all_text.txt")  # project root
_ds = None
tok = Tokenizer.from_file("tokenizer.json")


def normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def write_text(path: Path, lines: list[str], trailing_newline: bool):
    # newline="\n" stops Windows from writing \r\n
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if trailing_newline else ""))


def get_ds():
    global _ds
    if _ds is None:
        parts = [
            load_dataset("openslr/librispeech_asr", "clean", split=s).cast_column(
                "audio", Audio(decode=False)
            )
            for s in ("train.100", "train.360")
        ]
        _ds = concatenate_datasets(parts)
    return _ds


def write_shard(k):
    d = OUT / f"shard_{k:04d}"
    if (d / "text.txt").exists():  # written last -> presence means shard is complete
        return
    d.mkdir(parents=True, exist_ok=True)

    ds = get_ds()
    sub = ds.select(range(k * SHARD, min((k + 1) * SHARD, len(ds))))
    arrs, texts = [], []
    for s in sub:
        a, _ = sf.read(io.BytesIO(s["audio"]["bytes"]), dtype="int16")
        arrs.append(a)
        texts.append(normalize(s["text"]))
    
    encodings = tok.encode_batch(texts)
    token_ids = [np.array(e.ids, dtype=np.int32) for e in encodings]
    np.save(d / "tokens.npy", np.concatenate(token_ids))
    np.save(d / "tok_lens.npy", np.array([len(t) for t in token_ids], dtype=np.int64))
    np.save(d / "audio.npy", np.concatenate(arrs))
    np.save(d / "lens.npy", np.array([len(a) for a in arrs], dtype=np.int64))
    write_text(d / "text.txt", texts, trailing_newline=False)  # last write


def build_all_text(num_shards):
    total = 0
    with open(ALL_TEXT, "w", encoding="utf-8", newline="\n") as out:
        for k in range(num_shards):
            d = OUT / f"shard_{k:04d}"
            raw = (d / "text.txt").read_text(encoding="utf-8")
            lines = [normalize(l) for l in raw.split("\n")]

            write_text(d / "text.txt", lines, trailing_newline=False)  # fixes stale text
            out.writelines(l + "\n" for l in lines)
            total += len(lines)
    return total


if __name__ == "__main__":
    n = len(get_ds())
    num_shards = (n + SHARD - 1) // SHARD
    OUT.mkdir(exist_ok=True)

    with Pool(4) as p:
        p.map(write_shard, range(num_shards))

    total = build_all_text(num_shards)
    print(f"{total} utterances, expected {n}")
    print("done:", ALL_TEXT)