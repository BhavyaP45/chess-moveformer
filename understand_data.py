import numpy as np

def compute_block_size(txt_path: str, percentile: float = 95.0):
    with open(txt_path, "r") as f:
        lengths = [len(line.rstrip("\n")) for line in f if line.strip()]

    lengths = np.array(lengths)

    raw = int(np.percentile(lengths, percentile))
    recommended = 1
    while recommended < raw:
        recommended *= 2

    print(f"Games         : {len(lengths):,}")
    print(f"p50 / p90 / p95 / p99 : {np.percentile(lengths,np.array([50,90,95,99]).astype(int))}")
    print(f"Raw p{percentile:.0f}        : {raw}")
    print(f"Recommended   : {recommended}  (next power of 2)")
    print(f"Coverage      : {np.mean(lengths <= recommended)*100:.1f}% of games fit fully")


compute_block_size("data/train.txt")

with open('data/train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars, vocab_size)