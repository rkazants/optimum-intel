import argparse

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5Tokenizer, Qwen3_5TextConfig


def create_tiny_qwen3_5(save_directory):
    config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )

    with torch.no_grad():
        model = Qwen3_5ForCausalLM(config)
        model.eval()

    # Build a minimal byte-level BPE vocabulary matching the Qwen3_5 tokenizer format
    vocab = {"<|endoftext|>": 0}
    # Map all 256 bytes to unique unicode characters (ByteLevel pre-tokenizer convention)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    bytes_to_unicode = dict(zip(bs, [chr(c) for c in cs]))

    idx = 1
    for byte_val in range(256):
        unicode_char = bytes_to_unicode[byte_val]
        vocab[unicode_char] = idx
        idx += 1

    # Pad vocabulary up to config vocab_size with dummy tokens
    while len(vocab) < config.vocab_size:
        vocab[f"<dummy{len(vocab)}>"] = len(vocab)

    tokenizer = Qwen3_5Tokenizer(vocab=vocab, merges=[])

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Tiny Qwen3.5 model saved to {save_directory}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tiny Qwen3.5 model for testing")
    parser.add_argument("save_directory", type=str, help="Directory to save the tiny model")
    args = parser.parse_args()
    create_tiny_qwen3_5(args.save_directory)
