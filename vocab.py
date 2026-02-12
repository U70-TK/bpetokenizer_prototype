"""
Trusted boundary: loads cl100k_base via tiktoken and reconstructs the three
derived structures (merges, vocab, byte_shuffle) needed by core.py.

The only algorithmic work done here — BPE decomposition inside
recover_merges() — is delegated to core.decompose_token(), which is
covered by lemma_decompose_roundtrip.  What remains trusted is solely
the tiktoken data blob and the wiring that feeds it to the provable kernel.
"""

import tiktoken

from core import decompose_token


def recover_merges(mergeable_ranks):
    """
    tiktoken stores tokens as already-merged byte sequences.  This function
    recovers the original (left, right) -> merged pairings by running a small
    BPE decomposition on every multi-byte token.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # single-byte tokens are base vocabulary
        pair = tuple(decompose_token(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2, f"expected pair, got {len(pair)} parts for rank {rank}"
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges


def load_cl100k_base():
    """
    Load the cl100k_base encoding from tiktoken and return the three
    derived structures that core.py needs:

        merges      : dict[(int,int), int]   — BPE merge rules
        vocab       : dict[int, bytes]       — token-id → byte sequence
        byte_shuffle: dict[int, int]         — permutation on 0..255
    """
    enc = tiktoken.get_encoding("cl100k_base")
    mergeable_ranks = enc._mergeable_ranks

    # 1. recover merge rules
    merges = recover_merges(mergeable_ranks)

    # 2. build vocab deterministically from merges
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # 3. byte_shuffle: the historical byte-permutation quirk
    byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}

    return merges, vocab, byte_shuffle


def check_well_formed(merges, vocab, byte_shuffle):
    """
    Check the WellFormed invariants that the Lean 4 proof will assume.
    Returns True iff all invariants hold; raises AssertionError otherwise.

    Invariants:
      1. Base tokens: vocab contains exactly the 256 single-byte entries
         for the (shuffled) byte values.
      2. Merge decomposition: every merge token decomposes into exactly
         two shorter tokens already present in vocab.
      3. Injectivity: no two distinct token IDs map to the same byte sequence.
      4. byte_shuffle bijectivity: byte_shuffle is a bijection on {0..255}.
    """
    # --- invariant 1: base tokens ---
    for i in range(256):
        assert i in vocab, f"base token {i} missing from vocab"
        assert vocab[i] == bytes([i]), (
            f"base token {i} has wrong value: {vocab[i]!r}"
        )

    # --- invariant 2: merge decomposition ---
    for (p0, p1), idx in merges.items():
        assert p0 in vocab, f"merge left parent {p0} not in vocab"
        assert p1 in vocab, f"merge right parent {p1} not in vocab"
        assert idx in vocab, f"merge result {idx} not in vocab"
        assert vocab[idx] == vocab[p0] + vocab[p1], (
            f"merge {p0}+{p1}->{idx}: vocab[{idx}] != vocab[{p0}]+vocab[{p1}]"
        )
        assert len(vocab[p0]) < len(vocab[idx]), (
            f"left parent {p0} not shorter than merge result {idx}"
        )
        assert len(vocab[p1]) < len(vocab[idx]), (
            f"right parent {p1} not shorter than merge result {idx}"
        )

    # --- invariant 3: vocab injectivity ---
    seen = {}
    for idx, bs in vocab.items():
        if bs in seen:
            assert False, (
                f"vocab not injective: ids {seen[bs]} and {idx} both map to {bs!r}"
            )
        seen[bs] = idx

    # --- invariant 4: byte_shuffle bijectivity ---
    assert set(byte_shuffle.keys()) == set(range(256)), (
        "byte_shuffle domain is not {0..255}"
    )
    assert set(byte_shuffle.values()) == set(range(256)), (
        "byte_shuffle range is not {0..255} (not a bijection)"
    )

    return True
