"""
Tests for the proof-carrying BPE tokenizer (ASCII scope).

All test strings are ASCII-only.  All encode/decode paths go through
core.py (the provable kernel).  No regex, no boundary.py.

Phases:
  1. WellFormed invariants
  2. Lemma 3 — decompose_token roundtrip
  3. Byte-level roundtrip — core.roundtrip_check
  4. ASCII pre-tokenizer partition — Lemma 4
  5. Full ASCII string-level roundtrip — core.full_roundtrip_check (PROVABLE)
  6. Cross-check core.encode against tiktoken
"""

import tiktoken

from vocab import load_cl100k_base, check_well_formed
from core import (
    encode_chunk, decode_chunk, roundtrip_check,
    pre_tokenize_ascii, encode, decode, full_roundtrip_check,
    check_well_formed as core_check_well_formed,
    lemma_pre_tokenize_partition,
    lemma_decompose_roundtrip,
    lemma_ascii_utf8_roundtrip,
)


# ASCII-only test strings
TEST_STRINGS = [
    "",
    "hello world",
    "Hello, World! 123",
    "   \t\n\r\n  ",
    "def foo(x):\n    return x + 1\n",
    "int main() { return 0; }",
    "3.14159265358979",
    "a" * 1000,
    "The quick brown fox jumps over the lazy dog.",
    " leading and trailing spaces ",
    "\n\n\n",
    "mixed\tcontent\nwith\rvarious\r\nline endings",
    "<html><body>hello</body></html>",
    '{"key": "value", "num": 42}',
    "x" * 5 + " " + "y" * 5,
    "It's a test",
    "I'll say they've gone",
    "don't won't can't",
    "  trailing whitespace   ",
    "HELLO123world",
    "line1\nline2\nline3",
    "a!b@c#d$e%f",
    "100 200 300",
    "12345678901234567890",
    "'s 'd 'm 't 'll 've 're",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "   ",
    "\t\t\t",
    "a",
    " ",
    "\n",
    "Hello\r\nWorld\r\n",
    "func(arg1, arg2, arg3)",
    "if (x > 0) { return true; }",
    "SELECT * FROM users WHERE id = 1;",
    "https://example.com/path?q=hello&lang=en",
    "user@email.com",
    "2024-01-15T10:30:00Z",
    "the the the the the",
    "  \n  \n  \n  ",
    "a b c d e f g h i j k l m n o p",
]


def main():
    all_pass = True

    # ------------------------------------------------------------------
    # Load structures
    # ------------------------------------------------------------------
    print("Loading cl100k_base via vocab.py ...")
    merges, vocab, byte_shuffle = load_cl100k_base()
    inverse_byte_shuffle = {v: k for k, v in byte_shuffle.items()}
    print(f"  merges : {len(merges)} rules")
    print(f"  vocab  : {len(vocab)} entries")
    print()

    # ------------------------------------------------------------------
    # Phase 1: WellFormed invariants
    # ------------------------------------------------------------------
    print("Phase 1: WellFormed invariants")
    try:
        check_well_formed(merges, vocab, byte_shuffle)
        print("  PASS: vocab.check_well_formed")
    except AssertionError as e:
        all_pass = False
        print(f"  FAIL: vocab.check_well_formed -- {e}")

    try:
        core_check_well_formed(merges, vocab, byte_shuffle)
        print("  PASS: core.check_well_formed")
    except AssertionError as e:
        all_pass = False
        print(f"  FAIL: core.check_well_formed -- {e}")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Lemma 3 — decompose_token roundtrip
    # ------------------------------------------------------------------
    print("Phase 2: Lemma 3 (decompose roundtrip on sample tokens)")
    enc = tiktoken.get_encoding("cl100k_base")
    mergeable_ranks = enc._mergeable_ranks

    sample_tokens = [
        token
        for token, _rank in sorted(mergeable_ranks.items(), key=lambda x: x[1])
        if len(token) > 1
    ]
    mid = len(sample_tokens) // 2
    samples = sample_tokens[:5] + sample_tokens[mid:mid + 5]

    for token in samples:
        label = repr(token) if len(token) <= 30 else repr(token[:27] + b"...")
        try:
            lemma_decompose_roundtrip(mergeable_ranks, token)
            print(f"  PASS: {label}")
        except AssertionError as e:
            all_pass = False
            print(f"  FAIL: {label} -- {e}")

    for token in samples:
        rank = mergeable_ranks[token]
        label = f"max_rank={rank} {repr(token)[:30]}"
        try:
            lemma_decompose_roundtrip(mergeable_ranks, token, max_rank=rank)
            print(f"  PASS: {label}")
        except AssertionError as e:
            all_pass = False
            print(f"  FAIL: {label} -- {e}")
    print()

    # ------------------------------------------------------------------
    # Phase 3: Byte-level roundtrip
    # ------------------------------------------------------------------
    print("Phase 3: Byte-level roundtrip (core.roundtrip_check)")
    for s in TEST_STRINGS:
        label = repr(s) if len(s) <= 50 else repr(s[:47] + "...")
        try:
            roundtrip_check(
                merges, vocab, byte_shuffle, inverse_byte_shuffle,
                s.encode("utf-8"),
            )
            print(f"  PASS: {label}")
        except AssertionError as e:
            all_pass = False
            print(f"  FAIL: {label} -- {e}")
    print()

    # ------------------------------------------------------------------
    # Phase 4: ASCII pre-tokenizer partition (Lemma 4)
    # ------------------------------------------------------------------
    print("Phase 4: ASCII pre-tokenizer partition (Lemma 4)")
    for s in TEST_STRINGS:
        label = repr(s) if len(s) <= 50 else repr(s[:47] + "...")
        try:
            chunks = lemma_pre_tokenize_partition(s.encode("utf-8"))
            print(f"  PASS: {label} -> {len(chunks)} chunks")
        except AssertionError as e:
            all_pass = False
            print(f"  FAIL: {label} -- {e}")
    print()

    # ------------------------------------------------------------------
    # Phase 5: Full ASCII string-level roundtrip (PROVABLE)
    # ------------------------------------------------------------------
    print("Phase 5: Full ASCII string-level roundtrip (PROVABLE)")
    for s in TEST_STRINGS:
        label = repr(s) if len(s) <= 50 else repr(s[:47] + "...")
        try:
            lemma_ascii_utf8_roundtrip(s)
            result = full_roundtrip_check(
                merges, vocab, byte_shuffle, inverse_byte_shuffle, s
            )
            assert result == s
            print(f"  PASS: {label}")
        except AssertionError as e:
            all_pass = False
            print(f"  FAIL: {label} -- {e}")
    print()

    # ------------------------------------------------------------------
    # Phase 6: Cross-check core.encode against tiktoken
    # ------------------------------------------------------------------
    print("Phase 6: Cross-check core.encode vs tiktoken")
    for s in TEST_STRINGS:
        if s == "":
            continue
        our_ids = encode(merges, vocab, byte_shuffle, s)
        tik_ids = enc.encode(s)
        label = repr(s) if len(s) <= 50 else repr(s[:47] + "...")
        if our_ids != tik_ids:
            all_pass = False
            print(f"  FAIL: {label}")
            print(f"         core    = {our_ids[:10]}{'...' if len(our_ids)>10 else ''}")
            print(f"         tiktoken= {tik_ids[:10]}{'...' if len(tik_ids)>10 else ''}")
        else:
            print(f"  PASS: {label}")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")


if __name__ == "__main__":
    main()
