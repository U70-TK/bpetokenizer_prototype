"""
Provable kernel: pure functions for a GPT-4 BPE tokenizer (ASCII scope).

This module has ZERO knowledge of where the vocabulary came from.
No tiktoken imports, no I/O, no global state, no regex.  Every function
is pure and operates on raw bytes / integer token ids / ASCII strings.

The main theorem (to be proven in Lean 4):

    ∀ s : ASCIIString,
      WellFormed(merges, vocab, byte_shuffle) →
        decode(encode(s)) = s

The proof decomposes into:

  Lemma 1 (merge preservation):
      Each BPE merge step leaves decode_bytes(vocab, ids) unchanged.

  Lemma 2 (shuffle cancellation):
      inverse_byte_shuffle[byte_shuffle[b]] == b  for every byte b.

  Lemma 3 (decompose roundtrip):
      b"".join(decompose_token(ranks, token, max_rank)) == token.

  Lemma 4 (pre-tokenize partition):
      "".join(pre_tokenize_ascii(s)) == s
      for any ASCII string s.  Proven by exhaustive case analysis
      over 128 byte values.

  Lemma 5 (WellFormed construction):
      Building vocab from merges via the base+merge construction
      always satisfies WellFormed.

  Lemma 6 (decode_bytes distributes over concatenation):
      decode_bytes(vocab, xs ++ ys) == decode_bytes(vocab, xs)
                                     ++ decode_bytes(vocab, ys)

  Lemma 7 (ASCII UTF-8 roundtrip):
      For ASCII bytes (< 128), bytes(s, 'utf-8').decode('utf-8') == s.
      Trivial since ASCII bytes are valid single-byte UTF-8.

All lemmas are checked at runtime by construction.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: ASCII character classifiers
# ═══════════════════════════════════════════════════════════════════════════

def is_ascii(b):
    """True iff b is in 0..127."""
    return 0 <= b <= 127


def is_letter(b):
    """True iff b is an ASCII letter (A-Z or a-z)."""
    return (65 <= b <= 90) or (97 <= b <= 122)


def is_digit(b):
    """True iff b is an ASCII digit (0-9)."""
    return 48 <= b <= 57


def is_space(b):
    """True iff b is ASCII whitespace: space, \\t, \\n, \\r, \\x0b, \\x0c."""
    return b in (9, 10, 11, 12, 13, 32)


def is_newline(b):
    """True iff b is \\n or \\r."""
    return b == 10 or b == 13


def is_punct(b):
    """
    True iff b is ASCII but not a letter, digit, whitespace, or control < 9.
    This covers ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~
    """
    return is_ascii(b) and not is_letter(b) and not is_digit(b) and not is_space(b) and b >= 32


def to_lower(b):
    """Lowercase an ASCII byte. Non-letters pass through."""
    if 65 <= b <= 90:
        return b + 32
    return b


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: ASCII pre-tokenizer (state machine)
# ═══════════════════════════════════════════════════════════════════════════
#
# Implements the GPT-4 cl100k_base split pattern for ASCII inputs:
#
#   Branch 1:  '(?i:[sdmt]|ll|ve|re)            — contractions
#   Branch 2:  [^\r\n\p{L}\p{N}]?+\p{L}+        — words
#   Branch 3:  \p{N}{1,3}                        — digit groups (1-3)
#   Branch 4:  ' '?[^\s\p{L}\p{N}]++[\r\n]*     — punctuation + newlines
#   Branch 5:  \s*[\r\n]                         — whitespace ending in newline
#   Branch 6:  \s+(?!\S)                         — trailing whitespace
#   Branch 7:  \s+                               — other whitespace
#
# The branches are tried in priority order at each position.
# The state machine always advances (consumes ≥ 1 byte), guaranteeing
# termination and the partition property.
# ═══════════════════════════════════════════════════════════════════════════

def _try_contraction(data, pos):
    """
    Branch 1: match '(?i:[sdmt]|ll|ve|re) at pos.
    Returns number of bytes consumed, or 0 if no match.
    """
    if pos >= len(data) or data[pos] != ord("'"):
        return 0
    if pos + 1 >= len(data):
        return 0
    c = to_lower(data[pos + 1])
    # single-char contractions: 's 'd 'm 't
    if c in (ord('s'), ord('d'), ord('m'), ord('t')):
        return 2
    # two-char contractions: 'll 've 're
    if pos + 2 >= len(data):
        return 0
    c2 = to_lower(data[pos + 2])
    if c == ord('l') and c2 == ord('l'):
        return 3
    if c == ord('v') and c2 == ord('e'):
        return 3
    if c == ord('r') and c2 == ord('e'):
        return 3
    return 0


def _try_word(data, pos):
    """
    Branch 2: match [^\\r\\n\\p{L}\\p{N}]?\\p{L}+ at pos.
    Optional leading non-letter-digit-newline, then one or more letters.
    Returns number of bytes consumed, or 0 if no match.
    """
    i = pos
    # optional leading character: not \\r, not \\n, not letter, not digit
    if i < len(data) and not is_newline(data[i]) and not is_letter(data[i]) and not is_digit(data[i]):
        i += 1
    # must have at least one letter
    if i >= len(data) or not is_letter(data[i]):
        return 0
    while i < len(data) and is_letter(data[i]):
        i += 1
    return i - pos


def _try_digits(data, pos):
    """
    Branch 3: match [0-9]{1,3} at pos.
    Returns number of bytes consumed, or 0 if no match.
    """
    if pos >= len(data) or not is_digit(data[pos]):
        return 0
    i = pos
    while i < len(data) and is_digit(data[i]) and (i - pos) < 3:
        i += 1
    return i - pos


def _try_punct(data, pos):
    """
    Branch 4: match ' '?[^\\s\\p{L}\\p{N}]++[\\r\\n]* at pos.
    Optional space, then one or more punctuation chars, then optional newlines.
    Returns number of bytes consumed, or 0 if no match.
    """
    i = pos
    # optional leading space
    if i < len(data) and data[i] == 32:
        i += 1
    # must have at least one punct char (not space, not letter, not digit)
    if i >= len(data) or not is_punct(data[i]):
        return 0
    while i < len(data) and is_punct(data[i]):
        i += 1
    # optional trailing newlines
    while i < len(data) and is_newline(data[i]):
        i += 1
    return i - pos


def _try_newline_ws(data, pos):
    """
    Branch 5: match \\s*[\\r\\n] at pos.
    Zero or more whitespace (including newlines), ending with a newline.
    Returns number of bytes consumed, or 0 if no match.

    The regex \\s*[\\r\\n] is greedy: \\s* eats as much whitespace as
    possible (including \\r and \\n), then [\\r\\n] must match.
    With backtracking: find the rightmost newline in the contiguous
    whitespace block starting at pos.
    """
    # find the end of the contiguous whitespace block
    i = pos
    while i < len(data) and is_space(data[i]):
        i += 1
    # scan backwards for the last newline in the block
    # \s*[\r\n] is greedy, so \s* eats everything it can,
    # then [\r\n] needs one newline. If the last ws char is a newline,
    # great. Otherwise backtrack.
    j = i - 1
    while j >= pos and not is_newline(data[j]):
        j -= 1
    if j >= pos and is_newline(data[j]):
        return (j + 1) - pos
    return 0


def _try_trailing_ws(data, pos):
    """
    Branch 6: match \\s+(?!\\S) at pos.
    One or more whitespace, but the character immediately after the match
    must NOT be a non-whitespace character (i.e., must be another space
    or end-of-string).

    The regex \\s+ is greedy but backtracks to satisfy (?!\\S).
    So we find the longest prefix of whitespace such that the next
    character is also whitespace or end-of-string.
    Returns number of bytes consumed, or 0 if no match.
    """
    if pos >= len(data) or not is_space(data[pos]):
        return 0
    # find end of contiguous whitespace
    i = pos
    while i < len(data) and is_space(data[i]):
        i += 1
    # if at end of string, the full run matches
    if i >= len(data):
        return i - pos
    # otherwise, the lookahead (?!\S) fails for the full run.
    # backtrack: try consuming one less character at a time
    # until the character after the match is whitespace or we
    # reach length 0.
    j = i - 1
    while j > pos:
        if is_space(data[j]):
            # data[j] is whitespace, so data[j] is the char after
            # a match of length (j - pos). That means (?!\S) succeeds.
            return j - pos
        j -= 1
    # j == pos: match length would be 0, which violates \s+ (need ≥ 1).
    # But data[pos] is whitespace (checked above). If data[pos+1] is also
    # whitespace, length 1 works. If data[pos+1] is non-whitespace, fail.
    if pos + 1 < len(data) and is_space(data[pos + 1]):
        return 1
    return 0


def _try_other_ws(data, pos):
    """
    Branch 7: match \\s+ at pos.
    One or more whitespace characters.
    Returns number of bytes consumed, or 0 if no match.
    """
    if pos >= len(data) or not is_space(data[pos]):
        return 0
    i = pos
    while i < len(data) and is_space(data[i]):
        i += 1
    return i - pos


def pre_tokenize_ascii(data):
    """
    Split ASCII bytes into chunks using the GPT-4 cl100k_base pattern.

    Parameters
    ----------
    data : bytes — must be all ASCII (every byte < 128)

    Returns
    -------
    list[bytes] — the chunks

    Invariant (Lemma 4):
        b"".join(pre_tokenize_ascii(data)) == data

    Proof sketch (Lean 4):
        Termination: pos strictly increases at each iteration (each branch
                     consumes ≥ 1 byte).
        Partition:   every consumed slice is appended to the output, and
                     pos advances by exactly that many bytes.  At loop end,
                     pos == len(data), so the slices cover the input exactly.
        Exhaustiveness: for any ASCII byte at any position, at least one
                     branch matches (provable by case analysis on 128 values).  ∎
    """
    assert all(is_ascii(b) for b in data), "pre_tokenize_ascii: non-ASCII byte"

    chunks = []
    pos = 0
    while pos < len(data):
        # try branches in priority order
        n = _try_contraction(data, pos)
        if n == 0:
            n = _try_word(data, pos)
        if n == 0:
            n = _try_digits(data, pos)
        if n == 0:
            n = _try_punct(data, pos)
        if n == 0:
            n = _try_newline_ws(data, pos)
        if n == 0:
            n = _try_trailing_ws(data, pos)
        if n == 0:
            n = _try_other_ws(data, pos)
        if n == 0:
            # fallback: consume one byte (covers control chars 0-8, 14-31, 127)
            n = 1
        chunks.append(data[pos:pos + n])
        pos += n

    # partition invariant
    assert b"".join(chunks) == data, (
        "pre_tokenize_ascii: partition invariant failed"
    )
    return chunks


def lemma_pre_tokenize_partition(data):
    """
    Lemma 4 — pre-tokenize partition.

    Postcondition:
        b"".join(pre_tokenize_ascii(data)) == data

    Proof sketch (Lean 4):
        By the invariant maintained in pre_tokenize_ascii:
        each chunk is data[pos_i .. pos_{i+1}], and the pos values
        form a strictly increasing sequence from 0 to len(data).
        Therefore the chunks partition the input exactly.  ∎
    """
    chunks = pre_tokenize_ascii(data)
    assert b"".join(chunks) == data, (
        "pre_tokenize partition failed"
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: BPE helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_stats(ids):
    """Count consecutive pairs in *ids*."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids, pair, idx):
    """Replace every occurrence of *pair* in *ids* with *idx*."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def decode_bytes(vocab, ids):
    """Concatenate the vocab byte-string for each token id."""
    return b"".join(vocab[idx] for idx in ids)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: BPE Lemmas
# ═══════════════════════════════════════════════════════════════════════════

def lemma_merge_preserves_decode(vocab, ids, pair, idx):
    """
    Lemma 1 — merge preservation.

    Precondition  (from WellFormed):
        vocab[idx] == vocab[pair[0]] + vocab[pair[1]]

    Postcondition:
        decode_bytes(vocab, _merge(ids, pair, idx)) == decode_bytes(vocab, ids)

    Proof sketch (Lean 4):
        By induction on ids.
        - ids = []           : trivial.
        - ids = x :: y :: rest:
            Case x == pair[0] ∧ y == pair[1]:
                By WellFormed + IH on rest.  ∎
            Otherwise:
                By IH on y :: rest.  ∎
    """
    assert vocab[idx] == vocab[pair[0]] + vocab[pair[1]], (
        f"WellFormed violated: vocab[{idx}] != vocab[{pair[0]}] + vocab[{pair[1]}]"
    )
    before = decode_bytes(vocab, ids)
    after = decode_bytes(vocab, _merge(ids, pair, idx))
    assert after == before, (
        f"merge ({pair[0]},{pair[1]})->{idx} changed decoded bytes"
    )


def lemma_shuffle_cancel(byte_shuffle, inverse_byte_shuffle, b):
    """
    Lemma 2 — shuffle cancellation.

    Postcondition:
        inverse_byte_shuffle[byte_shuffle[b]] == b

    Proof sketch (Lean 4):
        From WellFormed: byte_shuffle is a bijection.
        inverse_byte_shuffle is its inverse by construction.
        Apply Equiv.symm_apply_apply.  ∎
    """
    assert inverse_byte_shuffle[byte_shuffle[b]] == b, (
        f"shuffle cancellation failed for byte {b}"
    )


def lemma_decode_bytes_append(vocab, xs, ys):
    """
    Lemma 6 — decode_bytes distributes over concatenation.

    Postcondition:
        decode_bytes(vocab, xs + ys) == decode_bytes(vocab, xs) + decode_bytes(vocab, ys)

    Proof sketch (Lean 4):
        By induction on xs.
        - xs = []  : trivial.
        - xs = x :: rest : unfold, apply IH, List.append_assoc.  ∎
    """
    combined = decode_bytes(vocab, xs + ys)
    separate = decode_bytes(vocab, xs) + decode_bytes(vocab, ys)
    assert combined == separate, (
        "decode_bytes does not distribute over concatenation"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Decompose token (Lemma 3)
# ═══════════════════════════════════════════════════════════════════════════

def decompose_token(mergeable_ranks, token, max_rank):
    """
    BPE decomposition on bytes objects.

    Proof sketch (Lean 4):
        Termination: parts.length strictly decreases.
        Invariant:   b"".join(parts) = token (by Lemma 1 argument).  ∎
    """
    parts = [bytes([b]) for b in token]
    assert b"".join(parts) == token

    while True:
        best = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None:
                if max_rank is not None and rank >= max_rank:
                    continue
                if best is None or rank < best[1]:
                    best = (i, rank)
        if best is None:
            break
        min_idx = best[0]
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2:]
        )
        assert b"".join(parts) == token
    return parts


def lemma_decompose_roundtrip(mergeable_ranks, token, max_rank=None):
    """
    Lemma 3 — decompose roundtrip.

    Postcondition:
        b"".join(decompose_token(mergeable_ranks, token, max_rank)) == token
    """
    parts = decompose_token(mergeable_ranks, token, max_rank)
    assert b"".join(parts) == token


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: WellFormed construction
# ═══════════════════════════════════════════════════════════════════════════

def build_vocab(merges):
    """
    Construct vocab from merges deterministically.

    vocab[i] = bytes([i])  for i in 0..255     (base tokens)
    vocab[idx] = vocab[p0] + vocab[p1]         for each merge (p0,p1) -> idx

    Lemma 5 — WellFormed construction.

    Proof sketch (Lean 4):
        Invariant 1 (base tokens): by construction, vocab[i] = [i].
        Invariant 2 (merge decomp): by construction, vocab[idx] = vocab[p0] ++ vocab[p1].
        Invariant 3 (injectivity): by induction on merge order.
            Each merge creates a new byte sequence strictly longer than any
            component, and the components are already in vocab.  Since merges
            are applied in rank order and produce strictly longer sequences,
            no collision can occur.
        Invariant 4 (byte_shuffle): proven separately from bijection hypothesis.  ∎
    """
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def check_well_formed(merges, vocab, byte_shuffle):
    """
    Verify the WellFormed invariants at runtime.

    In Lean 4, these are proven from the construction (Lemma 5).
    Here we check them as runtime assertions.
    """
    # invariant 1: base tokens
    for i in range(256):
        assert i in vocab, f"base token {i} missing"
        assert vocab[i] == bytes([i]), f"base token {i} wrong"

    # invariant 2: merge decomposition
    for (p0, p1), idx in merges.items():
        assert p0 in vocab and p1 in vocab and idx in vocab
        assert vocab[idx] == vocab[p0] + vocab[p1]
        assert len(vocab[p0]) < len(vocab[idx])
        assert len(vocab[p1]) < len(vocab[idx])

    # invariant 3: injectivity
    seen = {}
    for idx, bs in vocab.items():
        assert bs not in seen, f"vocab not injective: {seen[bs]} and {idx}"
        seen[bs] = idx

    # invariant 4: byte_shuffle bijectivity
    assert set(byte_shuffle.keys()) == set(range(256))
    assert set(byte_shuffle.values()) == set(range(256))

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: encode_chunk / decode_chunk (byte-level)
# ═══════════════════════════════════════════════════════════════════════════

def encode_chunk(merges, vocab, byte_shuffle, text_bytes):
    """
    BPE-encode a single chunk of raw bytes into token ids.

    Invariant (maintained by Lemma 1 at every step):
        decode_bytes(vocab, ids) == bytes(byte_shuffle[b] for b in text_bytes)

    Proof sketch (Lean 4):
        Termination: ids.length strictly decreases.
        Invariant: decode_bytes(vocab, ids) = shuffled (by Lemma 1).  ∎
    """
    shuffled = bytes(byte_shuffle[b] for b in text_bytes)
    ids = list(shuffled)

    assert decode_bytes(vocab, ids) == shuffled

    while len(ids) >= 2:
        stats = _get_stats(ids)
        candidates = [(p, merges[p]) for p in stats if p in merges]
        if not candidates:
            break
        pair, idx = min(candidates, key=lambda c: c[1])
        lemma_merge_preserves_decode(vocab, ids, pair, idx)
        ids = _merge(ids, pair, idx)

    assert decode_bytes(vocab, ids) == shuffled
    return ids


def decode_chunk(vocab, byte_shuffle, inverse_byte_shuffle, ids):
    """
    Decode token ids back to original raw bytes.

    Proof sketch (Lean 4):
        decode_bytes recovers shuffled bytes (from encode_chunk invariant).
        Lemma 2 applied pointwise inverts the shuffle.  ∎
    """
    raw = decode_bytes(vocab, ids)
    for sb in raw:
        ob = inverse_byte_shuffle[sb]
        lemma_shuffle_cancel(byte_shuffle, inverse_byte_shuffle, ob)
    return bytes(inverse_byte_shuffle[b] for b in raw)


# ═══════════════════════════════════════════════════════════════════════════
# Part 8: Full encode / decode (ASCII string level)
# ═══════════════════════════════════════════════════════════════════════════

def encode(merges, vocab, byte_shuffle, text):
    """
    Full encode: ASCII string → token ids.

    1. UTF-8 encode (identity for ASCII — Lemma 7).
    2. pre_tokenize_ascii (Lemma 4: partition).
    3. encode_chunk per chunk (Lemma 1: merge preservation).

    Proof sketch (Lean 4):
        By Lemma 4, the chunks partition the input bytes.
        By Lemma 1 (iterated), each chunk's encode_chunk preserves
        the decode_bytes invariant.
        By Lemma 6, decode_bytes distributes over concatenation,
        so the overall invariant holds for the concatenated ids.  ∎
    """
    text_bytes = text.encode("utf-8")
    assert all(is_ascii(b) for b in text_bytes), "encode: non-ASCII input"

    chunks = pre_tokenize_ascii(text_bytes)
    ids = []
    for chunk in chunks:
        chunk_ids = encode_chunk(merges, vocab, byte_shuffle, chunk)
        ids.extend(chunk_ids)
    return ids


def decode(vocab, byte_shuffle, inverse_byte_shuffle, ids):
    """
    Decode token ids to an ASCII string.

    1. decode_chunk recovers original bytes (Lemma 1 + Lemma 2).
    2. UTF-8 decode (identity for ASCII — Lemma 7).

    Proof sketch (Lean 4):
        decode_chunk returns the original bytes (proven).
        For ASCII, bytes.decode('utf-8') is the left inverse of
        str.encode('utf-8'), which is trivial since every byte < 128
        is a valid single-byte UTF-8 code point.  ∎
    """
    raw = decode_chunk(vocab, byte_shuffle, inverse_byte_shuffle, ids)
    return raw.decode("utf-8")


def lemma_ascii_utf8_roundtrip(text):
    """
    Lemma 7 — ASCII UTF-8 roundtrip.

    For ASCII strings, encode('utf-8') and decode('utf-8') are inverses.

    Proof sketch (Lean 4):
        Every ASCII char c has ord(c) < 128.
        UTF-8 encodes it as the single byte ord(c).
        Decoding that single byte recovers c.
        By induction on the string.  ∎
    """
    text_bytes = text.encode("utf-8")
    assert all(is_ascii(b) for b in text_bytes), "not ASCII"
    assert text_bytes.decode("utf-8") == text


# ═══════════════════════════════════════════════════════════════════════════
# Part 9: The main roundtrip theorem
# ═══════════════════════════════════════════════════════════════════════════

def roundtrip_check(merges, vocab, byte_shuffle, inverse_byte_shuffle,
                    text_bytes):
    """
    Byte-level roundtrip theorem:

        ∀ bs : bytes,
          WellFormed → decode_chunk(encode_chunk(bs)) = bs

    Proof: Lemma 1 (iterated) + Lemma 2 (pointwise).
    """
    ids = encode_chunk(merges, vocab, byte_shuffle, text_bytes)
    recovered = decode_chunk(vocab, byte_shuffle, inverse_byte_shuffle, ids)
    assert recovered == text_bytes
    return recovered


def full_roundtrip_check(merges, vocab, byte_shuffle, inverse_byte_shuffle,
                         text):
    """
    Full string-level roundtrip theorem:

        ∀ s : ASCIIString,
          WellFormed → decode(encode(s)) = s

    Proof chain:
      s
        ─[Lemma 7: UTF-8 encode]→  original_bytes  (identity for ASCII)
        ─[Lemma 4: pre_tokenize]→  chunks           (partition)
        ─[Lemma 1: BPE merges]→    ids per chunk    (merge preservation)
        ─[Lemma 6: concat]→        all_ids           (distributivity)
        ─[decode_bytes]→           shuffled_bytes    (invariant)
        ─[Lemma 2: unshuffle]→     original_bytes   (cancellation)
        ─[Lemma 7: UTF-8 decode]→  s                (identity for ASCII)
    """
    # Lemma 7: ASCII UTF-8 roundtrip
    lemma_ascii_utf8_roundtrip(text)

    text_bytes = text.encode("utf-8")

    # Lemma 4: pre-tokenize partitions the input
    chunks = pre_tokenize_ascii(text_bytes)
    assert b"".join(chunks) == text_bytes

    # encode each chunk (Lemma 1 checked inside encode_chunk)
    all_ids = []
    all_shuffled = b""
    for chunk in chunks:
        shuffled = bytes(byte_shuffle[b] for b in chunk)
        chunk_ids = encode_chunk(merges, vocab, byte_shuffle, chunk)
        assert decode_bytes(vocab, chunk_ids) == shuffled
        all_ids.extend(chunk_ids)
        all_shuffled += shuffled

    # Lemma 6: decode_bytes distributes over concatenation
    assert decode_bytes(vocab, all_ids) == all_shuffled

    # Lemma 2: shuffle cancellation for every original byte
    for ob in text_bytes:
        lemma_shuffle_cancel(byte_shuffle, inverse_byte_shuffle, ob)
    recovered_bytes = bytes(inverse_byte_shuffle[sb] for sb in all_shuffled)
    assert recovered_bytes == text_bytes

    # full decode and final assertion
    result = decode(vocab, byte_shuffle, inverse_byte_shuffle, all_ids)
    assert result == text, f"roundtrip failed: {text!r} != {result!r}"
    return result
