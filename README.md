# bpetokenizer_prototype

## Files
1. `core.py`: All the provable theorems (and lemmas) within my scope.

Final Goal:
```
∀ s : ASCIIString, WellFormed(merges, vocab, byte_shuffle) → decode(encode(s)) = s
```

Intermediate Lemmas:
```
s : ASCIIString
  ─[Lemma 7]→  original_bytes        (UTF-8 encode = identity for ASCII)
  ─[Lemma 4]→  chunks                (pre_tokenize partitions the input)
  ─[Lemma 1]→  token_ids per chunk   (each BPE merge preserves decoded bytes)
  ─[Lemma 6]→  all_ids               (decode_bytes distributes over ++)
  ─[Lemma 2]→  original_bytes        (inverse_shuffle cancels shuffle)
  ─[Lemma 7]→  s                     (UTF-8 decode = identity for ASCII)
```

2. `test_roundtrip.py`: Test cases.

Compare my result with the result by OpenAI's official tokenization library `tiktoken`. I'm testing it with the encoding data of GPT-4 `cl100k_base`, but the tokenization algorithm itself should be encoding-data-agnostic, and it should work if you change `cl100k_base` (GPT-4) to `o200k_base` (GPT-4o), etc. 

3. `vocab.py`: Helper functions.

Helper functions to load `cl100k_base`, check input well-formed (as a test), and recover_merges to construct a well-formed input. 

---

## Execution

To prove my prototype's correctness, I have implemented a `test_roundtrip.py` file so that I could compare it with OpenAI's official tokenization library `tiktoken`. You can run the test cases by running: 

```python
python test_roundtrip.py
```

---

## Pseudocode in provable `core.py`

1. Part 1: Character Classifiers

```
is_ascii(b)   := 0 ≤ b ≤ 127
is_letter(b)  := 65 ≤ b ≤ 90  ∨  97 ≤ b ≤ 122
is_digit(b)   := 48 ≤ b ≤ 57
is_space(b)   := b ∈ {9, 10, 11, 12, 13, 32}
is_newline(b) := b = 10  ∨  b = 13
is_punct(b)   := is_ascii(b) ∧ ¬is_letter(b) ∧ ¬is_digit(b) ∧ ¬is_space(b) ∧ b ≥ 32
to_lower(b)   := if 65 ≤ b ≤ 90 then b + 32 else b
```

2. Part 2: ASCII Pre-Tokenizer (State Machine)

Implements GPT-4's cl100k_base split pattern restricted to ASCII, via 7 priority-ordered branches:
```
Branch 1 (contraction):  '(?i:[sdmt]|ll|ve|re)
Branch 2 (word):         [^\r\n\p{L}\p{N}]?\p{L}+
Branch 3 (digits):       \p{N}{1,3}
Branch 4 (punctuation):  ' '?[^\s\p{L}\p{N}]+[\r\n]*
Branch 5 (newline-ws):   \s*[\r\n]
Branch 6 (trailing-ws):  \s+(?!\S)
Branch 7 (other-ws):     \s+
Fallback:                consume 1 byte  (control chars 0-8, 14-31, 127)
```

3. Lemma 4: Pre-tokenizer Splits Partitions

```
∀ data : ByteArray,
    (∀ b ∈ data, is_ascii(b)) → ByteArray.join(pre_tokenize_ascii(data)) = data
```

4. Part 3: BPE Helpers

```
get_stats(ids : List Nat) → Map (Nat × Nat) Nat :=
    count each consecutive pair (ids[i], ids[i+1])

merge(ids : List Nat, pair : Nat × Nat, idx : Nat) → List Nat :=
    replace every adjacent (pair.1, pair.2) with idx

decode_bytes(vocab : Map Nat ByteArray, ids : List Nat) → ByteArray :=
    ByteArray.join [vocab[id] | id ∈ ids]
```

5. Lemma 1: Merge Preservation

```
∀ vocab ids pair idx, 
    vocab[idx] = vocab[pair.1] ++ vocab[pair.2] → decode_bytes(vocab, merge(ids, pair, idx)) = decode_bytes(vocab, ids)
```

6. Lemma 2: Shuffle Cancellation

```
∀ b : Fin 256, inverse_byte_shuffle[byte_shuffle[b]] = b
```

7. Lemma 6: Decode-Bytes Distributivity

```
∀ vocab xs ys,
    decode_bytes(vocab, xs ++ ys) = decode_bytes(vocab, xs) ++ decode_bytes(vocab, ys)
```

8. Part 4
```
decompose_token(ranks : Map ByteArray Nat, token : ByteArray, max_rank : Option Nat)
    → List ByteArray :=
    parts ← [bytes([b]) | b ∈ token]     -- split into single bytes
    loop:
        best ← None                       -- : Option (Nat × Nat)
        for i in 0 .. parts.length - 1:
            merged ← parts[i] ++ parts[i+1]
            rank ← ranks.get?(merged)
            if rank < max_rank and rank < best.rank:
                best ← (i, rank)
        if best = None: break
        parts[best.i] ← parts[best.i] ++ parts[best.i + 1]
        remove parts[best.i + 1]
    return parts
```

9. Lemma 7: Decompose Roundtrip

```
∀ ranks token max_rank,
    ByteArray.join(decompose_token(ranks, token, max_rank)) = token
```

10. Part 5: Check Well-formed

```
structure WellFormed (merges : Map (Nat × Nat) Nat)
                     (vocab  : Map Nat ByteArray)
                     (byte_shuffle : Fin 256 → Fin 256) : Prop where
    base_tokens    : ∀ i : Fin 256, vocab[i] = ByteArray.mk #[i]
    merge_decomp   : ∀ (p0 p1 idx), merges[(p0,p1)] = idx →
                       vocab[idx] = vocab[p0] ++ vocab[p1]
                     ∧ vocab[p0].size < vocab[idx].size
                     ∧ vocab[p1].size < vocab[idx].size
    injectivity    : ∀ i j, vocab[i] = vocab[j] → i = j
    shuffle_bijection : Function.Bijective byte_shuffle
```

and:

```
build_vocab(merges) → vocab :=
    vocab[i] ← ByteArray.mk #[i]    for i ∈ 0..255
    for (p0, p1) ↦ idx in merges:
        vocab[idx] ← vocab[p0] ++ vocab[p1]
    return vocab
```

11. Part 6: Encode/Decode Byte

```
encode_chunk(merges, vocab, byte_shuffle, text_bytes) → List Nat :=
    shuffled ← [byte_shuffle[b] | b ∈ text_bytes]
    ids ← shuffled                              -- invariant: decode_bytes(vocab, ids) = shuffled
    while |ids| ≥ 2:
        stats ← get_stats(ids)
        candidates ← {(p, merges[p]) | p ∈ stats, p ∈ dom(merges)}
        if candidates = ∅: break
        (pair, idx) ← argmin rank candidates
        -- Lemma 1 fires here
        ids ← merge(ids, pair, idx)
    return ids

decode_chunk(vocab, byte_shuffle, inverse_byte_shuffle, ids) → ByteArray :=
    raw ← decode_bytes(vocab, ids)
    return [inverse_byte_shuffle[b] | b ∈ raw]  -- Lemma 2 fires here
```

12. Part 7: Encode/Decode String

```
encode(merges, vocab, byte_shuffle, text : ASCIIString) → List Nat :=
    text_bytes ← text.toUTF8                     -- identity for ASCII (Lemma 7)
    chunks ← pre_tokenize_ascii(text_bytes)       -- Lemma 4: partition
    return chunks.flatMap(encode_chunk)            -- Lemma 1: per chunk

decode(vocab, byte_shuffle, inverse_byte_shuffle, ids : List Nat) → ASCIIString :=
    raw ← decode_chunk(vocab, byte_shuffle, inverse_byte_shuffle, ids)
    return String.fromUTF8(raw)                   -- identity for ASCII (Lemma 7)
```

13: Part 8: Main theorem: For all ascii, Decode(Encode(s)) == s

```
theorem ascii_bpe_roundtrip
    (merges : Map (Nat × Nat) Nat)
    (vocab  : Map Nat ByteArray)
    (byte_shuffle : Fin 256 → Fin 256)
    (inverse_byte_shuffle : Fin 256 → Fin 256)
    (hw : WellFormed merges vocab byte_shuffle)
    (hinv : ∀ b, inverse_byte_shuffle (byte_shuffle b) = b)
    (s : ASCIIString) :
    decode(vocab, byte_shuffle, inverse_byte_shuffle,
           encode(merges, vocab, byte_shuffle, s)) = s
```