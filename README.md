# vgrep

Semantic search over your own images. Describe what you're looking for; get the photo. Runs entirely on your machine.

```console
$ vgrep "radio city music hall"
 24.5%  ~/nyc/radio-city-music-hall-v0-l8zui41fw5nf1.webp
 20.7%  ~/nyc/images__8_.jpeg
 13.6%  ~/corpus/toronto_skyline/8837155.jpg
```

No filenames were matched, no tags, no EXIF. Ranking is on image content alone.

## Status

v0.1.0-dev. Images only; the core index-and-search loop works and is benchmarked below.

## Why

Apple Photos and similar tools classify images against a fixed label vocabulary.
That handles "dog" and fails on "lands end sunset" — not because the ranking is
worse, but because the query can't be represented at all.

vgrep uses an open-vocabulary vision-language model, so images and text land in
the same embedding space and any phrase is a valid query.

Nothing leaves your machine. No API key, no account, no upload.

## Install

Python 3.10+. Apple Silicon recommended.

```bash
git clone https://github.com/ryamabra/vgrep
cd vgrep
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

First run downloads model weights (~1.5 GB), once.

## Use

```bash
vgrep index ~/Pictures        # encode everything (one-time, resumable)
vgrep "a dog"                 # search
vgrep "a dog" --open          # open the best match
vgrep "sunset" --k 20         # more results
vgrep "receipts" --paths      # bare paths, for piping
vgrep status                  # what's indexed
vgrep reset                   # start over
```

Re-running `index` only encodes files that are new or changed, so keeping the
index current is cheap.

## Results

Benchmarked on a 330-image corpus across 22 categories (`tools/fetch_corpus.py`
builds it; each photo's folder is its label, so precision is measurable rather
than eyeballed). Run `python tools/benchmark.py` to reproduce.

**Common visual categories are solved.** Queries like "a dog" and "food on a
plate" return 10/10 correct results, with no cross-category contamination.

**Named landmarks work, and beat descriptions.** This was the surprise. Naming a
place outscores describing it:

| query | top score | correct |
|---|---|---|
| "radio city music hall" | 24.5% | yes |
| "a neon theater sign at night" | 13.6% | no |
| "the brooklyn bridge" | 18.3% | yes |
| "a suspension bridge with stone arches" | 14.8% | partly |

The model carries real landmark knowledge. You don't need to describe an image
to find it — specificity helps.

**It encodes real-world relationships.** Querying "rockefeller center" ranks the
Rockefeller photo first and Radio City second and third. That looks like a false
positive until you notice Radio City *is* in Rockefeller Center.

**Where it fails: visually generic places.** "central park" returns Toronto
Islands; "bryant park" returns Radio City. All are green lawns with people and a
skyline behind, and telling them apart means identifying specific buildings in
the background — much finer-grained than reading a marquee. The rule that emerges:
distinctive architecture works, places identified by context don't.

Scores are low in absolute terms (SigLIP produces smaller raw cosines than CLIP);
what matters for ranking is the gap above the noise floor, measured at ~8%.

## Performance

MacBook Air M2, steady state. The chassis is fanless and throttles under sustained
load, so these are lower than a first-minute measurement would suggest.

| | |
|---|---|
| Encode, JPEG | 17.9 img/s |
| Encode, HEIC | 4.3 img/s |
| Search | <1 ms over 345 images |
| Storage | ~3 KB per image |

The 4x gap between JPEG and HEIC is the main finding here: **decode dominates,
not inference.** An early 4.3 img/s measurement looked like a model bottleneck
and was actually libheif. Image loading runs across a thread pool feeding a
single encoder.

## How it works

```
index:   files --> decode --> SigLIP 2 image encoder --> SQLite --> flat index
search:  query --> SigLIP 2 text encoder --> top-k --> paths
```

**SQLite is the source of truth; the index is derived.** Embeddings live in
SQLite keyed by `(path, mtime, size)`. The vector index is rebuilt from it and
can be deleted at any time. This makes interrupted runs resumable and
re-indexing incremental.

**Search is exact, and numpy-only.** This started on FAISS, but `faiss-cpu` and
`torch` each bundle their own `libomp.dylib`, and loading both into one process
aborts on macOS. For a flat index FAISS was doing nothing numpy can't: search is
one matrix-vector product against L2-normalised rows, i.e. exact cosine ranking.
Under 1 ms here, and still well under 100 ms at 100k images — the approximate
structures FAISS exists to provide aren't needed at this scale. Dropping it
removed both the crash and a native dependency.

**Scores are shown, not hidden.** Similarity varies enormously by query type.
Presenting ten results identically implies a confidence the model doesn't have,
so vgrep prints the score and colours low-confidence hits differently. `--min`
filters them.

## Roadmap

- [x] Image indexing and search
- [x] Incremental re-indexing
- [x] Benchmark harness with ground-truth labels
- [ ] `vgrep shell` — persistent session, no per-query model load
- [ ] Exclude paths from indexing
- [ ] PDF and text file support
- [ ] `vgrep watch` for background indexing

## Prior art

[mgrep](https://github.com/mixedbread-ai/mgrep) covers similar ground with a
much broader feature set, but indexes to a cloud service — files are uploaded,
and its author has said a local version is some way off. vgrep is smaller and
fully local.

## License

MIT
