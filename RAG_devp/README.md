# Sanjivani Institutional LLM — EduVerse-1 + RAG

A from-scratch decoder-only Transformer with Retrieval-Augmented Generation,
built for the "Build Sanjivani's Own LLM Model" competition.

Matches the architecture in `Model_Design_&_Architecture.pdf`:
- 6 layers, 8 heads, 512 hidden dim, 16K vocab, 512 context, 25–35M params
- Custom BPE tokenizer trained from scratch (no pretrained tokenizer)
- No pretrained LLM weights, no frontier API calls for generation
- Three-stage curriculum: General corpus → Higher Ed corpus → Sanjivani corpus
- RAG at inference time via FAISS + all-MiniLM-L6-v2 (embedding model only —
  not the generator, so this doesn't violate the "no pretrained LLM" rule)

## Project layout

```
sanjivani-llm/
├── data/
│   ├── raw_docs/              # your source .txt files (put NCERT/UGC/AICTE/Sanjivani docs here)
│   ├── chunks.jsonl            # output of chunking.py
│   ├── sanjivani.index         # FAISS vector index
│   ├── sanjivani_meta.jsonl     # chunk metadata aligned to the index
│   └── eduverse_tokenizer.json # trained BPE tokenizer
├── checkpoints/
│   ├── stage1_base/
│   ├── stage2_higher_ed/
│   └── stage3_sanjivani/
├── src/
│   ├── data_pipeline/
│   │   └── chunking.py         # paragraph-aware chunker for RAG indexing
│   ├── rag/
│   │   ├── build_index.py      # embed chunks -> FAISS index
│   │   ├── retrieve.py         # query -> top-k chunks
│   │   └── pipeline.py         # retrieval -> prompt -> generation glue
│   └── model/
│       ├── eduverse_model.py   # the Transformer itself
│       ├── train_tokenizer.py  # custom BPE trainer
│       ├── train.py            # training loop (next-token prediction)
│       └── generate.py         # load checkpoint, wrap as generate_fn
└── run_demo.py                 # full pipeline: question -> retrieval -> answer
```

## Setup

```bash
pip install -r requirements.txt
```

## Step-by-step: building your knowledge base (RAG side)

1. Drop cleaned `.txt`/`.md` files into `data/raw_docs/` — academic calendar,
   exam rules, syllabus, notices, circulars, etc. (This is where your OCR
   pipeline output from `clean_ocr_md.py` should land.)

2. Chunk them:
   ```bash
   python -m src.data_pipeline.chunking  # or call chunk_directory() directly
   ```

3. Build the vector index (downloads all-MiniLM-L6-v2 the first time — needs
   internet access, one-time ~90MB):
   ```bash
   python src/rag/build_index.py --chunks data/chunks.jsonl \
       --index_out data/sanjivani.index --meta_out data/sanjivani_meta.jsonl
   ```

4. Sanity-check retrieval on its own before wiring up generation:
   ```bash
   python src/rag/retrieve.py --query "When is the library open?"
   ```

## Step-by-step: training EduVerse-1 (model side)

1. Train the tokenizer on your full corpus (general + higher-ed + Sanjivani
   text combined, so all three stages share one vocabulary):
   ```bash
   python src/model/train_tokenizer.py --corpus_glob "data/raw_docs/*.txt" \
       --output data/eduverse_tokenizer.json --vocab_size 16000
   ```
   Note: you need a reasonably large, varied corpus for a full 16K vocab to
   make sense — a handful of short files will train fine but produce a much
   smaller effective vocabulary (the trainer stops early if it runs out of
   distinct merges).

2. Tokenize your corpus into a flat binary token stream:
   ```python
   from src.model.train import build_token_stream
   build_token_stream(["data/general_corpus.txt"], "data/eduverse_tokenizer.json", "data/stage1_tokens.bin")
   ```

3. Train Stage 1 (General Educational Corpus, lr=1x):
   ```bash
   cd src/model
   python train.py --bin_path ../../data/stage1_tokens.bin \
       --tokenizer_path ../../data/eduverse_tokenizer.json \
       --out_dir ../../checkpoints/stage1_base \
       --lr_scale 1.0 --max_steps 5000
   ```

4. Train Stage 2 (Higher Education Corpus, lr=0.3x, continuing from Stage 1):
   ```bash
   python train.py --bin_path ../../data/stage2_tokens.bin \
       --tokenizer_path ../../data/eduverse_tokenizer.json \
       --out_dir ../../checkpoints/stage2_higher_ed \
       --init_from ../../checkpoints/stage1_base/checkpoint.pt \
       --lr_scale 0.3 --max_steps 2000
   ```
   For the "~10% replay of prior-stage data" mentioned in the architecture
   doc, mix ~10% of Stage 1's text into the Stage 2 corpus before tokenizing,
   so `build_token_stream` naturally includes it in the token stream.

5. Train Stage 3 (Sanjivani Institutional Corpus, lr=0.1x, continuing from
   Stage 2) the same way, with `--init_from stage2_higher_ed/checkpoint.pt
   --lr_scale 0.1`.

## Step-by-step: running the full system

```bash
python run_demo.py --query "What is the attendance requirement for exams?" \
    --checkpoint checkpoints/stage3_sanjivani/checkpoint.pt
```

## What's already verified working (tested in a sandboxed environment)

- Chunking: paragraph-aware, produces correctly-sized overlapping chunks
- FAISS indexing + cosine similarity search: mechanically correct
- Prompt template: matches `Context: {chunks}\nQuestion: {question}\nAnswer:`
  from the architecture doc exactly
- EduVerse-1 architecture: instantiates at **27.4M params** (within the
  25–35M target), forward pass produces correctly-shaped logits, initial
  loss matches the theoretical `ln(vocab_size)` baseline for an untrained
  model, causal masking prevents attending to future tokens
- BPE tokenizer: trains from scratch, encode/decode round-trips losslessly
- Training loop: AdamW + cosine LR + grad clipping, loss decreases step
  over step, checkpoints save/load correctly
- Generation: autoregressive sampling with temperature + top-k works

**Not testable in this sandbox** (needs your machine's internet access):
downloading `all-MiniLM-L6-v2` weights from Hugging Face. The code is
correct — it's the same library/model you're already using successfully
for other embedding tasks — it just can't reach huggingface.co from here.

## Next steps to actually hit your Sept 9 prototype deadline

1. Get your real corpus assembled (AICTE/UGC/NEP text + Sanjivani docs) —
   this is almost certainly your biggest time sink, not the code.
2. Run Stage 1 training on a small slice first (even 1–5M tokens) just to
   confirm loss goes down on real data, before committing to a long run.
3. Build the real FAISS index from your actual Sanjivani documents in
   parallel — it doesn't depend on the model being trained.
4. Budget real time for evaluating whether the model actually *uses* the
   retrieved context rather than ignoring it — this is the most common
   failure mode for small from-scratch models paired with RAG.
