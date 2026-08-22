# CS779-Neural-Machine-Translation

Neural Machine Translation system for English → Hindi and English → Bengali, built for the **CS779: Statistical Natural Language Processing** capstone competition at IIT Kanpur. All models are implemented from scratch in PyTorch — no pretrained contextual models (BERT, mBART, etc.) were allowed, only static embeddings (FastText/GloVe) where used.

**Final result:** Bidirectional GRU Encoder–Decoder with Cross Attention and Beam Search decoding — chrF++ 0.331, ROUGE 0.431, BLEU 0.123. Ranked 58th on the test-phase leaderboard.

Full writeup: [`Report/CS779-CP-om-mallick-251110055-report.pdf`](./Report/CS779-CP-om-mallick-251110055-report.pdf)

## Repo structure

```
CS779-Neural-Machine-Translation/
├── README.md
├── .gitignore
├── Notebooks/
│   ├── 01-toy-model-gru-baseline.ipynb
│   └── 02-toy-model-gru-indicnlp.ipynb
├── Final Notebook/
│   └── CS779-CP-om-mallick-251110055.ipynb
├── Report/
│   └── CS779-CP-om-mallick-251110055-report.pdf
└── Submissions/
    └── answer.csv
```

## About the notebooks

The project went through three stages, kept here in progression:

### `Notebooks/` — Early iterations

1. **`01-toy-model-gru-baseline.ipynb`** — the earliest baseline. A simple unidirectional GRU encoder–decoder with no attention, trained separately for English→Bengali and English→Hindi. Basic preprocessing (NLTK tokenization, no Indic-specific normalization).

2. **`02-toy-model-gru-indicnlp.ipynb`** — same unidirectional GRU baseline architecture, but with proper text preprocessing added using the **IndicNLP Library** (Unicode normalization for Hindi/Bengali, Indic-aware tokenization). This was the preprocessing groundwork that carried through to the final model.

These two are kept for reference on the iterative development process, but were not the final submitted models.

### `Final Notebook/` — Submitted model

**`CS779-CP-om-mallick-251110055.ipynb`** — the final submitted model. A **Bidirectional GRU Encoder–Decoder with Bahdanau-style cross attention**, trained with a Warmup-Cosine LR scheduler and gradient clipping, decoded with **beam search** at inference. This notebook includes:
- Corpus vocabulary and statistical analysis (vocabulary sizes, sentence length distributions, Zipf analysis)
- Full data preprocessing pipeline (IndicNLP normalization, tokenization, padding, sequence encoding)
- Model training and validation on both English–Bengali and English–Hindi tasks
- Training curves (loss and accuracy plots over epochs)
- Attention heatmap visualizations for qualitative error analysis
- Beam search decoding and final predictions merged into a single output file

This is the model described in the report, the one that achieved chrF++ 0.331 / ROUGE 0.431 / BLEU 0.123, and ranked 58th on the final leaderboard.

## Report and submissions

- **`Report/CS779-CP-om-mallick-251110055-report.pdf`** — the full competition report: problem description, data analysis, model architecture, experiments, hyperparameter search, results, error analysis, and key learnings. Includes detailed discussion of architectural evolution (unidirectional → bidirectional, no attention → attention), ablations (GRU vs. LSTM, pretrained embeddings, Transformer variants), and insights on decoding strategies (greedy vs. beam search).

- **`Submissions/answer.csv`** — the final prediction output file in the competition's submission format (ID, Translation pairs for the test set).

## Model architecture (final)

- **Encoder:** token embeddings → bidirectional GRU; forward/backward hidden states concatenated and projected into a shared context.
- **Decoder:** single-layer GRU generating tokens one at a time, using additive (Bahdanau) attention over encoder outputs to dynamically focus on relevant source words.
- **Regularization:** dropout on embeddings and GRU outputs.
- **Loss:** cross-entropy, ignoring `<PAD>` tokens.
- **Optimizer:** Adam (`lr=1e-3`, weight decay `1e-5`) with a Warmup-Cosine LR scheduler and gradient clipping (L2 norm 1.0).
- **Decoding:** beam search (which replaced an earlier greedy-search approach).

For detailed architecture diagrams, hyperparameter sweep tables, and ablation studies, see the report.

## Key findings

- **Bidirectional encoding** significantly improved source sentence representation for morphologically complex languages.
- **Attention mechanisms** enabled better translation of longer sentences and improved handling of word-order differences between English and Indian languages.
- **Beam search decoding** produced more fluent and complete translations compared to greedy decoding, which often prematurely predicted the end-of-sequence token.
- **LSTM vs. GRU:** GRU variants converged faster with similar or better accuracy, while consuming less GPU memory.
- **Pretrained embeddings (FastText):** showed minimal improvement relative to the significant increase in training time and memory overhead.
- **Transformer architectures:** used ~3x more GPU memory than GRU models, with only marginal accuracy gains — not practical under resource constraints.
- **IndicNLP preprocessing:** proper Unicode normalization and Indic-aware tokenization were essential for stability and convergence.

## Data

The competition dataset consisted of parallel English–Bengali (~68.8K training, ~19.7K test) and English–Hindi (~80.8K training, ~23.1K test) corpora in JSON format, not redistributed in this repo. Vocabulary sizes: En-Bn (65K English, 92K Bengali), En-Hi (70K English, 69K Hindi). See the report for detailed corpus statistics and analysis.

## Requirements

- Python 3
- PyTorch
- NLTK
- `indic-nlp-library`
- pandas, numpy, matplotlib, tqdm

```bash
pip install torch nltk indic-nlp-library pandas numpy matplotlib tqdm
```

## Author

Om Mallick (251110055) — IIT Kanpur, 2025
