## An Intelligent Semantic Search System for Digital Product  Part Design Catalogues in Manufacturing

A *modular framework* for **semantic similarity search** over digital product catalogues in industrial settings. This repository focuses on end‑to‑end retrieval and ranking—starting from **precomputed embeddings** (see the companion repo _“Internship 2: Categorization of Textual Industrial (Technical) Reports”_ for raw PDF parsing and embedding generation) through **configurable retrieval pipelines** (VSM, BM25, RRF, hybrid reranking, single‑ vs. multi‑vector representations, normalization, query expansion, etc.). Its **flexible design** makes it easy to compare and fine‑tune diverse strategies under a unified experimental setup.  

---

### 📖 Context & Motivation

Industrial product catalogs combine structured tables, images, and descriptive text. Standard NLP tools often struggle with their complex formatting. Leveraging a dataset of \~100 labeled DigiKey PDFs per product category, our system enables robust information extraction and retrieval in the manufacturing domain.

### 🚀 Features & Capabilities

- **Document Processing**: Extract text, tables, and images using customizable parser/OCR backends.
- **Representation Modes**:
  - Single-vector embeddings (document-level)
  - Multi-vector embeddings (section-level)
- **Retrieval Strategies**:
  - Vector Space Model (VSM) & BM25
  - Fusion methods (Reciprocal Rank Fusion)
  - Hybrid pipelines with reranking
- **Optional Enhancements**:
  - Semantic query expansion
  - Score normalization (minmax, zscore)

### 📂 Project Structure

```bash
├── 00-documentation/       # In-depth design, research questions, and paper
├── 01-code/                # Core pipeline scripts and modules
└── 02-data/                # Raw PDFs, intermediate extracts, and outputs
```

### ⚙️ Configuration

Edit `config.yml` to customize:

- Input/output directories
- Parser/OCR selection
- Embedding and retrieval settings
- Logging levels

Example excerpt:

```yaml
paths:
  pdf_folder: ../02-data/raw/
  vsm_model: ../02-data/models/word2vec.bin
use_multivector: true
norm_method: minmax
```
Paths and variables will have to be adjusted for each use case


---

### 📝 Documentation & Report

See the included paper *"_Paper____Semantic_Similarity_Search_on_Product_Catalogues"* in `00-documentation/` for research background and evaluation results.

---

### 🧑‍💼 Team & Acknowledgments

| Role           | Name                             |
| -------------- | -------------------------------- |
| Lead Developer | Pablo de Vicente                 |
| Promotor       | Prof. Moharram Challenger        |
| Co-promotor    | Alireza Khalilipour              |
| Lab            | MICCS Lab, University of Antwerp |

Thanks to all collaborators and DigiKey for data access.

---

### 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

