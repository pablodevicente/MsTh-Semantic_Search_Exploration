## An Intelligent Semantic Search System for Digital Product  Part Design Catalogues in Manufacturing

Developed a modular framework for **semantic similarity** search over industrial product catalogues, enabling end-to-end **document retrieval** and **ranking**. The system supports a wide range of configurable strategies, including VSM, BM25, RRF, hybrid reranking, single- and multi-vector representations, normalization, and query expansion. Its flexible architecture allows for easy experimentation and fine-tuning of retrieval pipelines under a unified setup.

This work builds on precomputed embeddings generated from raw PDF data—see the companion repository “Internship 2: Categorization of Textual Industrial (Technical) Reports” for details on parsing and embedding generation.

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

