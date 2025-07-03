import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_semantic_search as aux_semantics
import aux_document_retrieval_hybrid as aux_hybrid
import pandas as pd
from pathlib import Path
import logging
import nltk
import yaml


nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main():
    setup_logging()
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Convert string paths to Path objects
    paths = {k: Path(v) for k, v in config["paths"].items()}
    query = config["query"]
    top_k = config["top_k"]

    results_df = pd.DataFrame()

    try:    #i am changing the methods for convenience sake
        #top_k_vsm = aux_vsm.run_word2vec_query(paths, query, top_k=top_k,use_expansion=True,multivector=True)

        #---------------------------------seems much more complicated than the line up top, but its modular
        top_k = 20
        use_expansion = True

        resources = aux_vsm.load_word2vec_resources(
            paths,
            use_multivector=False
        )
        query_runner = lambda q: aux_vsm.run_word2vec_query_preloaded(
            resources=resources,
            query=q,
            top_k=top_k,
            use_expansion=use_expansion
        )
        top_k_vsm = query_runner(query)
        # -----------------------------------------------------------------------------------------


        top_k_bm25 = aux_bm25.run_bm25_query(paths, query, top_k=top_k)

        aux_vsm.print_documents(top_k_vsm, top_k=top_k)
        aux_bm25.print_documents(top_k_bm25, top_k=top_k)
        # aux_hybrid.results_to_dataframe(top_k_vsm,ranking="vsm",)

        top_k_hybrid = aux_hybrid.hybrid_retrieval(
            top_k_vsm,
            top_k_bm25,
            weight_vsm=0.7,
            weight_bm25=0.3,
            norm_vsm='minmax',
            norm_bm25='zscore'
        )
        aux_hybrid.print_documents(top_k_hybrid, top_k=top_k, ranking="hybrid")

        # Reciprocal rank function is guided by the ranking position and not the score.
        top_k_rrf = aux_hybrid.rrf(
            top_k_vsm["results"],
            top_k_bm25,
            k=60
        )
        aux_hybrid.print_documents(top_k_rrf, top_k=top_k, ranking="RRF")

        top_k_reranked = aux_hybrid.rerank(paths, query, top_k=top_k,mode="bm25-vsm")
        #aux_hybrid.rerank(paths, query, top_k=top_k,mode="vsm-bm25")

        aux_hybrid.print_documents(top_k_reranked,top_k,"bm25 + vsm")

    except Exception as e:
        logging.error("Application error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()