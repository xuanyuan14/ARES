python -m pyserini.search --index path_to_index \
                          --topics path_to_queries \
                          --output path_to_trec \
                          --bm25 \
                          --hits 200