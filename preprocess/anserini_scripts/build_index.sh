python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 8 \
                         -input path_to_collection \
                         -index path_to_index \
                         -storePositions -storeDocvectors