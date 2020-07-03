# Sensecluster

Submission to [Semeval 2020 task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948#learn_the_details-overview)

The system embeds target words using xlmr.large, clusters the resulting *contextualized* embeddings using kmeans++, and uses the resulting cluster assignments as a direct proxy for senses.


To run: 
```
# install requirements found in requirements.txt using conda or pip

# Extract the contexts for the given target words
# This populates the directory with LANGUAGE_CORPUS.ctx files.
python mk_contexts.py /path/to/test_data_public

# Run XLMR to construct embeddings for each occurence
# This reads the LANGUAGE_CORPUS.ctx files and creates LANGUAGE_CORPUS.emb files.
python embed.py

# Run clustering on the contextualized embeddings
# This reads the LANGUAGE_CORPUS.emb files and populates the answer/ directory. 
python cluster.py
```
# References
```
@inproceedings{schlechtweg2020semeval,
title = "{S}em{E}val-2020 {T}ask 1: {U}nsupervised {L}exical {S}emantic {C}hange {D}etection",
author = "Schlechtweg, Dominik and McGillivray, Barbara and Hengchen, Simon and Dubossarsky, Haim and Tahmasebi, Nina",
booktitle = "To appear in Proceedings of the 14th International Workshop on Semantic Evaluation",
year = "2020",
address = "Barcelona, Spain",
publisher = "Association for Computational Linguistics"}
```
