# On English

1. download word2vec: <https://wikipedia2vec.github.io/wikipedia2vec/pretrained/> (I used *enwiki_20180420* *100d (txt)*)
2. exclude entities and take the first 10000 forms `grep -v ENTITY <enwiki_20180420_100d.txt | head -n 10001 | tail -n -10000 >enwiki_20180420_100d_10000.txt`
3. take forms `cut -d ' ' -f1 >enwiki_20180420_100d_10000_forms.txt <enwiki_20180420_100d_10000.txt` only
4. lemmatize and tag using MorphoDiTa `./morphodita.py enwiki_20180420_100d_10000_forms.txt`
5. add a header `echo -e "Word\tLemma\tTag\n$( cat annotated/enwiki_20180420_100d_10000_forms.txt )" >lemma_dict.tsv`
6. add POS labels to the dictionary `python prepare_lemma_dict.py lemma_dict.tsv Penn-POS.tsv lemma_dict_POS.tsv`
7. prepare vectors `python prepare.py enwiki_20180420_100d_10000.txt processed_vectors.tsv 10000`
8. add annotations to vectors `python prepare-add_lemma.py lemma_dict_POS.tsv processed_vectors.tsv processed_vectors_annot.tsv`
9. run SVM `python SVM.py processed_vectors_annot.tsv processed_vectors_annot_SVM.tsv rbf 1.0 scale`
10. UMAP `python UMAP.py`