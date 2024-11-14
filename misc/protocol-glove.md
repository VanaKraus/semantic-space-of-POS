# GloVe 1

1. download word2vec: <https://nlp.stanford.edu/projects/glove/> (I used *glove.840B.300d.zip*)
2. take forms only `cut -d ' ' -f1 <glove.840B.300d.txt >glove.840B.300d_forms.txt`
3. take top 20000 entries `head -n 20000 <glove.840B.300d_forms.txt >glove.840B.300d_forms_20000.txt`. they will be reduced to 10000 later on.
4. lemmatize and tag using MorphoDiTa `../morphodita.py glove.840B.300d_forms_20000.txt` (w/ guesser)
5. add a header `echo -e "Word\tLemma\tTag\n$( cat annotated/glove.840B.300d_forms_20000.txt )" >lemma_dict.tsv`
6. add POS labels to the dictionary `python ../prepare_lemma_dict.py lemma_dict.tsv ../Penn-POS.tsv lemma_dict_POS.tsv`
7. `head -n 20000 <glove.840B.300d.txt >glove.840B.300d_20000.txt`. wouldn't have been neccessary if step 2 came before step 1.
8. prepare vectors `python ../prepare.py glove.840B.300d_20000.txt processed_vectors.tsv 20000`
9. add annotations to vectors `python ../prepare-add_lemma.py lemma_dict_POS.tsv processed_vectors.tsv processed_vectors_annot.tsv 10000`
10. run SVM `python ../SVM.py processed_vectors_annot.tsv processed_vectors_annot_SVM.tsv rbf 1.0 scale | tee SVM.log`
11. UMAP `python ../UMAP.py`
12. pure SVM vectors `<processed_vectors_annot_SVM.tsv cut -f 206- | tail -n 10000 >processed_vectors_annot_SVM_pure.tsv`
13. metadata `<processed_vectors_annot_SVM.tsv cut -f -5 >processed_vectors_annot_SVM_metadata.tsv`

# GloVe 2

1. download word2vec: <https://nlp.stanford.edu/projects/glove/> (I used *glove.840B.300d.zip*)
2. take forms only `cut -d ' ' -f1 <glove.840B.300d.txt >glove.840B.300d_forms.txt`
3. take top 20000 entries `head -n 20000 <glove.840B.300d_forms.txt >glove.840B.300d_forms_20000.txt`. they will be reduced to 10000 later on.
4. lemmatize and tag using MorphoDiTa `../morphodita.py glove.840B.300d_forms_20000.txt` (wo/ guesser)
5. add a header `echo -e "Word\tLemma\tTag\n$( cat annotated/glove.840B.300d_forms_20000.txt )" >lemma_dict2.tsv`
6. add POS labels to the dictionary `python ../prepare_lemma_dict.py lemma_dict2.tsv ../Penn-POS.tsv lemma_dict_POS2.tsv`
7. `head -n 20000 <glove.840B.300d.txt >glove.840B.300d_20000.txt`. wouldn't have been neccessary if step 2 came before step 1.
8. prepare vectors `python ../prepare.py glove.840B.300d_20000.txt processed_vectors.tsv 20000`
9. add annotations to vectors `python ../prepare-add_lemma.py lemma_dict_POS2.tsv processed_vectors.tsv processed_vectors_annot2.tsv 5000`
10. run SVM `python ../SVM.py processed_vectors_annot2.tsv processed_vectors_annot_SVM2.tsv rbf 1.0 scale | tee SVM2.log`
11. UMAP `python ../UMAP.py`
12. pure SVM vectors `<processed_vectors_annot_SVM2.tsv cut -f 206- | tail -n 5000 >processed_vectors_annot_SVM_pure2.tsv`
13. metadata `<processed_vectors_annot_SVM2.tsv cut -f -5 >processed_vectors_annot_SVM_metadata2.tsv`

# Visualizations

[Tensorflow Projector](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/VanaKraus/6fc323bced9a326ec444db47289032b3/raw/projector_config.json) ([config here](https://gist.github.com/VanaKraus/6fc323bced9a326ec444db47289032b3))

