# cs conll17

1. download w2vec <http://vectors.nlpl.eu/repository/> and unzip
2. take top 20000 `head -n 20001 <model.txt | tail -n 20000 >model_20000.txt`
3. take forms only `cut -d ' ' -f1 <model_20000.txt >model_20000_forms.txt`
4. call morphodita `../morphodita.py model_20000_forms.txt`
5. add header `echo -e "Word\tLemma\tTag\n$( cat annotated/model_20000_forms.txt )" >lemma_dict.tsv`
6. add POS labels to the dictionary `../prepare_lemma_dict.py lemma_dict.tsv - lemma_dict_POS.tsv`
7. prepare vectors `../prepare.py model_20000.txt processed_vectors.tsv 20000`
8. add annotations to vectors `../prepare-add_lemma.py lemma_dict_POS.tsv processed_vectors.tsv processed_vectors_annot.tsv 10000`
9. run SVM `python ../SVM.py processed_vectors_annot.tsv processed_vectors_annot_SVM.tsv rbf 1.0 scale | tee SVM.log`
10. UMAP `python ../UMAP.py`

Use the following to prepare the data for [Tensorflow projector](https://projector.tensorflow.org/).

11. pure SVM vectors `<processed_vectors_annot_SVM.tsv cut -f 6- | tail -n 10000 >processed_vectors_annot_SVM_pure.tsv`
12. metadata `<processed_vectors_annot_SVM.tsv cut -f -5 >processed_vectors_annot_SVM_metadata.tsv`
