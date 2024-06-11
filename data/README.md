Please download the dataset and put it in the `data/original` folder. The dataset can be downloaded from [http://www.di.uevora.pt/~pq/echr/](http://www.di.uevora.pt/~pq/echr/). 


And please cite the following paper if you use the dataset:

```bibtex
@inproceedings{poudyal-etal-2020-echr,
    title = "{ECHR}: Legal Corpus for Argument Mining",
    author = "Poudyal, Prakash  and
      Savelka, Jaromir  and
      Ieven, Aagje  and
      Moens, Marie Francine  and
      Goncalves, Teresa  and
      Quaresma, Paulo",
    editor = "Cabrio, Elena  and
      Villata, Serena",
    booktitle = "Proceedings of the 7th Workshop on Argument Mining",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.argmining-1.8",
    pages = "67--75",
    abstract = "In this paper, we publicly release an annotated corpus of 42 decisions of the European Court of Human Rights (ECHR). The corpus is annotated in terms of three types of clauses useful in argument mining: premise, conclusion, and non-argument parts of the text. Furthermore, relationships among the premises and conclusions are mapped. We present baselines for three tasks that lead from unstructured texts to structured arguments. The tasks are argument clause recognition, clause relation prediction, and premise/conclusion recognition. Despite a straightforward application of the bidirectional encoders from Transformers (BERT), we obtained very promising results F1 0.765 on argument recognition, 0.511 on relation prediction, and 0.859/0.628 on premise/conclusion recognition). The results suggest the usefulness of pre-trained language models based on deep neural network architectures in argument mining. Because of the simplicity of the baselines, there is ample space for improvement in future work based on the released corpus.",
}
```