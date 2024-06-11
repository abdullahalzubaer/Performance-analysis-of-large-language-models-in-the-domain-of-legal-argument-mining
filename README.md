# Performance analysis of large language models in the domain of legal argument mining

Accompanying code for the paper "Performance analysis of large language models in the domain of legal argument mining" [![DOI](https://zenodo.org/badge/DOI/10.3389/frai.2023.1278796.svg)](https://doi.org/10.3389/frai.2023.1278796)
by the authors:



      Abdullah Al Zubaer[1], Granitzer Michael[1], Mitrović Jelena[1,2]

      Affiliation:
      [1] Faculty of Computer Science and Mathematics, Chair of Data Science, University of Passau, Passau, Germany
      [2] Group for Human Computer Interaction, Institute for Artificial Intelligence Research and Development of Serbia, Novi Sad, Serbia


## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- You have a machine with Windows, Linux, or macOS.

## Installation

Tested on `Ubuntu 22.04.4 LTS` using `Python 3.8.19`

Follow these steps to get your development environment running:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yourprojectname.git
   cd yourprojectname
   ```

2. **Create a Conda environment:**
   Replace `frontier` with the name of the environment you want to use:
   ```bash
   conda create --name frontier python=3.8  
   ```

3. **Activate the Conda environment:**
   ```bash
   conda activate frontier
   ```

4. **Install required packages:**
   This command installs all the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
## Usage


> `Step 1`

### Dataset Extraction

1. Please download the dataset and place it in the `./data/original` folder. We only need `ECHR_Corpus.json`. The dataset can be downloaded from [http://www.di.uevora.pt/~pq/echr/](http://www.di.uevora.pt/~pq/echr/). 

2. Extract the dataset using the following command. This will extract the json file from the dataset and save it in the `./data/experiment_data/` folder as pickle files.

> It will also save supplementary data in the `./data/supplementary_data/` folder. This data can be used for further analysis and understanding of the dataset. Currently this one is not used in our work. But it has good potential for further analysis.


```bash
python extraction-echr.py --json_file_path ./data/original/ECHR_Corpus.json
```
> `Step 2`
### Argument Mining
Please follow the configurations outlined in Table 6 of the paper to provide the appropriate keywords as arguments to run the code using `GPT-3.5` and `GPT-4` as our LLMs. The embeddings, model outputs, and results will be automatically saved in the current directory with appropriate names and formats.

> Embeddings will be saved in `./embeddings` directory, model outputs will be saved in `./model_classification` directory, and results will be saved in `./results` directory.


To get help on the command line arguments, run:
```bash
python classification.py -h
```

sample command to run the code is as follows:

```bash
python classification.py -e text-embedding-ada-002 -m gpt-3.5-turbo -pt twoshot_with_instruction -mode similar -c premise

python classification.py -e text-embedding-ada-002 -m gpt-3.5-turbo -pt twoshot_with_instruction -mode similar -c conclusion
```

Note: Certain configurations is not allowed by nature of our experiment. For example, using `--embedding_model multi-qa-mpnet-base-dot-v1` but with`--mode random`. 

Concrete example: `python classification.py -e multi-qa-mpnet-base-dot-v1 -m gpt-3.5-turbo -pt twoshot_with_instruction -mode random -c premise` This is because we do not perform any embedding operation when we are choosing few shot examples randomly. This will throw an error. Therfore, please follow the configurations outlined in Table 6 of the paper to provide the appropriate keywords as arguments to run the code.

# License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).


# Citation

If you use our code and paper, please cite our paper as follows:

[![DOI](https://zenodo.org/badge/DOI/10.3389/frai.2023.1278796.svg)](https://doi.org/10.3389/frai.2023.1278796)


```bibtex
@ARTICLE{10.3389/frai.2023.1278796,
  
AUTHOR={Al Zubaer, Abdullah and Granitzer, Michael and Mitrović, Jelena},   
	 
TITLE={Performance analysis of large language models in the domain of legal argument mining},      
	
JOURNAL={Frontiers in Artificial Intelligence},      
	
VOLUME={6},           
	
YEAR={2023},      
	  
URL={https://www.frontiersin.org/articles/10.3389/frai.2023.1278796},       
	
DOI={10.3389/frai.2023.1278796},      
	
ISSN={2624-8212},   
   
ABSTRACT={Generative pre-trained transformers (GPT) have recently demonstrated excellent performance in various natural language tasks. The development of ChatGPT and the recently released GPT-4 model has shown competence in solving complex and higher-order reasoning tasks without further training or fine-tuning. However, the applicability and strength of these models in classifying legal texts in the context of argument mining are yet to be realized and have not been tested thoroughly. In this study, we investigate the effectiveness of GPT-like models, specifically GPT-3.5 and GPT-4, for argument mining via prompting. We closely study the model's performance considering diverse prompt formulation and example selection in the prompt via semantic search using state-of-the-art embedding models from OpenAI and sentence transformers. We primarily concentrate on the argument component classification task on the legal corpus from the European Court of Human Rights. To address these models' inherent non-deterministic nature and make our result statistically sound, we conducted 5-fold cross-validation on the test set. Our experiments demonstrate, quite surprisingly, that relatively small domain-specific models outperform GPT 3.5 and GPT-4 in the F1-score for premise and conclusion classes, with 1.9% and 12% improvements, respectively. We hypothesize that the performance drop indirectly reflects the complexity of the structure in the dataset, which we verify through prompt and data analysis. Nevertheless, our results demonstrate a noteworthy variation in the performance of GPT models based on prompt formulation. We observe comparable performance between the two embedding models, with a slight improvement in the local model's ability for prompt selection. This suggests that local models are as semantically rich as the embeddings from the OpenAI model. Our results indicate that the structure of prompts significantly impacts the performance of GPT models and should be considered when designing them.}
}
```


And please cite the following paper if you use their dataset in your research:

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
