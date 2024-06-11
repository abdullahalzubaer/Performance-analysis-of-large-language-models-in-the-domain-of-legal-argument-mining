import argparse
import json
import os
import pickle

import numpy as np
import openai
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split

from helper_functions import (
    encode_sentences_openai,
    encode_sentences_sentencetransformer,
    find_sentences_openai,
    find_sentences_sentencetransformer,
    initialize_model,
    record_response,
    select_random_sentences,
)
from prompt_template_conclusion import create_prompt_conclusion
from prompt_template_premise import create_prompt_premise

# this is saved as environment variable in the system.
openai.api_key = os.getenv("OPENAI_API_KEY_MICHAEL")

def main():
    parser = argparse.ArgumentParser(description="""
                                     
Developer: Abdullah Al Zubaer
Email: abdullahal.zubaer@uni-passau.de
Institution: University of Passau
                                     
Accompanying code for the paper titled: "Performance analysis of large language models in the domain of legal argument mining"
Please cite the paper if you use this code for your research.

And cite the following paper for the original ECHR corpus:
"ECHR: Legal Corpus for Argument Mining". More information of how to cite in the ./data/original/README.md file.

                                  
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-e', '--embedding_model',
                        type=str, required=True,
                        choices=['multi-qa-mpnet-base-dot-v1', 'text-embedding-ada-002', 'none'],
                        help='''Model identifier for the sentence transformer or OpenAI embedding model.
                        Choose from:
                        - multi-qa-mpnet-base-dot-v1: A sentence transformer model.
                        - text-embedding-ada-002: An OpenAI embedding model.
                        - none: Must be used when --mode is set to 'random' or --prompt_type is 'zeroshot_with_instruction'.
                        ''')
    
    parser.add_argument('-m','--model_name',
                        type=str, required=True,
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        help='''Model for text generation. 
                        Choose from:
                        - gpt-3.5-turbo: GPT-3.5 Turbo model.
                        - gpt-4: GPT-4 model.
                        ''')
    
    parser.add_argument('-pt','--prompt_type',
                        type=str, required=True,
                        choices=['zeroshot_with_instruction', 'twoshot_with_instruction', 
                                 'fourshot_with_instruction', 'eightshot_with_instruction'],
                        help='''Type of prompt to use for the model.
                        Choose from:
                        - zeroshot_with_instruction: No examples are given, only instructions.
                          --embedding_model must be 'none' if this is selected.
                          --mode must be 'none' if this is selected.
                        - twoshot_with_instruction: Two examples are given with instructions.
                        - fourshot_with_instruction: Four examples are given with instructions.
                        - eightshot_with_instruction: Eight examples are given with instructions.
                        ''')
    
    parser.add_argument('-mode', '--mode',
                        type=str, required=True,
                        choices=['similar', 'dissimilar', 'random', 'none'],
                        help='''Mode for selecting sentences to add as few-shot examples in the prompt.
                        Choose from:
                        - similar: Sentences semantically similar to the input.
                        - dissimilar: Sentences semantically dissimilar to the input.
                        - random: Randomly selected sentences.
                          --embedding_model must be 'none' if this is selected.
                        - none: No additional sentences are added.
                          Must be used if --prompt_type is 'zeroshot_with_instruction'.
                        ''')
    
    parser.add_argument('-c', '--component_to_classify',
                        type=str, required=True,
                        choices=['premise', 'conclusion'],
                        help='''The component to classify in the text.
                        Choose from:
                        - premise: Classify sentences as premise or non-premise.
                        - conclusion: Classify sentences as conclusion or non-conclusion.
                        ''')
    
    args = parser.parse_args()
    
    # Accessing arguments
    which_embedding_model = args.embedding_model
    current_model_name = args.model_name
    current_prompt_type = args.prompt_type
    mode = args.mode
    component_to_classify = args.component_to_classify
    
    # Example usage of the arguments
    print(f"Using embedding model: {which_embedding_model}")
    print(f"Text generation model: {current_model_name}")
    print(f"Prompt type: {current_prompt_type}")
    print(f"Mode: {mode}")
    print(f"Component to classify: {component_to_classify}")
    
    # Assertion checks for not allowed configurations
    assert not (which_embedding_model != 'none' and mode == 'random'), \
        "Invalid configuration: --embedding_model must be 'none' when --mode is 'random'."
    
    assert not (which_embedding_model != 'none' and current_prompt_type == 'zeroshot_with_instruction'), \
        "Invalid configuration: --embedding_model must be 'none' when --prompt_type is 'zeroshot_with_instruction'."
    
    assert not (mode == 'random' and which_embedding_model != 'none'), \
        "Invalid configuration: --mode must be 'none' when --embedding_model is 'none'."
    
    assert not (current_prompt_type == 'zeroshot_with_instruction' and mode != 'none'), \
        "Invalid configuration: --mode must be 'none' when --prompt_type is 'zeroshot_with_instruction'."
    
    assert not (which_embedding_model == 'none' and current_prompt_type in ['twoshot_with_instruction', 'fourshot_with_instruction', 'eightshot_with_instruction'] and mode == 'none'), \
        "Invalid configuration: --embedding_model cannot be 'none' when using shot prompt types with mode 'none'."
        
    assert not (which_embedding_model != 'none' and mode == 'none' and current_prompt_type != 'zeroshot_with_instruction'), \
        "Invalid configuration: --mode cannot be 'none' when --embedding_model is specified and --prompt_type is not 'zeroshot_with_instruction'."
        
    assert not (which_embedding_model == 'none' and current_prompt_type in ['twoshot_with_instruction', 'fourshot_with_instruction', 'eightshot_with_instruction'] and mode in ['similar', 'dissimilar']), \
        "Invalid configuration: --embedding_model cannot be 'none' when using shot prompt types with modes 'similar' or 'dissimilar'."



    
    # Main code starts here
    

    # Prompt types and corresponding settings\
    '''
    These sequence of True and False must be exactly the sequence expected in the prompt_template_{conclusions/premise}.py file.
    The binary is used in the create_prompt_{conclusion/premise}. 
    '''
    prompt_types = {
        "zeroshot_with_instruction": {"top_k": 0, "flags": [True, False, False, False]},
        "twoshot_with_instruction": {"top_k": 2, "flags": [False, True, False, False]},
        "fourshot_with_instruction": {"top_k": 4, "flags": [False, False, True, False]},
        "eightshot_with_instruction": {"top_k": 8, "flags": [False, False, False, True]}
    }
    
    # Make sure current_prompt_type is valid
    assert current_prompt_type in prompt_types, f"Invalid prompt type: {current_prompt_type}"

    # Set top_k and flags based on current_prompt_type
    top_k = prompt_types[current_prompt_type]["top_k"]
    zeroshot_with_instruction, twoshot_with_instruction, fourshot_with_instruction, eightshot_with_instruction = prompt_types[current_prompt_type]["flags"]

    # File paths for each component type
    component_paths = {
        "conclusion": './data/experiment_data/df_decomposed_conclusion.pickle',
        "premise": './data/experiment_data/df_decomposed_premise.pickle'
    }

    # Make sure component_to_classify is valid
    assert component_to_classify in component_paths, f"Invalid component to classify: {component_to_classify}"

    # Get path based on component_to_classify
    path_to_df_decomposed = component_paths[component_to_classify]    
    
    # This is an identifier for saving files with correct names based on
    # which classification task I am working on i.e. premise or conclusion
    conclusion_or_premise = path_to_df_decomposed.split('_')[-1].split('.')[0]
        
    with open(path_to_df_decomposed, 'rb') as f:
        df_conclusion_or_premise = pickle.load(f)
    
    # df_conclusion_or_premise = df_conclusion_or_premise[:200].copy() # for testing purposes reduce the size of the dataset
    
    
    df_conclusion_or_premise['article'] = df_conclusion_or_premise['article'].astype('category')

    kf = KFold(n_splits=5, shuffle=False)

    fold = 0

    # Initialize a dictionary to hold data for each fold
    data_dict = {}

    # Initializing a dictionary for storing the cross-validation result
    cv_dictionary = dict()
    
    
    for train_index, test_index in kf.split(df_conclusion_or_premise['article'].unique()):
        
        fold += 1

        print(f"========= FOLD: {fold} starting ==============")

        train_articles = df_conclusion_or_premise['article'].unique()[train_index]
        test_articles = df_conclusion_or_premise['article'].unique()[test_index]

        # Split the train_articles further into training and validation articles
        train_articles, validation_articles = train_test_split(train_articles, test_size=0.2, random_state=42)

        # Select the corresponding rows for each set
        train_df = df_conclusion_or_premise[df_conclusion_or_premise['article'].isin(train_articles)]
        validation_df = df_conclusion_or_premise[df_conclusion_or_premise['article'].isin(validation_articles)]
        test_df = df_conclusion_or_premise[df_conclusion_or_premise['article'].isin(test_articles)]
        
        # Store the dataframes in the dictionary
        # Not needed now, but if you want you can
        data_dict[f'fold_{fold}_for_{conclusion_or_premise}'] = {
            'train': train_df,
            'validation': validation_df,
            'test': test_df,
        }
        #CHANGEHERE#
        
        
        
        '''
        folder that will contain the embeddings:
        base_embeddings_folder = f"./embeddings/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}"
        embeddings_file = f"{base_embeddings_folder}" + f"/fold_{fold}_embeddings.npy"
        '''
        
        base_embeddings_folder = f"./embeddings/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}"

        # Ensure the base folder exists
        if which_embedding_model != "none" and mode not in ["random", "none"] and current_prompt_type != "zeroshot_with_instruction":
            if not os.path.exists(base_embeddings_folder):
                os.makedirs(base_embeddings_folder)
        
        
        if which_embedding_model != "none":
            # Therefore we need to encode the sentences using the embedding model
            # embeddings_file = f"./embeddings/embeddings_using_{which_embedding_model}_for_fold_{fold}_for_{conclusion_or_premise}.npy"
            embeddings_file = f"{base_embeddings_folder}/fold_{fold}_embeddings.npy"
            
            print("Encoding in progress...")
            
            '''
            One caveat is that the embeddings are created everytim we run this code. Although
            this is not optimized as the dataset is not changing so calculating themn one time would be sufficient.
            Just to keep it consistent with the code I used for the experiments done in the paper I am keeping it the 
            way it is. The overhead is not much given it is cheap to calculate the embedding using openai and its free
            for sentence-transformers.
            '''

            if which_embedding_model == "multi-qa-mpnet-base-dot-v1":
                # Initialize the model using the which_embedding_model
                initialize_model(which_embedding_model)
                train_df = encode_sentences_sentencetransformer(df=train_df,
                                                                column_name='argument',
                                                                embeddings_file=embeddings_file)
                
            elif which_embedding_model == "text-embedding-ada-002":
                train_df = encode_sentences_openai(df=train_df,
                                                column_name='argument',
                                                embeddings_file=embeddings_file,
                                                which_embedding_model=which_embedding_model)
                
            print("Encoding finished!")
            
        else:
            embeddings_file = "none" # this is not needed actually as we are not using this file anywhere.
        
        for index, row in test_df.iterrows():
            
            test_text = row['argument']
            test_text_ground_truth = row['argument_type']
            
            if mode == 'random':
                nshot_sentences = select_random_sentences(df=train_df,
                                                          top_k=top_k) # Although top_k is not needed for random selection but it is used to select n examples.
                
            elif mode == 'similar' or mode == 'dissimilar':    
            # if which_embedding_model == "multi-qa-mpnet-base-dot-v1" or which_embedding_model == "text-embedding-ada-002":
            #     i.e. mode is similar or dissimilar
                if which_embedding_model == "multi-qa-mpnet-base-dot-v1":
                    
                    nshot_sentences = find_sentences_sentencetransformer(query=test_text,
                                                                         df=train_df,
                                                                         top_k=top_k,
                                                                         mode=mode,
                                                                         embeddings_file=embeddings_file)
                elif which_embedding_model == "text-embedding-ada-002":
                    
                    nshot_sentences = find_sentences_openai(query=test_text,
                                                            df=train_df,
                                                            top_k=top_k,
                                                            mode=mode,
                                                            embeddings_file=embeddings_file,
                                                            which_embedding_model=which_embedding_model)
            # elif which_embedding_model == "none":
                    
            if component_to_classify == "conclusion":
                
                if current_prompt_type == "zeroshot_with_instruction":
                    nshot_sentences=() # require empty tuple for the prompt template.
                    prompt_created = create_prompt_conclusion(test_text,
                                                              *nshot_sentences,
                                                              zeroshot_with_instruction=zeroshot_with_instruction,
                                                              twoshot_with_instruction=twoshot_with_instruction,
                                                              fourshot_with_instruction=fourshot_with_instruction,
                                                              eightshot_with_instruction=eightshot_with_instruction)
                else:            
                    prompt_created = create_prompt_conclusion(test_text,
                                                            *nshot_sentences,
                                                            zeroshot_with_instruction=zeroshot_with_instruction,
                                                            twoshot_with_instruction=twoshot_with_instruction,
                                                            fourshot_with_instruction=fourshot_with_instruction,
                                                            eightshot_with_instruction=eightshot_with_instruction)
            elif component_to_classify == "premise":
                
                if current_prompt_type == "zeroshot_with_instruction":
                    nshot_sentences=()
                    prompt_created = create_prompt_premise(test_text,
                                                          *nshot_sentences,
                                                          zeroshot_with_instruction=zeroshot_with_instruction,
                                                          twoshot_with_instruction=twoshot_with_instruction,
                                                          fourshot_with_instruction=fourshot_with_instruction,
                                                          eightshot_with_instruction=eightshot_with_instruction)
                else:
                    prompt_created = create_prompt_premise(test_text,
                                                        *nshot_sentences,
                                                        zeroshot_with_instruction=zeroshot_with_instruction,
                                                        twoshot_with_instruction=twoshot_with_instruction,
                                                        fourshot_with_instruction=fourshot_with_instruction,
                                                        eightshot_with_instruction=eightshot_with_instruction)
                
            if component_to_classify == "conclusion":
                message=[
                    {"role": "system", "content": "You must reply with either 'conclusion' or 'non-conclusion'."},
                    {"role": "user", "content": prompt_created}
                ]
            elif component_to_classify == "premise":
                message=[
                    {"role": "system", "content": "You must reply with either 'premise' or 'non-premise'."},
                    {"role": "user", "content": prompt_created}
                ]  
            
            '''
            base_csv_folder = f"./model_classification/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}"
            csv_filename = f"{base_csv_folder}/echr_test_fold_{fold}_classification.csv"
            # Ensure the base folder exists
            if not os.path.exists(base_csv_folder):
                os.makedirs(base_csv_folder)
            '''
            # to contain the csv files for each fold with models predictions.
            base_csv_folder = f"./model_classification/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}"
            
            if not os.path.exists(base_csv_folder):
                os.makedirs(base_csv_folder)
            
            # Call the record_response function
            record_response(model_name = current_model_name,
                            messages = message,
                            #CHANGEHERE#
                            csv_filename = f"{base_csv_folder}/echr_test_fold_{fold}_classification.csv",

                            # csv_filename = f"./results/echr_test_fold_{fold}_for_{conclusion_or_premise}_using_{current_model_name}_for_{current_prompt_type}_embeddingmodel_{which_embedding_model}_mode_{mode}.csv",
                            prompt_type = current_prompt_type,
                            temperature = 0,
                            top_p = 1,
                            test_text = test_text,
                            groundtruth_label_for_test_text = test_text_ground_truth)
            
        #CHANGEHERE#   
        # Ground truth data extracting from the saved csv file in the current iteration
        y_true = pd.read_csv(f"{base_csv_folder}/echr_test_fold_{fold}_classification.csv", encoding="utf-8")['Ground Truth']
        
        #CHANGEHERE#
        # Model output data extracting from the saved csv file in the current iteration
        y_pred = pd.read_csv(f"{base_csv_folder}/echr_test_fold_{fold}_classification.csv", encoding="utf-8")['Only Model Output'].str.lower()
        # lowercase the model output for comparison (to avoid case sensitivity given that gpt models are non-deterministic)
        # this was not necessary when I wrote the paper, everything was in lowercase- but when refactroring the code for publication
        # I realized that the model output sometimes is not in lowercase. This does not change the results but it is good to have it in lowercase.
        # Generate classification report
        report = classification_report(y_true, y_pred, digits=3)
        
        report_dictionary = classification_report(y_true, y_pred, output_dict=True)
        #saving each classificaiton report for the current run
        
        # cv_dictionary[f'fold_{fold}'] = report_dictionary
        cv_dictionary[f'llm_{current_model_name}_em_{which_embedding_model}_mode_{mode}_nshots_{current_prompt_type}_c_{component_to_classify}_fold_{fold}'] = report_dictionary
        
        print(report)
        print("\n")
        print(f"========= Fold: {fold} Finished ==============\n")
    
    #CHANGEHERE#    
    '''
    base_result_folder = f"./results/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}/complete_evalution_5_fold"
    '''
    base_result_folder = f"./results/llm_{current_model_name}/em_{which_embedding_model}/mode_{mode}/nshots_{current_prompt_type}/c_{component_to_classify}/complete_evalution_5_fold"
    if not os.path.exists(base_result_folder):
        os.makedirs(base_result_folder)
        
        
    # saving the complete report for each run 
    with open(f"{base_result_folder}/evaluation_5_fold.json", "w") as f:
            json.dump(cv_dictionary, f, indent=4)  
            
    # >>> saving the summarized report for each run (mean and standard deviation) for 5 fold cross validation >>>      
    with open(f"{base_result_folder}/evaluation_5_fold.json", "r") as f:
        data = json.load(f)
        
    metrics = ["precision", "recall", "f1-score"]
    
    metric_results = {}

    categories = [component_to_classify, f"non-{component_to_classify}", "macro avg", "weighted avg"]
    
    for category in categories:
        metric_results[category] = {}
        for metric in metrics:
            metric_list = []
            for i in range(1, len(data) + 1):
                
                
                metric_list.append(data[f'llm_{current_model_name}_em_{which_embedding_model}_mode_{mode}_nshots_{current_prompt_type}_c_{component_to_classify}_fold_{i}'][category][metric])
                # metric_list.append(data[f'fold_{i}'][category][metric])
            mean_metric = np.mean(metric_list)
            stdev_metric = np.std(metric_list)

            metric_results[category][metric] = {
                "mean": mean_metric,
                "standard_deviation": stdev_metric
            }
            
    # Calculate average accuracy and standard deviation
    accuracy_list = [data[f'llm_{current_model_name}_em_{which_embedding_model}_mode_{mode}_nshots_{current_prompt_type}_c_{component_to_classify}_fold_{i}']['accuracy'] for i in range(1, len(data) + 1)]
    mean_accuracy = np.mean(accuracy_list)
    stdev_accuracy = np.std(accuracy_list)
    
    metric_results["accuracy"] = {
        "mean": mean_accuracy,
        "standard_deviation": stdev_accuracy
    }
    
    # Write tbhe result to the output JSON file
    output_file_path = f"{base_result_folder}/evaluation_5_fold_summarized.json"
    with open(output_file_path, "w") as output_file:
        json.dump(metric_results, output_file, indent=4)
        
    print("Summarized metric results have been saved to", output_file_path)
    # <<< saving the summarized report for each run (mean and standard deviation) for 5 fold cross validation <<<
if __name__ == "__main__":
    exit(main())
