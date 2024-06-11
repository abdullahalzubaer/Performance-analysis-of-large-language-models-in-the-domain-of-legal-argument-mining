import argparse
import json
import os
import pickle
import time

import pandas as pd


def process_echr_corpus_to_dataframe(json_file_path,
                                     csv_file_path=None,
                                     pickle_file_path=None):
    """
    Processes a JSON file containing articles from the ECHR Corpus and structures the data into a DataFrame, which organizes premises and conclusions by article and argument group. The function can also optionally save this DataFrame to a CSV file or serialize it to a pickle file.

    The JSON file should have a structure where each article contains multiple argument groups, each consisting of premises and a conclusion. These are linked by IDs to specific clauses within the article's text. This function parses through each article, extracts the relevant text snippets for premises and conclusions based on these IDs, and organizes them into a structured format.

    Parameters:
        json_file_path (str): The file path for the JSON data file to be processed.
        csv_file_path (str, optional): The file path for the CSV file to save the resulting DataFrame. If None, the DataFrame will not be saved to a file.
        pickle_file_path (str, optional): The file path for the pickle file to save the serialized DataFrame. If None, the DataFrame will not be serialized to a file.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data with columns 'article', 'group-number', 'premise', and 'conclusion'. Each row corresponds to an argument group within an article, listing the associated premises and the conclusion.

    Example:
        # Process data, save to CSV, and serialize to pickle
        df = process_echr_corpus_to_dataframe('ECHR_Corpus.json', 'output.csv', 'output.pkl')

        # Process data without saving to CSV or serializing to pickle
        df = process_echr_corpus_to_dataframe('ECHR_Corpus.json')
    """
    # Load JSON data from file
    with open(json_file_path, 'rb') as f:
        data = json.load(f)

    # Process data into a structured dictionary
    final_dict = {}
    for i, article in enumerate(data):
        article_key = f'article_{i}'
        final_dict[article_key] = {}
        group_num = 0

        for arg_group in article['arguments']:
            group_key = f'group_number_{group_num}'
            final_dict[article_key][group_key] = {'premise': [], 'conclusion': []}

            for clause_id in arg_group['premises']:
                clause = next((c for c in article['clauses'] if c['_id'] == clause_id), None)
                if clause:
                    start, end = clause['start'], clause['end']
                    text = article['text'][start:end]
                    final_dict[article_key][group_key]['premise'].append(text)

            conclusion_id = arg_group['conclusion']
            conclusion = next((c for c in article['clauses'] if c['_id'] == conclusion_id), None)
            if conclusion:
                start, end = conclusion['start'], conclusion['end']
                text = article['text'][start:end]
                final_dict[article_key][group_key]['conclusion'].append(text)

            group_num += 1

    # Convert the dictionary to a DataFrame
    data = []
    for article, groups in final_dict.items():
        for group_num, content in groups.items():
            if not isinstance(content['premise'], list):
                content['premise'] = [content['premise']]
            if not isinstance(content['conclusion'], list):
                content['conclusion'] = [content['conclusion']]
            data.append([article, group_num, content['premise'], content['conclusion']])

    df = pd.DataFrame(data, columns=['article', 'group-number', 'premise', 'conclusion'])

    
    # create directory if it does not exist
    
    base_dir = "./data/supplementary_data/"
    if not os.path.exists(base_dir):
        os.makedirs(os.path.dirname(base_dir))
    
    # Optionally save the DataFrame to a CSV file
    if csv_file_path:
        df.to_csv(csv_file_path, index=False)

    # Optionally serialize the DataFrame to a pickle file
    if pickle_file_path:
        with open(pickle_file_path, 'wb') as p:
            pickle.dump(df, p)

    return df

def decompose_arguments(df_group):
    """
    Decomposes a dataframe of arguments into separate premises and conclusions, 
    and combines them into a single dataframe with appropriate ordering.

    Parameters:
    df_group (pd.DataFrame): Input dataframe containing 'article', 'group-number', 'premise', and 'conclusion' columns.

    Returns:
    pd.DataFrame: Decomposed dataframe with arguments properly ordered.
    """
    # Create separate dataframes for 'premise' and 'conclusion'
    df_group['orig_index'] = df_group.index

    # Process 'premise' data
    df_premise = (
        df_group[['article', 'group-number', 'premise', 'orig_index']]
        .explode('premise')
        .reset_index(drop=True)
        .assign(argument_type='premise')
        .rename(columns={'premise': 'argument'})
    )
    df_premise['sequence'] = df_premise.groupby(['article', 'group-number']).cumcount()

    # Calculate group sizes for df_premise
    group_sizes = (
        df_premise.groupby(['article', 'group-number'])
        .size()
        .reset_index(name='group_size')
    )

    # Process 'conclusion' data
    df_conclusion = (
        df_group[['article', 'group-number', 'conclusion', 'orig_index']]
        .explode('conclusion')
        .reset_index(drop=True)
    )

    # Merge df_conclusion with group_sizes and process further
    df_conclusion = (
        df_conclusion.merge(group_sizes, on=['article', 'group-number'], how='left')
        .assign(argument_type='conclusion')
        .rename(columns={'conclusion': 'argument'})
    )
    df_conclusion['sequence'] = df_conclusion.groupby(['article', 'group-number']).cumcount() + df_conclusion['group_size']

    # Drop the 'group_size' column
    df_conclusion.drop(columns=['group_size'], inplace=True)

    # Concatenate the two dataframes
    df_decomposed = pd.concat([df_premise, df_conclusion], ignore_index=True)

    # Sort by the original index and sequence, then drop these columns
    df_decomposed = (
        df_decomposed.sort_values(['orig_index', 'sequence'])
        .drop(columns=['orig_index', 'sequence'])
        .reset_index(drop=True)
    )

    return df_decomposed

def prepare_for_binary_classifiers(df_decomposed,
                                   conclusion_pickle_path='./data/experiment_data/df_decomposed_conclusion.pickle',
                                   premise_pickle_path='./data/experiment_data/df_decomposed_premise.pickle'):
    """
    Transforms the 'argument_type' column of the input DataFrame for binary classification by creating two variants:
    1. Changing 'premise' to 'non-conclusion' for one classifier.
    2. Changing 'conclusion' to 'non-premise' for another classifier.

    Each transformed DataFrame is saved as a separate pickle file.

    Parameters:
        df_decomposed (pandas.DataFrame): The DataFrame containing decomposed arguments.
        conclusion_pickle_path (str): Path where the DataFrame for the conclusion to non-premise transformation will be saved.
        premise_pickle_path (str): Path where the DataFrame for the premise to non-conclusion transformation will be saved.

    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame is transformed for the conclusion classifier and the second for the premise classifier.
    """
    # Change premise to Non-conclusion
    def change_premise_to_not_conclusion(argument_type):
        return 'non-conclusion' if argument_type == 'premise' else argument_type

    # Change conclusion to Non-premise
    def change_conclusion_to_not_premise(argument_type):
        return 'non-premise' if argument_type == 'conclusion' else argument_type

    # Transform for conclusion classifier
    df_decomposed_conclusion = df_decomposed.copy()
    df_decomposed_conclusion['argument_type'] = df_decomposed_conclusion['argument_type'].apply(change_premise_to_not_conclusion)


    # Transform for premise classifier
    df_decomposed_premise = df_decomposed.copy()
    df_decomposed_premise['argument_type'] = df_decomposed_premise['argument_type'].apply(change_conclusion_to_not_premise)


    
    base_dir = "./data/experiment_data/"
    if not os.path.exists(base_dir):
        os.makedirs(os.path.dirname(base_dir))
        
    # Save the dataframes as pickle objects
    with open(conclusion_pickle_path, 'wb') as f:
        pickle.dump(df_decomposed_conclusion, f)
    with open(premise_pickle_path, 'wb') as f:
        pickle.dump(df_decomposed_premise, f)
        
    print(f"Successfully saved processed ECHR dataset and saved the dataframes as pickle objects.\n\nConclusion: {conclusion_pickle_path}\nPremise:{premise_pickle_path}")

    # return df_decomposed_conclusion, df_decomposed_premise


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="""
Process ECHR Corpus JSON file into a DataFrame save it to ./data/experiment_data 
and save the supplementary data to ./data/supplementary_data/ as CSV or pickle.

Developer: Abdullah Al Zubaer
Email: abdullahal.zubaer@uni-passau.de
Institution: University of Passau
                                     
Accompanying code for the paper titled: "Performance analysis of large language models in the domain of legal argument mining"
Please cite our paper if you use this code for your research.

And cite the following paper for the original ECHR corpus:
"ECHR: Legal Corpus for Argument Mining". More information of how to cite in the ./data/original/README.md file.

""",formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--json_file_path",
                        type=str, default= "./data/original/ECHR_Corpus.json",
                        help="The file path for the JSON data file. Default: %(default)s.")
    
    parser.add_argument("--csv_file_path",
                        type=str, default="./data/supplementary_data/echr_am_supplementary.csv",
                        help="""Optional file path to save the resulting supplemetnary DataFrame as a CSV file. 
                        
                        Note: the csv file will have premise and conclusion as a string of list therefore it is recommended to use pickle file. Or else
                        you can use the following command to convert the string of list to list and then do the further processing:
                        
                        df = pd.read_csv("./data/supplementary_data/echr_am_supplementary.csv")
                        df['premise'] = df['premise'].apply(eval)
                        df['conclusion'] = df['conclusion'].apply(eval)""")
    
    parser.add_argument("--pickle_file_path",
                        type=str, default="./data/supplementary_data/echr_am_supplementary.pickle",
                        help="Optional file path to serialize the supplemetnary DataFrame as a pickle file. Default: %(default)s.")

    # Parse arguments
    args = parser.parse_args()

    # Process the file

    print("Processing the ECHR corpus...")
    time.sleep(2)
    df = process_echr_corpus_to_dataframe(json_file_path=args.json_file_path,
                                        csv_file_path=args.csv_file_path,
                                        pickle_file_path=args.pickle_file_path)
    
    print("Decomposing arguments and preparing data for binary classifiers...")
    time.sleep(2)
    
    df_decomposed = decompose_arguments(df_group=df)
    prepare_for_binary_classifiers(df_decomposed)

    

if __name__ == "__main__":
    exit(main())
