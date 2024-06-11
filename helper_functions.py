# utility
##########################################################################################################

import csv
import datetime
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import openai
import torch
from sentence_transformers import SentenceTransformer
from tenacity import (  # , before_sleep_log # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# import logging
# Initialize the SentenceTransformer model
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# logging.basicConfig(filename='openai_api_call.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logging.basicConfig(level=logging.INFO)  # Setup logging



# Define function to initialize model
def initialize_model(model_name):
    global embedder
    embedder = SentenceTransformer(model_name)

def encode_sentences_sentencetransformer(df, column_name, embeddings_file):
    
    # print(embedder)

    
    """
    Converts a column of sentences from a DataFrame into sentence embeddings, saves the embeddings to a file,
    and adds an 'embedding_index' to the DataFrame that maps each sentence to its corresponding embedding.

    Args:
        df (pandas.DataFrame): DataFrame containing the sentences to be embedded.
        column_name (str): The column in the DataFrame that contains the sentences.
        embeddings_file (str): Path to the file where the sentence embeddings should be saved.

    Returns:
        df (pandas.DataFrame): The original DataFrame with an additional 'embedding_index' column. 

    """
    
    # to get rid of a pandas warning i am using df.copy()
    '''
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['embedding_index'] = range(len(df))
    '''
    df = df.copy()
    
    # Convert the texts to embeddings
    embeddings = embedder.encode(df[column_name].tolist(), convert_to_tensor=True)

    # Convert tensor to numpy array for efficient storage
    embeddings = embeddings.cpu().numpy()

    # Save embeddings to a separate structure (numpy array)
    np.save(embeddings_file, embeddings)

    # In the DataFrame, just store the index of the sentence embeddings
    df['embedding_index'] = range(len(df))
    
    # Set 'embedding_index' as the DataFrame's index
    df.set_index('embedding_index', inplace=True)

    return df

def find_sentences_sentencetransformer(query, df, top_k, mode, embeddings_file):
    
    """
    Finds sentences in a given DataFrame based on a query. Depending on the mode, it returns 
    sentences that are most similar, most dissimilar or random.

    Parameters:
    query (str): The query sentence.
    df (pd.DataFrame): The DataFrame which contains the sentences.
    top_k (int): The number of sentences to return.
    mode (str): The mode of operation - 'similar', 'dissimilar', or 'random'.
                'similar': returns the top_k sentences most similar to the query.
                'dissimilar': returns the top_k sentences most dissimilar to the query.
                'random': returns top_k randomly picked sentences.
    embeddings_file (str): The file where the sentence embeddings are stored.

    Returns:
    sentences (list of tuples): Each tuple contains a sentence from the DataFrame, 
                                its label and its similarity score to the query 
                                (None for 'random' mode).

    Raises:
    ValueError: If the mode is not 'similar', 'dissimilar' or 'random'.
    """
    
    corpus_embeddings = torch.tensor(np.load(embeddings_file))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    if mode == 'similar' or mode == 'dissimilar':
        # Normalize the embeddings so that they have unit length
        query_embedding = query_embedding / torch.norm(query_embedding)
        corpus_embeddings = corpus_embeddings / torch.norm(corpus_embeddings, dim=1, keepdim=True)

        # Ensure query_embedding is a 2D tensor for matrix multiplication
        query_embedding = query_embedding.unsqueeze(0)

        # Calculate cosine similarity
        cos_scores = torch.mm(query_embedding, corpus_embeddings.transpose(0, 1))[0]

        # Depending on the mode, find the most similar or dissimilar sentences
        if mode == 'similar':
            top_results = torch.topk(cos_scores, k=top_k)
        elif mode == 'dissimilar':
            top_results = torch.topk(-cos_scores, k=top_k)

    # elif mode == 'random':
    #     indices = random.sample(range(len(df)), top_k)
    else:
        raise ValueError("Invalid mode. Choose from 'similar', 'dissimilar'.")

    # print("\n\n======================\n\n")
    # print("Query:", query)#new
    # print(top_results)
    sentences = []
    # if mode == 'random':
    #     for idx in indices:
    #         sentence = df.loc[idx]['argument']
    #         label = df.loc[idx]['argument_type']
    #         sentences.append((sentence, label, None))
    # else:
    # print(df)
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        score = score.item()
        sentence = df.loc[idx]['argument']
        label = df.loc[idx]['argument_type']
        sentences.append((sentence, label, score))
    # print(sentences) #new
    return sentences

# def select_random_sentences_test(df, top_k):
#     '''Testing why this approach gives error but the select_random_sentences() function works fine.'''
#     indices = random.sample(range(len(df)), top_k)
#     sentences = []
#     print(df)
#     for idx in indices:
#         sentence = df.loc[idx]['argument']
#         label = df.loc[idx]['argument_type']
#         sentences.append((sentence, label, None))
    
#     return sentences

def select_random_sentences(df, top_k, mode='random'):
    if mode == 'random':
        if 'argument' not in df.columns or 'argument_type' not in df.columns:
            raise KeyError("The DataFrame does not contain the required columns 'argument' and 'argument_type'.")
        
        if top_k > len(df):
            raise ValueError("top_k is larger than the number of rows in the DataFrame.")
        '''
        In find_sentences_sentencetransformer() function we could use the range(len(df)) to get the indices of the DataFrame.
        because the indices was actually altered in the encode_sentences_sentencetransformer() function. But here we are using
        original dataframe without any alteration. So we have to use the list(df.index) to get the indices of the DataFrame.
        As there is no 'embedding_index' column in the DataFrame that starts from 0 to len(df) like in the case of encode_sentences_sentencetransformer() function..
        
        '''
        indices = random.sample(list(df.index), top_k)  # Sample from the DataFrame's index
        # print("Sampled Indices:", indices)

        sentences = []
        for idx in indices:
            # print("Current Index:", idx)
            sentence = df.loc[idx, 'argument']
            label = df.loc[idx, 'argument_type']
            sentences.append((sentence, label, None))
    
        return sentences
    else:
        raise ValueError("Invalid mode. Choose 'random'.")


#>>> for openai embedding >>>
def get_embedding_from_openai(text: str, model , retry_limit=20):
    # print(f"from get_embedding_from_openai() function using {model}")
    try_count = 0
    while try_count < retry_limit:
        try:
            response = openai.Embedding.create(input=text, model=model)
            # If the response is successful, return the response
            return response["data"][0]["embedding"]

        except openai.error.RateLimitError as e:
            print("Rate limit exceeded. Waiting for 3 minutes before retrying...")
            time.sleep(180)  # Wait for 3 minutes
            try_count += 1  # Increment the try count and retry

        except Exception as e:
            print(f"This Exception: {e}\nTrying After 3 min")
            time.sleep(180)  # Wait for 3 minutes
            try_count += 1  # Increment the try count and retry

    print(f"Reached the maximum number of retries: {retry_limit}. Unable to get a successful response.")
    return None


def encode_sentences_openai(df, column_name, embeddings_file, which_embedding_model):
    # print(f"from encode_sentences_openai() function using {which_embedding_model}")
    df = df.copy()
    
    # Convert the texts to embeddings
    embeddings = [get_embedding_from_openai(text, which_embedding_model) for text in df[column_name].tolist()]

    # Convert list of embeddings to numpy array for efficient storage
    embeddings = np.array(embeddings)

    # Save embeddings to a separate structure (numpy array)
    np.save(embeddings_file, embeddings)

    # In the DataFrame, just store the index of the sentence embeddings
    df['embedding_index'] = range(len(df))
    
    # Set 'embedding_index' as the DataFrame's index
    df.set_index('embedding_index', inplace=True)

    return df




def find_sentences_openai(query, df, top_k, mode, embeddings_file, which_embedding_model):
    
    # print(f"From find_sentences_openai() function - using  {which_embedding_model}")

    corpus_embeddings = torch.tensor(np.load(embeddings_file))
    
    #changedhereforopenai
    query_embedding = get_embedding_from_openai(query, which_embedding_model)  # Use the same function to generate the query embedding
    query_embedding = torch.tensor(query_embedding).double()  # Convert to double precision i.e. float64
    # since when you save the embedding from openai using numpy, numpy convert it to float64 by deafult. Therefore when you
    # are getting the embedding fo a text directly from openai they actualyl give you float32

    
    if mode == 'similar' or mode == 'dissimilar':
        # Normalize the embeddings so that they have unit length
        query_embedding = query_embedding / torch.norm(query_embedding)
        corpus_embeddings = corpus_embeddings / torch.norm(corpus_embeddings, dim=1, keepdim=True)
        
        # Ensure query_embedding is a 2D tensor for matrix multiplication
        query_embedding = query_embedding.unsqueeze(0)

        # Calculate cosine similarity
        cos_scores = torch.mm(query_embedding, corpus_embeddings.transpose(0, 1))[0]

        # Depending on the mode, find the most similar or dissimilar sentences
        if mode == 'similar':
            top_results = torch.topk(cos_scores, k=top_k)
        elif mode == 'dissimilar':
            top_results = torch.topk(-cos_scores, k=top_k)

    # elif mode == 'random':
    #     indices = random.sample(range(len(df)), top_k)
    else:
        raise ValueError("Invalid mode. Choose from 'similar', 'dissimilar'.")

    # print("\n\n======================\n\n")
    # print("Query:", query)

    sentences = []
    # if mode == 'random':
    #     for idx in indices:
    #         sentence = df.loc[idx]['argument']
    #         label = df.loc[idx]['argument_type']
    #         sentences.append((sentence, label, None))
    # else:
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        score = score.item()
        sentence = df.loc[idx]['argument']
        label = df.loc[idx]['argument_type']
        sentences.append((sentence, label, score))

    return sentences

#<<< for iopenai embedding <<<

'''
@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(20), before_sleep=before_sleep_log(logging, logging.INFO))
def chat_with_openai(model_name, messages, temperature, top_p):
    """
    This function initiates a chat with the specified OpenAI model.

    Args:
        model_name (str): The name of the model to be used.
        messages (List[Dict[str, str]]): The list of messages to be sent. Each message is a dictionary with 'role' and 'content'.
        temperature (float): The 'temperature' parameter to be used for the chat completion. Lower values (close to 0) make the output more deterministic, while higher values (close to 1) make it more random.
        top_p (float): The 'top_p' parameter to be used for the chat completion. A float between 0 and 1. 

    Returns:
        dict or str: The chat completion response if successful, otherwise an error message.
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    # If the response is successful, return the response
    return response
'''







#'''
@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(20))
def chat_with_openai(model_name, messages, temperature, top_p):
    """
    This function initiates a chat with the specified OpenAI model.

    Args:
        model_name (str): The name of the model to be used.
        messages (List[Dict[str, str]]): The list of messages to be sent. Each message is a dictionary with 'role' and 'content'.
        temperature (float): The 'temperature' parameter to be used for the chat completion. Lower values (close to 0) make the output more deterministic, while higher values (close to 1) make it more random.
        top_p (float): The 'top_p' parameter to be used for the chat completion. A float between 0 and 1. 

    Returns:
        dict or str: The chat completion response if successful, otherwise an error message.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )
        # If the response is successful, return the response
        return response

    except openai.error.RateLimitError as e:
        print("Rate limit exceeded. Retrying...")
        raise e  # Retrying is handled by `tenacity` now, so re-raise the exception

    except Exception as e:
        print(f"This Exception: {e}\nRetrying...")
        raise e  # Retrying is handled by `tenacity` now, so re-raise the exception
#'''



def record_response(model_name: str,
                    messages: List[Dict[str, str]],
                    csv_filename: str, 
                    prompt_type: str,
                    top_p: float,
                    temperature: float,
                    test_text: str,
                    groundtruth_label_for_test_text: str):

    
    """
    Records a response from the OpenAI model in a CSV file.

    Args:
        model_name (str): The name of the model to be used.
        messages (List[Dict[str, str]]): The list of messages to be sent. Each message is a dictionary with 'role' and 'content'.
        csv_filename (str): The filename of the CSV file where the responses will be recorded.
        prompt_type (str): The type of prompt used. Must be one of ['zero-shot', '2-shot', '4-shot', 'cot', 'tot'].
        top_p (float): The 'top_p' parameter to be used for the chat completion.
        temperature (float): The 'temperature' parameter to be used for the chat completion.
        test_text (str): The test text to be classified by the model.
        groundtruth_label_for_test_text (str): The ground truth label for the test text.

    Raises:
        AssertionError: If any of the provided inputs are not valid.

    Notes:
        - The function checks if the CSV file does not exists, if it does not then it creates one.
        - It calls the `chat_with_openai` function to get a response from the OpenAI model.
        - After getting the response, it adds a row to the CSV file with all the data
    """
    
    
    # valid_prompt_types = ['zeroshot_with_instruction',
    #                       # '2-shot-with-instruction', 
    #                       'fourshot_with_instruction',
    #                       # 'cot-with-instruction',
    #                       # 'tot-with-instruction',
    #                       # 'zero-shot-no-instruction',
    #                       # '2-shot-no-instruction', 
    #                       # '4-shot-no-instruction',
    #                       # 'cot-no-instruction',
    #                       # 'tot-no-instruction'
    #                      ]
    
    
    
    model_name_types = ['gpt-3.5-turbo',
                        'gpt-4',
                        'gpt-4-32k',
                        'gpt-3.5-turbo-16k']
    
    # assert prompt_type in valid_prompt_types, f"Invalid prompt_type. Must be one of {valid_prompt_types}"
    assert model_name in model_name_types, f"Invalid model_name. Must be one of {model_name_types}"



    if not os.path.exists(csv_filename):
        # os.remove(csv_filename)
        with open(csv_filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Complete Input", "Prompt Type", "Complete Response", "Only Model Output", "Date and Time", "Model",
                            "Hyperparameter","Text To Classify", "Ground Truth"])

    response = chat_with_openai(model_name, messages, temperature, top_p)
    
    
    hyperparameter = {'Temperature': temperature, 'top_p': top_p}
    now = datetime.datetime.now()
    
    
    # make sure the response is valid before accessing its content
    if isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
        model_output = response['choices'][0]['message']['content']
    else:
        model_output = None

    data = [json.dumps(messages),
            prompt_type,
            json.dumps(response),
            model_output,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            json.dumps(hyperparameter),
            test_text,
            groundtruth_label_for_test_text,
           ]

    # Open the CSV file for writing
    with open(csv_filename, "a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the data rows`
        writer.writerow(data)
