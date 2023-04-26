import math 
import numpy as np
import torch
from collections import Counter
from data import get_training_data
from sentence_transformers import SentenceTransformer, util
import nltk
import pandas as pd
import os

def convert_to_string_list(list):
    res = []
    for sen in list:
        res.append(" ".join(sen))
    return res


def calculate_cosine_similarity(docs1, docs2, mode='mean'):
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    docs1 = convert_to_string_list(docs1)
    docs2 = convert_to_string_list(docs2)

    embedding_1 = model.encode(docs1, convert_to_tensor=True, show_progress_bar=True, device='cuda')
    embedding_2 = model.encode(docs2, convert_to_tensor=True, show_progress_bar=True, device='cuda')

    scores = []
    max_scores = []
    min_scores = []
    batch_size = 64
    for emb1 in embedding_1:
        temp_scores = torch.tensor([]).to('cuda')
        temp_max_score = 0
        temp_min_score = 1
        for i in range(0,len(embedding_2),batch_size):
            emb_batch = embedding_2[i:i + batch_size]
            batch_score = util.pytorch_cos_sim(emb1, emb_batch)
            
            temp_scores = torch.cat((temp_scores, batch_score), dim=1)

            max = batch_score.max().item()
            min = batch_score.min().item()
            if max > temp_max_score:
                temp_max_score = max
            if min < temp_min_score:
                temp_min_score = min
        
        # append mean of temp_scores to scores
        scores.append(temp_scores.mean().item())
        # append max of temp_scores to max_scores
        max_scores.append(temp_max_score)
        # append min of temp_scores to min_scores
        min_scores.append(temp_min_score)
    if (mode == 'mean'):
        return([np.mean(scores), np.mean(max_scores), np.mean(min_scores)])
    else:
        return(scores)

# Not used atm
def cosine_sim_test(sentence1, sentence2):
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
    
    tokens1 = tokenizer.encode(sentence1, add_special_tokens=True, return_tensors='pt')
    tokens2 = tokenizer.encode(sentence2, add_special_tokens=True, return_tensors='pt')
    
    ## Generate embeddings for the tokens using the pre-trained BERT model

    with torch.no_grad():
        embeddings1 = model(tokens1)[0].squeeze(0) # remove batch dimension and take the first (and only) element
        embeddings2 = model(tokens2)[0].squeeze(0)
        
        cos_sim = cosine_similarity(embeddings1.detach().numpy(), embeddings2.detach().numpy())

    print("KB-bert cosine sim:", cos_sim[0][0])

#Calculate the euclidean distance between two documents
#A higher distance means that the two documents are more different
def euclidean_distance(doc1, doc2):
    """Calculate the euclidean distance between two documents
    A higher distance means that the two documents are more different
    
    Args:
        doc1 (list): A list of tokens
        doc2 (list): A list of tokens
            
    Returns:
        float: The euclidean distance between the two documents
    """

    #Count the number of occurences of each token
    count1 = Counter(doc1)
    count2 = Counter(doc2)

    #Get the union of the two documents
    union = set(doc1).union(set(doc2))

    #Calculate the euclidean distance
    distance = 0
    for token in union:
        distance += (count1[token] - count2[token])**2

    return math.sqrt(distance)


def calculate_bleu_or_euclidean(docs1, docs2, method):
    """Calculate the BLEU score using nltk or Euclidean distance between docs1 and docs2

    If the two documents are identical, the BLEU score will be 1.0
    If the two documents are completely different, the BLEU score will be 0.0
    
    Args:
        docs1 (list): A list of a list of tokens
        docs2 (list): A list of a list of tokens
            
    Returns:
        list: The mean, Avg_max and Avg_min scores between the two documents
    """
    total_scores = []
    max_scores = []
    min_scores = []
    for doc1 in docs1:
        temp_scores = []
        temp_max_score = 0
        temp_min_score = 1
        for doc2 in docs2:
            if method == 'bleu':
                score = nltk.translate.bleu_score.sentence_bleu([doc2], doc1, weights=(1, 0, 0, 0))
            elif method == 'euclidean':
                score = euclidean_distance(doc1, doc2)
            if score > temp_max_score:
                temp_max_score = score
            if score < temp_min_score:
                temp_min_score = score
            temp_scores.append(score)
        total_scores.append(sum(temp_scores) / len(temp_scores))
        max_scores.append(temp_max_score)
        min_scores.append(temp_min_score)

    return([np.mean(total_scores), np.mean(max_scores), np.mean(min_scores)])


def create_dir(path):
    if not os.path.exists(path):
        print("Creating a new dir for saving results..")
        os.makedirs(path, exist_ok=True)

def save_result_file(subfolder, filename, result):
    if(os.environ.get("RESULT_DIR") == None):
        print("Please set the RESULT_DIR environment variable.")
        return

    path = os.environ.get("RESULT_DIR") + subfolder + "/"

    #Try to create the folder if it doesn't exist
    create_dir(path)

    #create file path
    filepath = path+filename

    # write dataframe to csv file
    result.to_csv(filepath, index=False)


def print_dataset_similarity_scores(metric='all'):
    percentages = [25, 50, 75, 100]
    for percentage in percentages:
        X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=percentage)

        for dataset in ['test']:
            scores = {}
            hyp = X_val if dataset == 'val' else X_test
            ref = X_train
            if metric == 'cosine' or metric == 'all':
                cos_sim_scores = calculate_cosine_similarity(hyp, ref)
                scores['Cos_sim'] = cos_sim_scores
            if metric == 'bleu' or metric == 'all':
                bleu_scores = calculate_bleu_or_euclidean(hyp, ref, method='bleu')
                scores['BLEU'] = bleu_scores
            if metric == 'euclidean' or metric == 'all':
                euclidean_scores = calculate_bleu_or_euclidean(hyp, ref, method='euclidean')
                scores['Euclidean'] = euclidean_scores
            
            results = pd.DataFrame(columns=['Metric', 'Mean', 'Max', 'Min'])
            for key, value in scores.items():
                results = pd.concat([results, pd.DataFrame([[key, round(value[0], 4), round(value[1], 4), round(value[2], 4)]], columns=['Metric', 'Mean', 'Max', 'Min'])], ignore_index=True)

            print(f'\nWriting scores for {percentage}% of {dataset}-dataset to file..')
            filename = f'similarity_scores_{dataset}_data_{percentage}.csv'
            save_result_file('similarity', filename, results)


def compare_dataset(X1,X2,metrics=["cosine","bleu","euclidean"]):
    #Check that the two datasets have the same length
    if len(X1) != len(X2):
        raise ValueError("The two datasets have different lengths")

    scores = []

    for i in range(len(X1)):
        #Get the current two sentences
        x1 = X1[i]
        x2 = X2[i]

        #Calculate the similarity scores for the two sentences
        score = {}

        if  'cosine' in metrics:
            cos_sim_score = calculate_cosine_similarity([x1], [x2])
            cos_sim_score = cos_sim_score[0]
            score['cosine'] = cos_sim_score
        if 'bleu' in metrics:
            bleu_score = nltk.translate.bleu_score.sentence_bleu([x1], x2, weights=(1, 0, 0, 0))
            score['bleu'] = bleu_score
        if 'euclidean' in metrics:
            euclidean_score = euclidean_distance(x1, x2)
            score['euclidean'] = euclidean_score
        
        scores.append(score)
        
    return scores
    

def generate_scores_per_sentence(metric='all'):
    percentages = [25, 50, 75, 100]
    for percentage in percentages:
        X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=percentage)
        hyp = X_test
        ref = X_train
        scores = {}
        print('hyp length: ', len(hyp))
        if metric == 'cosine' or metric == 'all':
            cos_sim_scores = calculate_cosine_similarity(hyp, ref, mode='per_sentence')
            # round all values to 4 decimals
            cos_sim_scores = [round(x, 4) for x in cos_sim_scores]
            print('scores length: ', len(cos_sim_scores))
            print('first 10 items of scores: ', cos_sim_scores[:10])
            scores['Cos_sim'] = cos_sim_scores

        results = pd.DataFrame(columns=['Metric', 'Scores'])
        for key, value in scores.items():
            results = pd.concat([results, pd.DataFrame([[key, value]], columns=['Metric', 'Scores'])], ignore_index=True)

        print(f'\nWriting sentence_scores for {percentage}% of dataset to file..')
        filename = f'per_sentence_similarity_scores_{percentage}.csv'
        save_result_file('similarity', filename, results)


#Not used atm
def jaccard_similarity(doc1, doc2):
    """Calculate the jaccard similarity between two documents
    If the two documents are identical, the similarity will be 1.0
    If the two documents are completely different, the similarity will be 0.0
    
    Args:
        doc1 (list): A list of tokens
        doc2 (list): A list of tokens
            
    Returns:
        float: The jaccard similarity between the two documents
    """

    #Get the union of the two documents
    union = set(doc1).union(set(doc2))

    #Get the intersection of the two documents
    intersection = set(doc1).intersection(set(doc2))

    #Calculate the jaccard similarity
    similarity = len(intersection) / len(union)

    return similarity