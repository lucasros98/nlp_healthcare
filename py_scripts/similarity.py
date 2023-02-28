import math 
import numpy as np
import torch
from collections import Counter
from data import get_training_data


def sentence_transformer_score(sentence1, sentence2):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
    #Compute embedding for both lists
    
    a = " ".join(sentence1)
    b = " ".join(sentence2)

    embedding_1= model.encode(a, convert_to_tensor=True)
    embedding_2 = model.encode(b, convert_to_tensor=True)
    score_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)
    return score_tensor.item()


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

#Calculate the BLEU score between two documents using nltk
def bleu_score_nltk(doc1, doc2):
    """Calculate the BLEU score between two documents using nltk
    If the two documents are identical, the BLEU score will be 1.0
    If the two documents are completely different, the BLEU score will be 0.0
    
    Args:
        doc1 (list): A list of tokens
        doc2 (list): A list of tokens
            
    Returns:
        float: The BLEU score between the two documents
    """

    #Import the nltk library
    import nltk

    #Split sentences into list of tokens
    doc1 = doc1.split()
    doc2 = doc2.split()

    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    #Calculate the BLEU score
    score = nltk.translate.bleu_score.sentence_bleu([doc1], doc2, weights=(1, 0, 0, 0), smoothing_function=chencherry.method0)

    print("BLEU Score:", score)


def similarity_loop():
    sentence1 = 'planeringsansvarig ssk vid inskrivning ssk Roland'
    sentence2 = 'journalförare Per Ekström m48.5 kotkompression som ej klassificeras annorstädes'
    cosine_sim_test(sentence1, sentence2)
    sentence_transformer_score(sentence1, sentence2)
    bleu_score_nltk(sentence1, sentence2)

    sentence1 = 'välbefinnande säger att det är okej att ligga här'
    sentence2 = 'välbefinnande säger att det är okej att ligga här'
    cosine_sim_test(sentence1, sentence2)
    sentence_transformer_score(sentence1, sentence2)
    bleu_score_nltk(sentence1, sentence2)

    sentence1 = 'välbefinnande säger att det är okej att ligga här'
    sentence2 = 'välbefinnande säger att det inte är okej att ligga här'
    cosine_sim_test(sentence1, sentence2)
    sentence_transformer_score(sentence1, sentence2)
    bleu_score_nltk(sentence1, sentence2)

    sentence1 = 'jag gillar bilar'
    sentence2 = 'jag gillar inte bilar'
    cosine_sim_test(sentence1, sentence2)
    sentence_transformer_score(sentence1, sentence2)
    bleu_score_nltk(sentence1, sentence2)


# Calculate the average cosine similarity between two lists of documents
def compute_cosine_similarity(docs1, docs2):
    total_scores = []
    for doc1 in docs1:
        partial_scores = []
        # compute high, low and mean 
        for doc2 in docs2:
            partial_scores.append(sentence_transformer_score(doc1, doc2))
        total_scores.append(sum(partial_scores))
    return(sum(total_scores) / len(docs1))


def print_dataset_similarity_scores(metric='cosine'):
    print("STARTING")
    percentages = [25, 50, 75, 100]
    for percentage in percentages:
        print(f'Percentage: {percentage}%')
        X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=percentage)
        score = compute_cosine_similarity(X_test, X_train)
        print(f'Score: {score}')

print_dataset_similarity_scores()

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