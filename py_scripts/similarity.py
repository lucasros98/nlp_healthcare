import math 
import numpy as np
import torch
from collections import Counter

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