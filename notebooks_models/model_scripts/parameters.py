import torch

class NERParameters():

    # Random seed, for reproducibility.
    random_seed = 0
    
    # cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"   
    
    #Tagging scheme used in data: IO or BIO (Inside-Outside-Beginning)
    tagging_scheme="BIO"
    
    #Training parameters
    n_epochs = 3
    batch_size = 32 
    learning_rate = 0.0001
    weight_decay = 0

    # Word dropout rate.
    word_dropout_prob = 0.0

    bert_max_len = 512