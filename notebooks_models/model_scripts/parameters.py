import torch

class NERParameters():

    # Random seed, for reproducibility.
    random_seed = 1
    
    # cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"   
    
    #Tagging scheme used in data: IO or BIO (Inside-Outside-Beginning)
    tagging_scheme="BIO"
    
    #Training parameters
    n_epochs = 25
    batch_size = 32

    learning_rate = 2e-5
    epsilon = 1e-08
    weight_decay = 0.0

    #Early stopping parameters
    early_stopping = True
    patience = 5

    #For linear learning rate decay
    lr_decay = True
    warmup_steps = 0.1 # Warmup steps (In precentage)

    # Word dropout rate.
    word_dropout_prob = 0

    bert_max_len = 512