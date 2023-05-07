import torch

class NERParameters():
    def __init__(self):
        # Random seed, for reproducibility.
        self.random_seed = 1

        
        # cuda or cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   
        
        #Tagging scheme used in data: IO or BIO (Inside-Outside-Beginning)
        self.tagging_scheme="BIO"
        
        #Training parameters
        self.n_epochs = 25
        self.batch_size = 32

        self.learning_rate = 4e-5
        self.epsilon = 1e-08
        self.weight_decay = 0.0

        #Early stopping parameters
        self.early_stopping = True
        self.patience = 5

        #For linear learning rate decay
        self.lr_decay = True
        self.warmup_steps = 0.1 # Warmup steps (In precentage)

        #Logging parameters (For wandb)
        self.logging = True

        #Augmented

        # Word dropout rate.
        self.word_dropout_prob = 0

        self.bert_max_len = 512