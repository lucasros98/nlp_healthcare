import random
from collections import defaultdict, Counter    
import sys
import pandas as pd
import time
import copy
import os
import evaluate as ev
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import torch

seqeval = ev.load('seqeval')

#Base import on the path when importing vocab.py
#The path will need to be nlp_healthcare/py_scripts/ner_util
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts/ner_util')
from vocab import Vocabulary
from logger import Logger

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.counter = 0

    def load_checkpoint(self):
        return self.best_model
    

class SequenceDataset(Dataset):
    """A Dataset that stores a list of sentences (X) and their corresponding labels (Y)"""
    def __init__(self, X, Y, word_dropout_prob=None, word_dropout_id=None):
        self.X = X
        self.Y = Y
        self.word_dropout_prob = word_dropout_prob
        self.word_dropout_id = word_dropout_id
    
    # return a single item in the dataset given its index.
    def __getitem__(self, idx):
        if self.word_dropout_prob:
            words = [ w if random.random() > self.word_dropout_prob else self.word_dropout_id for w in self.X[idx] ]
        else:
            words = self.X[idx]
        
        return words, self.Y[idx]

    # returns the number of samples in our dataset.
    def __len__(self):
        return len(self.Y)

class SequenceBatcher:
    """A collator that builds a batch from a number of sentence--labeling pairs."""
    
    def __init__(self, device):
        self.device = device
    
    def __call__(self, XY):
        """Build a batch from a number of sentences. Returns two tensors X and Y, where
        X is the sentence tensor, of shape [n_sentences, max_sen_length]
        and 
        
        Y is the label tensor, of the same shape as X.
        """
        
        # Assume that the padding id is 0.
        pad_id = 0
                
        # How long is the longest document in this batch?
        max_sen_len = max(len(row[0]) for row in XY)

        # Build the document tensor. We pad the shorter documents so that all documents
        # have the same length.
        Xpadded = torch.as_tensor([row[0] + [pad_id]*(max_sen_len-len(row[0])) for row in XY], device=self.device)
        
        # Build the label tensor.
        Ypadded = torch.as_tensor([row[1] + [pad_id]*(max_sen_len-len(row[1])) for row in XY], device=self.device)

        return Xpadded, Ypadded
       


# Aligns encoded output BIO labels with the word pieces created by the tokenizer.
def remap_entity_indices_iob(word_tokenized, label_encoded, voc):
    out = []
    O = voc.stoi['O']
    for i, y in enumerate(label_encoded):
        tokens_sen = word_tokenized[i]
        new_len = len(tokens_sen.ids)
        spans = to_spans_iob(y, voc)
        y_new = [y[0]]
        prev = 1
        for start, (lbl, end) in sorted(spans.items()):
            start_remapped = tokens_sen.word_to_tokens(start-1)#[0]
            if start_remapped is None:
                continue
            y_new.extend([O]*(start_remapped[0]-prev))
            y_new.append(voc.stoi['B-' + lbl])
                        
            end_remapped = tokens_sen.word_to_tokens(end-1)
            if end_remapped is None:
                end_remapped = (new_len-1, )
            
            y_new.extend([voc.stoi['I-' + lbl]]*(end_remapped[0]-start_remapped[0]-1))
            prev = end_remapped[0]
        y_new.extend([O]*(new_len-prev-1))
        y_new.append(y[-1])
        out.append(y_new)
    return out


# Aligns encoded output labels (just a lebel or outside) with the word pieces created by the tokenizer.
def remap_entity_indices(word_tokenized, label_encoded, voc):
    out = []
    O = voc.stoi['O']
    for i, y in enumerate(label_encoded):
        tokens_sen = word_tokenized[i]
        new_len = len(tokens_sen.ids)
        spans = to_spans(y, voc)
        y_new = [y[0]]
        prev = 1
        for start, (lbl, end) in sorted(spans.items()):
            start_remapped = tokens_sen.word_to_tokens(start-1)#[0]
            if start_remapped is None:
                continue
            y_new.extend([O]*(start_remapped[0]-prev))
            y_new.append(voc.stoi[lbl])
                        
            end_remapped = tokens_sen.word_to_tokens(end-1)
            if end_remapped is None:
                end_remapped = (new_len-1, )
            
            y_new.extend([voc.stoi[lbl]]*(end_remapped[0]-start_remapped[0]-1))
            prev = end_remapped[0]
        y_new.extend([O]*(new_len-prev-1))
        y_new.append(y[-1])
        out.append(y_new)
    return out

# Convert a list of BIO labels, coded as integers, into spans identified by a beginning, an end, and a label.
# To allow easy comparison later, we store them in a dictionary indexed by the start position.
def to_spans_iob(l_ids, vocab):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = vocab.itos[l_id]

        if l[0] == 'B': 
            # Beginning of a named entity: B-something.
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            # Continuation of an entity: I-something.
            if current_lbl:
                # If we have an open entity, but its label does not
                # correspond to the predicted I-tag, then we close
                # the open entity and create a new one.
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                # If we don't have an open entity but predict an I tag,
                # we create a new entity starting here even though we're
                # not following the format strictly.
                current_lbl = l[2:]
                current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans


# Convert a list of labels, coded as integers, into spans.
# To allow easy comparison later, we store them in a dictionary indexed by the start position.
def to_spans(l_ids, vocab):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = vocab.itos[l_id]
        #Check if it is is a label

        if l in ['First_Name', 'Last_Name', 'Phone_Number', 'Age', 'Full_Date', 'Date_Part', 'Health_Care_Unit', 'Location']: 
            # A named entity
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l
            current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans


# Compares two sets of spans and records the results for future aggregation.
def compare(gold, pred, stats):
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    for start, (glbl, gend) in gold.items():
        if start in pred:
            plbl, pend = pred[start]
            if glbl == plbl and gend == pend:
                stats['total']['corr'] += 1
                stats[glbl]['corr'] += 1

# Decodes a list of encoded labels
def decode_labels(l_ids, vocab, max_len):
    decoded_ids = []
    for l_id in l_ids:
        decoded_ids.append(vocab.itos[l_id])
    return split_list(decoded_ids, max_len, vocab)

# Splits the long list of all labels into lists separated for each sentence
def split_list(lst, max_len, vocab):
    result = []

    for i in range(0, len(lst), max_len):
        sublist = lst[i:i+max_len]

        #Replace dummies with '0'
        sublist = filter_list(sublist, vocab.dummies)

        result.append(sublist)

    return result

# Replace dummy labels from list
def filter_list(lst,dummies):
    return list(map(lambda x: 'O' if x in dummies else x, lst))
 

# This function combines the auxiliary functions we defined above.
def evaluate(words, predicted, gold, vocab, stats, predictions, references, params=None):
    
    pad_id = vocab.get_pad_idx()
    padding = list((words == pad_id).reshape(-1).cpu().numpy())
                
    # The gold-standard labels are assumed to be an integer tensor of shape
    # (n_sentences, max_len).
    gold_cpu = gold.cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))

    if not isinstance(predicted, list):        
        pred_flat = predicted.reshape(-1).cpu().numpy()
    else:
        pred_flat = [l for sen in predicted for l in sen]
    pred_flat = [pad_id if is_pad else l for l, is_pad in zip(pred_flat, padding)]

    # Decode encoded list of labels to the respective BIO-label.
    gold_decoded = decode_labels(gold_cpu, vocab, params.bert_max_len)
    pred_decoded = decode_labels(pred_flat, vocab, params.bert_max_len)

    # Concat the labels of this batch for gold and pred
    predictions.extend(pred_decoded)
    references.extend(gold_decoded)
    
    
    # Compute spans for the gold standard and prediction.
    if params.tagging_scheme == 'BIO':
        gold_spans = to_spans_iob(gold_cpu, vocab)
        pred_spans = to_spans_iob(pred_flat, vocab)
    else:
        gold_spans = to_spans(gold_cpu, vocab)
        pred_spans = to_spans(pred_flat, vocab)


    # Finally, update the counts for correct, predicted and gold-standard spans.
    compare(gold_spans, pred_spans, stats)    


#Print recision, recall and F1 score for entities
def print_report(results):

    #Extract precision, recall and F1 score from results
    p = results["overall_precision"]
    r = results["overall_recall"]
    f1 = results["overall_f1"]

    print()
    print('Evaluation report: \n')
    print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f} \n')

    report = pd.DataFrame(columns=["entity", "precision", "recall", "f1", "number"])

    for key, value in results.items():
        if isinstance(value, dict):
            report = pd.concat([report, pd.DataFrame([[key, round(value["precision"], 4), round(value["recall"], 4), round(value["f1"], 4), value["number"]]], columns=["entity", "precision", "recall", "f1", "number"])], ignore_index=True)
    report = report.sort_values(by="f1", ascending=False)

    number_all = report["number"].sum()

    #Add the overall row
    report = pd.concat([report, pd.DataFrame([["overall", round(p, 4), round(r, 4), round(f1, 4), number_all]], columns=["entity", "precision", "recall", "f1", "number"])], ignore_index=True)

    #remove the index column
    report = report.reset_index(drop=True)
    
    #print the report
    print(report)

    return report

class SequenceLabeler:
    def __init__(self, params, model_factory, bert_tokenizer, verbose=True):
        self.params = params        
        self.model_factory = model_factory
        self.bert_tokenizer = bert_tokenizer
        self.verbose = verbose

        # Initialize the logger.
        self.logger = Logger(project=params.model_name, config=vars(params))

    # Preprocess the data, build vocabularies and data loaders.
    def preprocess(self, Xtrain, Ytrain, Xval, Yval,tagging_scheme=None):
        # Build vocabularies
        p = self.params
        
        self.label_voc = Vocabulary(include_unknown=False)
        self.label_voc.build(Ytrain)

        self.n_labels = len(self.label_voc)
        train_lbl_encoded = self.label_voc.encode(Ytrain)
        val_lbl_encoded = self.label_voc.encode(Yval)
        
        #Tokenize
        train_tokenized = self.bert_tokenizer(Xtrain, is_split_into_words=True, truncation=True, max_length=p.bert_max_len)
        val_tokenized = self.bert_tokenizer(Xval, is_split_into_words=True, truncation=True, max_length=p.bert_max_len)
        
        #Encode labels
        if(tagging_scheme=='BIO'):
            train_lbl_encoded = remap_entity_indices_iob(train_tokenized, train_lbl_encoded, self.label_voc)
            val_lbl_encoded = remap_entity_indices_iob(val_tokenized, val_lbl_encoded, self.label_voc)
        else: 
            train_lbl_encoded = remap_entity_indices(train_tokenized, train_lbl_encoded, self.label_voc)
            val_lbl_encoded = remap_entity_indices(val_tokenized, val_lbl_encoded, self.label_voc)
        
        train_word_encoded = train_tokenized.input_ids
        val_word_encoded = val_tokenized.input_ids
       
        dropout_id = self.bert_tokenizer.unk_token_id

        # Put the training and validation data into Datasets and DataLoaders for managing minibatches.
        self.batcher = SequenceBatcher(p.device)
        
        train_dataset = SequenceDataset(train_word_encoded, train_lbl_encoded,
                                        word_dropout_prob=p.word_dropout_prob, word_dropout_id=dropout_id)
        self.train_loader = DataLoader(train_dataset, p.batch_size, shuffle=True, collate_fn=self.batcher)        
        
        val_dataset = SequenceDataset(val_word_encoded, val_lbl_encoded)
        self.val_loader = DataLoader(val_dataset, p.batch_size, shuffle=False, collate_fn=self.batcher)

    def fit(self, Xtrain, Ytrain, Xval, Yval):
        
        p = self.params
        
        # Setting a fixed seed for reproducibility.
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)
                           
        # Preprocess the data, build vocabularies and data loaders.
        self.preprocess(Xtrain, Ytrain, Xval, Yval, tagging_scheme=p.tagging_scheme)
            
        # Now, let's build the model!
        self.model = self.model_factory(self)

        # Create the sequence labeling neural network defined above.
        self.model.to(p.device)

        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=p.learning_rate, weight_decay=p.weight_decay)

        #Get the total number of steps
        total_steps = len(self.train_loader) * p.n_epochs

        # Create the learning rate scheduler.
        if p.lr_decay:
            warmup_steps = int(p.warmup_steps * total_steps)
            
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps,
                                                num_training_steps = total_steps)


        # Cross-entropy loss function that we will use to optimize the model.
        # In particular, note that by using ignore_index, we will not compute the loss 
        # for the positions where we have a padding token.
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.label_voc.get_pad_idx())
        
        self.history = defaultdict(list)
        best_f1 = -1 

        #Early stopping
        if p.early_stopping:
            early_stopping = EarlyStopping(patience=p.patience)
     
            
        for i in range(p.n_epochs):

            t0 = time.time()

            loss_sum = 0
            self.model.train()

            for j, batch in enumerate(self.train_loader, 1):

                # Compute the output scores.                
                scores = self.model(batch[0])                
                
                # The scores tensor has the shape (n_sentences, n_words, n_labels).
                # We reshape this to (n_sentences*n_words, n_labels) because the loss requires
                # a 2-dimensional tensor. Similarly for the gold standard label tensor.                
                loss = loss_func(scores.view(-1, len(self.label_voc)), batch[1].view(-1))
                    
                #Back propagation
                optimizer.zero_grad()            
                loss.backward()

                optimizer.step()
                if p.lr_decay:
                    scheduler.step()  # decay LR


                loss_sum += loss.item()
                
                if self.verbose:
                    print('.', end='')
                    sys.stdout.flush()
                    if j % 50 == 0:
                        print(f' ({j})')               
            if self.verbose:
                print()

            # Compute the training loss. 
            train_loss = loss_sum / len(self.train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate on the validation set.
            stats = defaultdict(Counter)

            predictions = []
            references = []

            self.model.eval()

            val_loss_sum = 0

            with torch.no_grad():
                for j, batch in enumerate(self.val_loader, 1):
                    
                    # Compute the output scores.
                    scores = self.model(batch[0])   
                    
                    # Compute the highest-scoring labels at each word position.
                    predicted = scores.argmax(dim=2)

                    # Compute the validation loss.
                    loss = loss_func(scores.view(-1, len(self.label_voc)), batch[1].view(-1))
                    val_loss_sum += loss.item()
                    
                    # Update the evaluation statistics for this batch.
                    evaluate(batch[0], predicted, batch[1], self.label_voc, stats, predictions, references, params=self.params)

                    if self.verbose:
                        print('.', end='')
                        sys.stdout.flush()
                        if j % 50 == 0:
                            print(f' ({j})')
            if self.verbose:
                print()

            # Compute the overall loss for the validation set.
            val_loss = val_loss_sum / len(self.val_loader)
            self.history['val_loss'].append(val_loss)
   
            # Compute the overall F-score for the validation set.            
            results = seqeval.compute(predictions=predictions, references=references, mode='strict', scheme='IOB2',zero_division=1)
            val_f1 = results["overall_f1"]

            self.history['val_f1'].append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1

            t1 = time.time()
            print(f'Epoch {i+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val f1: {val_f1:.4f}, time = {t1-t0:.4f}')

            # Log the metrics for this epoch.
            values = {"Epoch": i+1, "train loss": train_loss, "val loss": val_loss, "val f1": val_f1, "time": t1-t0, "learning_rate": scheduler.get_last_lr()[0] if p.lr_decay else "None"}
            if self.logger:
                self.logger(values=values)

            # If we have not improve the validation loss over 2 epochs, we stop the training.
            # We also restore the previously best model state
            if p.early_stopping:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    self.model = early_stopping.load_checkpoint()
                    break

        #Calculate the number of training steps
        total_steps = len(self.train_loader) * (i + 1)
        print("Total training steps: {}".format(total_steps))

        # Log the total number of training steps
        self.logger(values={"Total training steps": total_steps})

        #Load the best model
        if p.early_stopping:
            self.model = early_stopping.load_checkpoint()

        # After the final evaluation, we print more detailed evaluation statistics, including
        # precision, recall, and F-scores for the different types of named entities.
        results = seqeval.compute(predictions=predictions, references=references, mode='strict', scheme='IOB2',zero_division=1)

        return self.history
        
    def predict(self, sentences):
        # This method applies the trained model to a list of sentences.
        
        word_encoded = self.bert_tokenizer(sentences, is_split_into_words=True, truncation=True, 
                                               max_length=self.params.bert_max_len).input_ids
            
        Ydummy = [[0]*len(x) for x in word_encoded]
                
        dataset = SequenceDataset(word_encoded,
                                  Ydummy)
        loader = DataLoader(dataset, self.params.batch_size, shuffle=False, collate_fn=self.batcher)
                
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:

                scores = self.model(batch[0])                
                predicted = scores.argmax(dim=2) 
                
                # Convert the integer-encoded tags to tag strings.
                for pred_sen in predicted.cpu().numpy():
                    tokens = word_encoded[len(out)]
                    out.append([self.label_voc.itos[pred_id] for _, pred_id in zip(tokens, pred_sen[1:-1])])
        return out



    def evaluate_model(self, X, Y):
        # This method evaluates the model on test data.
        
        word_tokenized = self.bert_tokenizer(X, is_split_into_words=True, truncation=True, 
                                               max_length=self.params.bert_max_len)
        
        label_encoded = self.label_voc.encode(Y)

        #Encode labels
        if(self.params.tagging_scheme=='BIO'):
            label_encoded = remap_entity_indices_iob(word_tokenized, label_encoded, self.label_voc)
        else: 
            label_encoded = remap_entity_indices(word_tokenized, label_encoded, self.label_voc)
        
        word_encoded = word_tokenized.input_ids
        
        dataset = SequenceDataset(word_encoded,
                                  label_encoded)

        loader = DataLoader(dataset, self.params.batch_size, shuffle=False, collate_fn=self.batcher)
                
        stats = defaultdict(Counter)

        predictions = []
        references = []

        self.model.eval()
        with torch.no_grad():
            for j, batch in enumerate(loader, 1):
                
                scores = self.model(batch[0])                
                predicted = scores.argmax(dim=2) 
                
                evaluate(batch[0], predicted, batch[1], self.label_voc, stats, predictions, references, params=self.params)

        #Print evalutation statistics
        results = seqeval.compute(predictions=predictions, references=references, mode='strict', scheme='IOB2',zero_division=1)
        report = print_report(results)

        # Log the results
        if self.logger:
            self.logger(values={"overall_precision": results["overall_precision"],"overall_recall": results["overall_recall"], "overall_f1": results["overall_f1"]})

        return report