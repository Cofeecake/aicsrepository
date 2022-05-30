"""
    Description: construct the NLP Transformer model here
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from text_preprocessing import clean_texts
# import config
import torch
import os
import numpy as np
# disable HuggingFace Warning
import logging
logging.disable(logging.WARNING)
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')



class BertUncased():
    """
    A Class for A.I model, optimizer, and scheduler storage
    """
    def __init__(self):
        self.optimizer = None
        self.lrScheduler = None
        self.model = None

    def build_model(self, num_labels):
        # Class method to initialize the A.I. model -> self.model
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",          # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels,      # The number of output labels
            output_attentions = False,    # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    
    def load_weights(self, filepath, device, predict_mode=True):
        # Class method to load pretrained model weights into the A.I. model -> self.model
        # NOTE: this loading method is for making predictions 
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if predict_mode:
            self.model.eval()
    
    def load_weights_torchhub(self, url, progress=False, predict_mode=True, save_path=None):
        # Class method to load pretrained model weights by utilizing torch hub into the A.I. model -> self.model
        # NOTE: this loading method is for making predictions
        if not save_path:
            loaded_weights = torch.hub.load_state_dict_from_url(url, progress=progress, map_location="cpu")
            self.model.load_state_dict(
                loaded_weights['model_state_dict']
            )
        else:
            if not os.path.isfile(save_path):
                loaded_weights = torch.hub.load_state_dict_from_url(url, progress=progress, map_location="cpu")
                torch.save(loaded_weights['model_state_dict'], save_path)
                self.model.load_state_dict(
                    loaded_weights['model_state_dict']
                )
            else:
                self.model.load_state_dict(
                    torch.load(save_path)
                )
        
        if predict_mode:
            self.model.eval()


def predict(input, model, input_tknz, label_tknz, device=None):
    """
    Purpose: Utilize the A.I. model to make predictions
    Params:  1. input (1d list of string sentences):
                - The preprocessed string sentence the model is trying to make prediction of.
             2. model (BertForSequenceClassification):
                - The model to make predictions with.
             3. input_tknz (transformers.models.bert.tokenization_bert.BertTokenizer):
                - The input tokenizer that transform string inputs into a list of pytorch float
             4. label_tknz (Label2Id object):
                - Label tokenizer that converts string labels into integers.
             5. device (torch.device):
                - What device to use for A.I. training, generally 'cpu' or 'cuda'
    Returns: A list of predictions containing string labels
    """
    
    # preprocess the input with function clean_texts from file text_preprocessing.py
    preprocessed_input = clean_texts(input)

    # set model to evaluation mode
    model.eval()

    # set model to utilize device
    model.to(device if device != None else 'cpu')
    predictions = []
    for sent in preprocessed_input:

        # Detach any tensors with gradients that are currently attached to the computational graph.
        with torch.no_grad():
            input_seqs = []
            attention_masks = []
            """
            "encode_plus" will:
            (1) Tokenize the sentence.
            (2) Prepend the `[CLS]` token to the start.
            (3) Append the `[SEP]` token to the end.
            (4) Map tokens to their IDs.
            (5) Pad or truncate the sentence to `max_length`
            (6) Create attention masks for [PAD] tokens
            """
            encoded_dict = input_tknz.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 200,           # Pad & truncate all sentences., config.max_len
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
            
            # append outputs from input tokenizer
            input_seqs.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            input_seqs = torch.cat(input_seqs, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            # cast inputs to utilize device
            input_seqs = input_seqs.to(device if device != None else 'cpu')
            attention_masks = attention_masks.to(device if device != None else 'cpu')
            output = model(input_seqs, 
                     token_type_ids=None, 
                     attention_mask=attention_masks,
                     return_dict=True)
            
            # cast outputs to numpy
            output_np = output[0].to('cpu').numpy()
            str_output = label_tknz.decoder(np.argmax(output_np[0]))
            predictions.append(str_output)
    
    # get list of predictions
    return predictions