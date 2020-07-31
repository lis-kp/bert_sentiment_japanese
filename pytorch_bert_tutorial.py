#!/usr/bin/env python
# coding: utf-8

# In this tutorial, we are going to fine-tune a pre-trained BERT model for a sentiment classification test. For fine-tuning, we use the KNBC corpus.
#
import pandas as pd
import os
import glob
import transformers
import torch
import random
import numpy as np

# Set the seed value all over the place to make this reproducible.
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 
#  ### Load dataset
#
import pandas as pd
# Load the dataset into a pandas dataframe.
df = pd.read_csv("sentiment_bert/sentiment_data/all.tsv", delimiter='\t', header=None, names=['domain', 'sentence', 'label'])

# Report the number of sentences.
print('Total number of sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(10).style.hide_index()

# ### Split dataset into train, val and test parts

from sklearn.model_selection import train_test_split

#get the sentences and their labels only
sentences = df.sentence.values
labels = df.label.values

# Use 70% for training, 15% for validation and 15% for test.
train_sents, validation_sents, train_labels, validation_labels = train_test_split(sentences, labels, 
                                                            random_state=2018, test_size=0.3)

test_sents, validation_sents, test_labels, validation_labels = train_test_split(validation_sents, validation_labels, 
                                                            random_state=2018, test_size=0.5)

print("Number of train sentences: ", len(train_sents))
print("Number of validation sentences: ", len(validation_sents))
print("Number of test sentences: ", len(test_sents))


# ### Convert dataset into BERT input format

from sentiment_bert.bert_data_processor_ja import BERTInputConverter

train_set = BERTInputConverter(train_sents, train_labels)
validation_set = BERTInputConverter(validation_sents, validation_labels)
test_set = BERTInputConverter(test_sents, test_labels)

print("Dataset converted to BERT input format!!!")

# ### Convert dataset into pytorch format

from torch.utils.data import DataLoader
batch_size = 16

train_dataloader = DataLoader(train_set, batch_size=batch_size)
validation_dataloader = DataLoader(validation_set, batch_size=batch_size)
test_dataloader = DataLoader(test_set, batch_size=batch_size)

print("Dataset converted to pytorch format!")

# ### Build BERT classifer

from transformers import BertForSequenceClassification, AdamW, BertConfig
#check if gpu is available
if torch.cuda.is_available():
  device = torch.device("cuda")

  #print number and type of gpu available
  print("Number of GPUs available: %d" % torch.cuda.device_count())
  #print("GPU type:", torch.cuda.get_device_name(0))
  print("")

else:
  device = torch.device("cpu")

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking", #we use Japanese BERT model
    num_labels = 2, #number of labels
    output_attentions = False, # Whether the model returns attentions weights
    output_hidden_states = False, # Whether the model returns all hidden-states
)

# run the model on the GPU, if available, or CPU, if not.
model.to(device)

# #### Set up optimizer and learning rate scheduler

# set up optimizer
learning_rate = 2e-5
adam_eps = 1e-8
optimizer = AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = adam_eps 
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# compute warmup step
warmup_steps = 0

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

# ## Fine-Tuning BERT
#

import sentiment_bert.eval_utils

best_acc = 0
best_model = None

# For each epoch...
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss
    total_loss = 0

    # Put the model into training mode
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Unpack training batch    
        b_input_ids = batch[0].to(device)  #[0]: input ids 
        b_input_mask = batch[1].to(device) #[1]: attention masks
        b_labels = batch[2].to(device)     #[2]: labels 

        # clear gradients
        model.zero_grad()        

        # evaluate the model on this training batch
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # loss value
        loss = outputs[0]

        # Accumulate the training loss over all of the batches
        total_loss += loss.item()

        # backward pass 
        loss.backward()

        # Clip the norm  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters 
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data
    avg_train_loss = total_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    
    #computes model accuracy on the validation set (for current epoch)
    print("")
    print("Running Validation...")
    val_acc = sentiment_bert.eval_utils.evaluate(model, validation_dataloader)
    
    #report accuracy
    print("  Accuracy: {0:.2f}".format(val_acc))

    #saves best model so far
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model

print("")
print("Training complete!")
print("Best accuracy on validation set: {0:.2f}".format(best_acc))

# ## Evaluation on test set

print("")
print("Running evaluation on test set...")
test_acc = sentiment_bert.eval_utils.evaluate(best_model, test_dataloader)
#report accuracy
print("  Accuracy: {0:.2f}".format(test_acc))

# ####Test on a single sentence

sentence = "中華そばより高いが、アサリが入ったスープはボンゴレ風で、これが今までにない新しい感覚でうまい。"
predicted_label, probability = sentiment_bert.eval_utils.evaluate_single_sentence(best_model, sentence)

print("Predicted label:", predicted_label)
print("Probability: {0:.4f}".format(probability))
