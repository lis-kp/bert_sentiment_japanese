import numpy as np
import torch
from sentiment_bert.bert_data_processor_ja import BERTInputConverter

# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def evaluate(model, dataloader):
    # check if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    acc = eval_accuracy / nb_eval_steps
    return acc

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def evaluate_single_sentence(model, sentence):
    # check if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    bert = BERTInputConverter()
    input_ids_tensor, input_mask_tensor = bert.get_bert_features(sentence)

    # Put the model in evaluation mode
    model.eval()

    input_ids_tensor = input_ids_tensor.to(device)
    input_mask_tensor = input_mask_tensor.to(device)

    # Telling the model not to compute or store gradients
    with torch.no_grad():
        output = model(input_ids_tensor.unsqueeze(0),
                                token_type_ids=None,
                                attention_mask=input_mask_tensor.unsqueeze(0))

    logits = output[0].detach().cpu().numpy()

    predicted_label = np.argmax(logits, axis=1).flatten()

    if predicted_label[0] == 0:
        predicted_label = "Negative"
    else:
        predicted_label = "Positive"


    probability = np.max(softmax(logits)[0])

    return predicted_label, probability

