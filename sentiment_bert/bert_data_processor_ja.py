import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

# customized class for preparing datasets
class BERTInputConverter(Dataset):
    def __init__(self, sentences=None, labels=None, maxlen=128):

        self.sentences = sentences
        self.labels = labels

        # Initialize the BERT tokenizer
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

        self.maxlen = maxlen

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index
        sentence = self.sentences[index]
        label = self.labels[index]

        #get bert features
        input_ids_tensor, input_mask_tensor, label_tensor = self.get_bert_features(sentence, label)

        return input_ids_tensor, input_mask_tensor, label_tensor

    def get_bert_features(self, sentence, label=None):
        # Tokenize the sentence
        tokenized_sentence = self.tokenizer.tokenize(sentence)

        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokenized_sentence) > self.maxlen - 2:
            tokenized_sentence = tokenized_sentence[:self.maxlen - 2]

        tokens = ['[CLS]'] + tokenized_sentence + [
            '[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence

        input_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.maxlen:
            input_ids.append(0)
            input_mask.append(0)

        # Converting to pytorch tensors
        input_ids_tensor = torch.tensor(input_ids)

        input_mask_tensor = torch.tensor(input_mask)

        if label is not None:
            label_tensor = torch.tensor(label)
            return input_ids_tensor, input_mask_tensor, label_tensor

        return input_ids_tensor, input_mask_tensor

