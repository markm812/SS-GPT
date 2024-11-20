import torch
import tokenizer


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.text = None
        self.unique_chars = []
        self.vocab_size = 0
        self.load_data()
        
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as file:
            self.text = file.read()
        self.unique_chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.unique_chars)

    def get_unique_chars(self):
        return self.unique_chars

    def get_vocab_size(self):
        return self.vocab_size

    def get_data(self):
        return self.text

    def get_data_size(self):
        return len(self.text)
