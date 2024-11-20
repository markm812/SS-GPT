#  Abstract class for tokenizers
class Tokenizer:
    def encode(self, s):
        raise NotImplementedError

    def decode(self, l):
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.char_to_idx[char] for char in s]

    def decode(self, l):
        return "".join([self.idx_to_char[idx] for idx in l])
