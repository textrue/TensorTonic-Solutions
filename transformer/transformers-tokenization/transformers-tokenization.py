import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        current_id = 4
        # Since this won't be large
        vocab = sorted(" ".join(texts).split())
        for text in vocab:
            if text not in self.word_to_id:
                self.word_to_id[text] = current_id
                self.id_to_word[current_id] = text
                current_id += 1

        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        res = []
        for word in text.split():
            word = word.lower()
            if word in self.word_to_id:
                res.append(self.word_to_id[word])
            else:
                res.append(self.word_to_id[self.unk_token])
        return res
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        res = []
        for id in ids:
            if id in self.id_to_word:
                res.append(self.id_to_word[id])
            else:
                res.append(self.id_to_word[1])
        return " ".join(res)
