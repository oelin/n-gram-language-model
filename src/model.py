from typing import Any, Callable, Sequence
import math

from collections import Counter
from nltk.util import ngrams


String = Sequence[Any]
ProbabilityDistribution = Callable[[Any], float]
LanguageModel = Callable[[String], ProbabilityDistribution]


class NGramLanguageModel:
    """N-gram language model with Laplace smoothing."""

    def __init__(self, vocabulary_size: int, context_size: int = 2) -> None:
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
    
    def fit(self, string: String) -> None:
        """Fits the model."""
        
        observed_ngrams = []

        for i in range(1, self.context_size + 2):
            observed_ngrams += ngrams(string, i)
        
        self.counter = Counter(observed_ngrams)
        self.counter[('',)] = len(string)  # A "zero-gram" occurs before every token.
    
    def __call__(self, string: String) -> ProbabilityDistribution:
        """Computes the conditoinal probability distribution p(token|string)."""
      
        string = (*string[-self.context_size :])
        denominator = (self.counter.get(string) or 0) + self.vocabulary_size

        return lambda token: (self.counter.get((*prefix, token)) or 0) + 1) / denominator
