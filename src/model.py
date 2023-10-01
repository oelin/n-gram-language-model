from typing import Any, Callable, Sequence
import math
from collections import Counter
from nltk.util import ngrams

String = Sequence[Any]
ProbabilityDistribution = Callable[[Any], float]
LanguageModel = Callable[[String], ProbabilityDistribution]


class NGramLanguageModel:
    """N-gram language model with Laplace smoothing.

    Args:
        vocabulary_size (int): The size of the vocabulary.
        context_size (int, optional): The size of the n-gram context (default is 2).

    Attributes:
        vocabulary_size (int): The size of the vocabulary.
        context_size (int): The size of the n-gram context.
        counter (Counter): Counter object to store n-gram counts.

    Methods:
        fit(string: String) -> None:
            Fits the model to the input string by counting n-grams.

        __call__(string: String) -> ProbabilityDistribution:
            Computes the conditional probability distribution p(token|string).
    """

    def __init__(self, vocabulary_size: int, context_size: int = 2) -> None:
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.counter = Counter()

    def fit(self, string: String) -> None:
        """
        Fit the model to the input string by counting n-grams.

        Args:
            string (String): The input sequence.

        Returns:
            None
        """
        
        for i in range(1, self.context_size + 2):
            observed_ngrams = ngrams(string, i)
            self.counter.update(observed_ngrams)

        # Add a "zero-gram" for the entire string.
        
        self.counter[('',)] = len(string)

    def __call__(self, string: String) -> ProbabilityDistribution:
        """
        Compute the conditional probability distribution p(token|string).

        Args:
            string (String): The input sequence.

        Returns:
            ProbabilityDistribution: A function that calculates the negative log probability
            of a given token based on the model.
        """
        
        context = (*string[-self.context_size:],)
        denominator = self.counter.get(context, 0) + self.vocabulary_size
        
        return lambda token: -math.log((self.counter.get((*context, token), 0) + 1) / denominator)
