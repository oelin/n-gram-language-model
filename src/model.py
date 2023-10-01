from typing import List, Callable, Sequence
import math

from collections import Counter
from nltk.util import ngrams


String = Sequence[Any]
ProbabilityDistribution = Callable[[Any], float]
LanguageModel = Callable[[String], ProbabilityDistribution]


class NGramLanguageModel:
    """
    N-gram language model with Laplace smoothing.

    Parameters:
    - vocabulary_size (int): The size of the vocabulary.
    - context_size (int, optional): The size of the context for n-grams. Default is 2.
    """

    def __init__(self, vocabulary_size: int, context_size: int = 2) -> None:
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.counter = Counter()

    def fit(self, string: String) -> None:
        """
        Fit the model to the input string.

        Parameters:
        - string (Sequence[Any]): The input sequence.
        """

        observed_ngrams = []

        for i in range(1, self.context_size + 2):
            observed_ngrams += ngrams(string, i)

        self.counter = Counter(observed_ngrams)
        self.counter[('',)] = len(string)  # A "zero-gram" occurs before every token.

    def __call__(self, string: String) -> ProbabilityDistribution:
        """
        Compute the conditional probability distribution p(token|string).

        Parameters:
        - string (Sequence[Any]): The input sequence.

        Returns:
        - ProbabilityDistribution: A function that calculates the log probability of a token.
        """

        context = (*string[-self.context_size :],)
        context_count = self.counter.get(context, 0)
        denominator = context_count + self.vocabulary_size

        def probability_distribution(token: Any) -> float:
            numerator = self.counter.get((*context, token), 0) + 1
            return -math.log(numerator / denominator)

        return probability_distribution
