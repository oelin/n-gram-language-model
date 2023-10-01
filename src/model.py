from typing import Any, Callable, Sequence
import math
from collections import Counter
from nltk.util import ngrams


LogProbabilityDistribution = Callable[[Any], float]
LanguageModel = Callable[[Sequence], LogProbabilityDistribution]


class NGramLanguageModel:
    """
    N-gram language model with Laplace smoothing.
    
    Parameters
    ----------
    
    vocabulary_size: int - The size of the vocabulary.
    context_size: int - The size of the context window.

    Attributes
    ----------

    vocabulary_size: int - The size of the vocabulary.
    context_size: int - The size of the context window.
    counter: Counter - A collection of n-gram counts.

    Examples
    --------

    >>> string = 'the quick brown fox jumps over the lazy dog.'
    >>> vocabulary = set(string)
    >>> model = NGramLanguageModel(
            vocabulary_size=len(vocabulary), 
            context_size=2,
        )
    >>> model.fit(string)
    >>> model('the quick brown')('f') 
    -3.332204510175204
    """

    def __init__(self, vocabulary_size: int, context_size: int) -> None:
        """Initializes the model."""
        
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.counter = Counter()
    
    def fit(self, sequence: Sequence) -> None:
        """Fits the model to a sequence using MLE.
        
        Parameters
        ----------

        sequence: Sequence - the sequence.

        Returns
        -------

        None
        """

        self.counter.clear()

        for n in range(1, self.context_size + 2):
            self.counter.update(ngrams(sequence, n))
    
    def __call__(self, sequence: Sequence) -> LogProbabilityDistribution:
        """Returns a Laplace smoothed log probability distribtion over tokens 
        appearing after a given sequence.

        Parameters
        ----------

        sequence: Sequence - the sequence.

        Returns
        -------

        LogProbabilityDistribution - the log probability distribution.
        """

        sequence = (*sequence[-self.context_size :], )
        denominator = self.counter.get(sequence, 0)
        denominator = denominator + self.vocabulary_size  # Laplace smoothing.

        def log_probability_distribution(token: Any) -> float:
            """A Laplace smoothed log probability distribtion over tokens appearing 
            after the sequence.

            Parameters
            ----------

            tokens: Any - a token.

            Returns
            -------

            float: The log probability of the token appearing after the sequence.
            """

            numerator = self.counter.get((*sequence, token), 0)
            numerator = numerator + 1  # Laplace smoothing.

            return math.log(numerator / denominator)
        return log_probability_distribution
