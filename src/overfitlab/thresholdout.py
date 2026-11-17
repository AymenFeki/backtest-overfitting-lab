"""Thresholdout: a holdout that can be asked many times (Dwork et al., 2015).

Paper: Dwork, Feldman, Hardt, Pitassi, Reingold and Roth, "Generalization in
Adaptive Data Analysis and Holdout Reuse" (arXiv 1506.02629), Section 4.1,
Figure 1.

The problem. A holdout is an honest test only while the analyst's choices do
not depend on it. Every answer read from it and used for the next decision
leaks a little of its noise into the research, and after enough rounds the
chosen model fits the holdout's noise as well.

The idea. Answer from the training data whenever training and holdout agree,
because then the answer tells the analyst nothing new about the holdout. Only
when they disagree by more than a threshold is the holdout value given, with
noise added, and each such answer is paid for from a fixed budget. When the
budget is used up the holdout stops answering.

The algorithm, with Lap(b) a Laplace draw of scale b:

    threshold' = T + Lap(2 sigma)
    for each query, with training value a_t and holdout value a_h:
        if the budget is below 1: no answer
        else if |a_h - a_t| > threshold' + Lap(4 sigma):
            budget -= 1;  threshold' = T + Lap(2 sigma)
            answer a_h + Lap(sigma)
        else:
            answer a_t

The noise is what limits how much one answer can reveal: the analyst cannot
tell on which side of the threshold a query fell from the value alone.

What the guarantee needs (Theorem 25 of the paper): queries that are averages
of a function with values in [0, 1] over independent samples, and a holdout
much larger than budget / tolerance^2. A Sharpe ratio on 756 autocorrelated
daily returns meets neither condition, so this project uses Thresholdout as a
rule of thumb and measures what it does. The paper's own experiment also used
settings far from the theorem's.

Source note: the algorithm and theorem were checked against the arXiv HTML
version through an automated summary, not read line by line, and agree with
the statement above.
"""

import numpy as np


class Thresholdout:
    """Answers queries about one holdout; keeps the budget and noisy threshold."""

    def __init__(
        self,
        threshold: float,
        sigma: float,
        budget: int,
        rng: np.random.Generator,
    ) -> None:
        self.threshold = threshold
        self.sigma = sigma
        self.budget = budget
        self.rng = rng
        self._noisy_threshold = threshold + rng.laplace(0.0, 2 * sigma)

    def answer(self, train_value: float, holdout_value: float) -> float | None:
        """The answer to one query, or None once the budget is used up."""
        if self.budget < 1:
            return None
        gap = abs(holdout_value - train_value)
        if gap > self._noisy_threshold + self.rng.laplace(0.0, 4 * self.sigma):
            self.budget -= 1
            self._noisy_threshold = self.threshold + self.rng.laplace(
                0.0, 2 * self.sigma
            )
            return holdout_value + self.rng.laplace(0.0, self.sigma)
        return train_value
