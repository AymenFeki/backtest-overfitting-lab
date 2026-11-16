import numpy as np

from overfitlab.thresholdout import Thresholdout


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_without_noise_the_rule_is_exact():
    guard = Thresholdout(threshold=0.5, sigma=0.0, budget=3, rng=_rng())
    # Training and holdout agree within the threshold: the training value.
    assert guard.answer(1.0, 1.4) == 1.0
    assert guard.answer(1.0, 0.5) == 1.0  # a gap of exactly 0.5 is not above it
    # They disagree: the holdout value, in both directions.
    assert guard.answer(1.0, 0.2) == 0.2
    assert guard.answer(-1.0, 0.2) == 0.2


def test_only_holdout_answers_use_the_budget_and_then_it_stops():
    guard = Thresholdout(threshold=0.5, sigma=0.0, budget=2, rng=_rng())
    for _ in range(50):  # agreeing queries are free, however many
        assert guard.answer(1.0, 1.1) == 1.0
    assert guard.budget == 2
    assert guard.answer(1.0, 0.0) == 0.0 and guard.budget == 1
    assert guard.answer(1.0, 0.0) == 0.0 and guard.budget == 0
    # Used up: no answer at all, also for a query that would have been free.
    assert guard.answer(1.0, 0.0) is None
    assert guard.answer(1.0, 1.1) is None
    assert guard.budget == 0


def test_holdout_answers_carry_laplace_noise_of_scale_sigma():
    sigma, n = 0.1, 20_000
    guard = Thresholdout(threshold=0.5, sigma=sigma, budget=n, rng=_rng(1))
    # A gap of 100 is far above the threshold whatever the noise.
    noise = np.array([guard.answer(100.0, 0.0) for _ in range(n)])
    assert guard.budget == 0
    # Laplace with scale b: mean 0, standard deviation b * sqrt(2), and
    # mean absolute value b. Bands are about 4 standard errors wide.
    assert abs(noise.mean()) < 4 * sigma * np.sqrt(2 / n)
    assert abs(noise.std() - sigma * np.sqrt(2)) < 0.005
    assert abs(np.abs(noise).mean() - sigma) < 0.003
    assert len(np.unique(noise)) == n  # a fresh draw every time


def test_the_threshold_itself_is_noisy():
    # A gap exactly at the threshold is flagged when the two noise terms
    # (scales 2 sigma and 4 sigma, both symmetric around zero) sum to less
    # than zero: half of the time. A gap far inside it is never flagged.
    n, flagged, far = 4000, 0, 0
    for seed in range(n):
        guard = Thresholdout(threshold=0.5, sigma=0.01, budget=1, rng=_rng(seed))
        guard.answer(0.0, 0.5)
        flagged += guard.budget == 0
        guard = Thresholdout(threshold=0.5, sigma=0.01, budget=1, rng=_rng(seed))
        guard.answer(0.0, 0.1)
        far += guard.budget == 0
    assert abs(flagged / n - 0.5) < 3 * np.sqrt(0.25 / n)
    assert far == 0


def test_same_seed_same_answers():
    def answers(seed: int) -> list:
        guard = Thresholdout(threshold=0.5, sigma=0.2, budget=5, rng=_rng(seed))
        values = np.random.default_rng(9).normal(size=(30, 2))
        return [guard.answer(a, b) for a, b in values]

    assert answers(3) == answers(3)
    assert answers(3) != answers(4)


def _accuracy(x: np.ndarray, y: np.ndarray, signs: np.ndarray) -> float:
    return float((np.sign(x @ signs) == y).mean())


def test_no_signal_demonstration_of_the_paper():
    """Section 5 of the paper, at a fifth of its size.

    Labels are coin flips and no variable predicts them, so any classifier is
    right half of the time on fresh data. The analyst picks the variables
    whose correlation with the label has the same sign in the training and
    the holdout set, and votes with them. Read directly, the holdout then
    reports an accuracy well above 50%: it was used to choose the variables.
    Asked through Thresholdout, the holdout only repeats the training
    correlations (they agree within the threshold), the choice no longer
    depends on it, and the reported accuracy is close to the fresh one.
    """
    rng = np.random.default_rng(2015)
    n, d, k = 2000, 2000, 50
    x = {name: rng.standard_normal((n, d)) for name in ("train", "holdout", "fresh")}
    y = {name: rng.choice([-1.0, 1.0], size=n) for name in x}
    w_train = x["train"].T @ y["train"] / n
    w_holdout = x["holdout"].T @ y["holdout"] / n
    cut = 1 / np.sqrt(n)  # one standard error of such a correlation

    def classifier(holdout_view: np.ndarray) -> np.ndarray:
        agree = (w_train * holdout_view > 0) & (np.abs(w_train) >= cut)
        agree &= np.abs(holdout_view) >= cut
        chosen = np.argsort(-np.abs(w_train) * agree)[:k]
        signs = np.zeros(d)
        signs[chosen] = np.sign(w_train[chosen])
        return signs

    # Standard holdout: the analyst reads every holdout correlation.
    signs = classifier(w_holdout)
    reported = _accuracy(x["holdout"], y["holdout"], signs)
    fresh = _accuracy(x["fresh"], y["fresh"], signs)
    assert reported - fresh > 0.05 and abs(fresh - 0.5) < 0.04

    # Thresholdout: threshold 4 standard errors, as in the paper.
    guard = Thresholdout(threshold=4 * cut, sigma=cut / 4, budget=d, rng=rng)
    view = np.array([guard.answer(a, b) for a, b in zip(w_train, w_holdout)])
    signs = classifier(view)
    train = _accuracy(x["train"], y["train"], signs)
    reported = guard.answer(train, _accuracy(x["holdout"], y["holdout"], signs))
    fresh = _accuracy(x["fresh"], y["fresh"], signs)
    assert train > 0.6  # the training set is overfitted, as before
    assert abs(reported - fresh) < 0.04
