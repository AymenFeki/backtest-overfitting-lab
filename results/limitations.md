# Limitations

The full list. The README repeats the most important ones.

- **Simulated markets.** They have fat tails, volatility clustering, correlated
  assets, and optionally regime switching and alpha decay, but are far simpler
  than real markets.
- **Regime switching and signal strength are only partly separated.** Scaling
  the signal back up matches the planted rule's Sharpe, not how hard the world
  is for the grid; the comparison is clean for one of three signal types
  ([Stage 3](stage3.md)).
- **Daily data only.**
- **Thresholdout runs outside its guarantee.** The theorem needs statistics
  bounded in [0, 1] and a holdout far larger than the budget over the squared
  accuracy; a Sharpe ratio on 756 days is neither. The settings were chosen in
  standard errors before the runs, and the paper's own figures (10,000
  samples, threshold 0.04, noise 0.01) come from a summary of the paper, not
  from the original ([Stage 5](stage5.md)).
- **Experiment 3 gave the agent nothing to game.** No strategy ever passed a
  check, so the feedback both arms saw was constant. The design does not test
  what happens when the checks can be passed ([Stage 5](stage5.md)).
- **Small agent samples.** 15 worlds per arm in Experiments 2 and 3: one world
  moves a share by 6.7 points.
- **Two methods rest on secondary sources.** The CPCV chapter (Lopez de Prado
  2018) and the paper behind the effective number of trials (Galwey 2009) are
  not freely available. CPCV follows a library's documentation and my own
  derivation of the split and path counts; the effective-trials formula comes
  from an R package's documentation. Neither was checked against the original
  ([Stage 2](stage2.md)).
- **The CPCV accept rule is my own.** The book gives the paths, not a decision
  rule; the one used here is stated and tested in the [Stage 2 note](stage2.md).
- **Alpha decay is tested in one setting.** One half-life (5 years) on one
  signal strength (obvious). A slower decay or a weaker signal would give
  different numbers; the direction of the result is what carries over
  ([Stage 3](stage3.md)).
- **One model, one prompt.** The agent results come from Claude Haiku 5.5 with
  one prompt, 30 worlds per type. Another model or prompt could propose
  different strategies ([Stage 4](stage4.md)).
- **Not finished.** The real-data check (Stage 6) is still to come.
