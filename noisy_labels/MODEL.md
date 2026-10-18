# Climbing on noisy labels: the model

The simulation behind `index.html`. The code is in `sim.js`; the tests are in `tests/sim.test.cjs`.

## Entities

| Entity | What it holds |
|---|---|
| **Team** | One of 3 teams a ticket can be routed to (Billing, Access, Network). |
| **Pattern** | A kind of ticket. 30 patterns with long-tailed frequencies (weight `1 / (i+1)^0.9`), each owned by one true team. |
| **Prompt** | The router's current behaviour: for each pattern, the share of its tickets routed to each team (`teamRates`). At the start a pattern goes to its true team 60% to 95% of the time, and to the other two teams equally otherwise. |
| **Ticket** | A pattern, its true team, its **label** (the team in the dataset), and whether the label is wrong. A ticket can be **memorized** by the prompt: a rule written for that exact ticket makes the router answer the ticket's label 95% of the time. |
| **Set** | Training (200), validation (100) or test (100): tickets drawn from the pattern frequencies, labeled with the set's own noise. |

```
Pattern 1 ── * Ticket * ── 1 Set (training | validation | test)
   │                │
   └─ true team     └─ label (wrong with probability q of its set)
Prompt ── teamRates per Pattern ── memorized Tickets
```

## Label noise

- **Random slips:** each ticket's label is wrong with probability `q`; a wrong label is one of the other two teams, uniformly.
- **A labeler's habit:** the labeler gets a few patterns wrong every time, always with the same wrong team. The patterns are chosen so they carry about `q` of the traffic. Two sets with the same habit at the same rate were labeled by the same labeler and share their mistakes.

Measured accuracy against randomly wrong labels is `a(1−q) + (1−a)·q/2`, where `a` is the true accuracy. A real difference `Δ` between two prompts shows as `Δ·(1 − 1.5q)`.

## The climb

Each round the router runs on the training tickets; every ticket whose answer differs from its label is a miss, and it is patched. A patch memorizes the ticket. With probability `g` (fix generality) it also writes a rule for the ticket's pattern, which moves the pattern's answers halfway toward the team in the label. If the label was right, the pattern gets better on new tickets; if it was wrong, the pattern moves toward the wrong team. Optional check: before a miss is patched, a reviewer catches a wrong label 4 times out of 5 and relabels the ticket instead.

Validation and test are scored after every round, against their labels and against the truth. Nobody reads them, so they never change the climb.

## The four experiments

1. **Wrong labels in training.** Validation and test are clean. Compare the training score, the true accuracy, and the true accuracy of the same climb on clean labels.
2. **Wrong labels in validation.** Training and test are clean.
   - *Watching the climb:* the training–validation gap is compared with the real overfitting, training score minus true accuracy.
   - *Picking a strategy:* four climbs with `g` = 10%, 30%, 50% and 70% on the same clean training set; validation picks the highest score. At every rate the validation tickets and the router's answers are the same, and only the labels change.
3. **Wrong labels in test.** A frozen prompt with true accuracy `a` is scored once on `n` fresh test tickets with random wrong labels at rate `q`. Repeated many times, this gives the spread of reported numbers and how often their 95% Wilson interval contains `a`.
4. **One labeler for all three sets.** The same habit in training, validation and test, compared with validation and test labeled by a second labeler whose labels are right.

True accuracy is always `Σ frequency × share routed to the true team`: what the prompt does on new tickets, which memorized tickets do not affect.

Sweeps use fixed seeds (1000 + run for the pipeline, 5000 + run for the strategy pick), so the numbers on the page and in the slides can be reproduced.

## What the model leaves out

- A rule written for one ticket never misfires on other tickets; only pattern rules change true accuracy.
- The router's answers are redrawn on every run, as with an LLM sampled at a temperature above zero.
- The reviewer never mistakes a right label for a wrong one.
- Wrong labels are either random or a habit; real labels mix both, and include tickets that fit two teams.
