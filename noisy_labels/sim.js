/* Climbing on noisy labels: the simulation behind index.html. See MODEL.md. */
(function (root) {
  'use strict';

  const TEAM_COUNT = 3;
  const PATTERN_COUNT = 30;
  const MEMORIZED_ANSWER_RATE = 0.95;
  const RULE_LIFT = 0.5;
  const REVIEWER_CATCH_RATE = 0.8;
  const STRATEGY_GENERALITIES = [0.1, 0.3, 0.5, 0.7];
  const CLEAN = { rate: 0, kind: 'random' };

  // ===== checks =====
  function requireNumber(name, value, minimum, maximum) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
      throw new Error(`${name} must be a number in [${minimum}, ${maximum}], got ${value}`);
    }
  }
  function requireInteger(name, value, minimum, maximum) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
      throw new Error(`${name} must be an integer in [${minimum}, ${maximum}], got ${value}`);
    }
  }
  function requireNoise(noise) {
    if (noise === null || typeof noise !== 'object') throw new Error(`noise must be an object, got ${noise}`);
    requireNumber('noise.rate', noise.rate, 0, 0.5);
    if (noise.kind !== 'random' && noise.kind !== 'habit') {
      throw new Error(`noise.kind must be 'random' or 'habit', got ${noise.kind}`);
    }
  }

  // ===== random numbers =====
  function mulberry32(seed) {
    requireInteger('seed', seed, 0, 2 ** 32 - 1);
    let state = seed;
    return function () {
      state |= 0; state = state + 0x6D2B79F5 | 0;
      let t = Math.imul(state ^ state >>> 15, 1 | state);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function samplePattern(rng, frequencies) {
    const draw = rng();
    let cumulative = 0;
    for (let pattern = 0; pattern < frequencies.length; pattern++) {
      cumulative += frequencies[pattern];
      if (draw < cumulative) return pattern;
    }
    return frequencies.length - 1;
  }
  // One of the two teams other than `team`, chosen by a uniform draw in [0, 1).
  function otherTeam(team, draw) {
    return (team + (draw < 0.5 ? 1 : 2)) % TEAM_COUNT;
  }
  function shuffled(rng, items) {
    const copy = items.slice();
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }
  const sum = values => values.reduce((total, value) => total + value, 0);
  const mean = values => {
    if (values.length === 0) throw new Error('mean of an empty list');
    return sum(values) / values.length;
  };
  const indexOfMax = values => values.reduce((best, value, index) => (value > values[best] ? index : best), 0);

  // ===== world, sets, prompt =====
  function makeWorld(rng) {
    const weights = Array.from({ length: PATTERN_COUNT }, (_, index) => 1 / Math.pow(index + 1, 0.9));
    const total = sum(weights);
    const frequencies = weights.map(weight => weight / total);
    const trueTeams = frequencies.map((_, pattern) => pattern % TEAM_COUNT);
    const startCorrectRates = frequencies.map(() => 0.6 + 0.35 * rng());
    const habitOrder = shuffled(rng, frequencies.map((_, pattern) => pattern));
    const habitTeams = trueTeams.map(team => otherTeam(team, rng()));
    return { frequencies, trueTeams, startCorrectRates, habitOrder, habitTeams };
  }

  // The patterns a labeler with this habit always gets wrong: about `rate` of the traffic.
  function habitPatterns(world, rate) {
    requireNumber('rate', rate, 0, 0.5);
    const chosen = new Set();
    let total = 0;
    for (const pattern of world.habitOrder) {
      if (total + world.frequencies[pattern] <= rate + 0.005) {
        chosen.add(pattern);
        total += world.frequencies[pattern];
      }
    }
    return chosen;
  }

  // Three draws per ticket whatever the noise, so the same seed gives the same
  // tickets at every rate, and a higher random rate only adds wrong labels.
  function drawTickets(rng, world, count, noise) {
    requireInteger('count', count, 1, 1000000);
    requireNoise(noise);
    const habit = noise.kind === 'habit' ? habitPatterns(world, noise.rate) : null;
    const tickets = [];
    for (let index = 0; index < count; index++) {
      const pattern = samplePattern(rng, world.frequencies);
      const noiseDraw = rng();
      const teamDraw = rng();
      const trueTeam = world.trueTeams[pattern];
      let label = trueTeam;
      if (noise.kind === 'random' && noiseDraw < noise.rate) label = otherTeam(trueTeam, teamDraw);
      if (noise.kind === 'habit' && habit.has(pattern)) label = world.habitTeams[pattern];
      tickets.push({ pattern, trueTeam, label, labelIsWrong: label !== trueTeam });
    }
    return tickets;
  }

  function startPrompt(world) {
    const teamRates = world.startCorrectRates.map((correctRate, pattern) => {
      const rates = Array(TEAM_COUNT).fill((1 - correctRate) / (TEAM_COUNT - 1));
      rates[world.trueTeams[pattern]] = correctRate;
      return rates;
    });
    return { teamRates, memorized: new Set() };
  }

  // What the prompt does on new tickets: memorized tickets play no part.
  function trueAccuracy(world, prompt) {
    return sum(world.frequencies.map((frequency, pattern) => frequency * prompt.teamRates[pattern][world.trueTeams[pattern]]));
  }

  function answer(rng, prompt, ticket) {
    const draw = rng();
    if (prompt.memorized.has(ticket)) {
      if (draw < MEMORIZED_ANSWER_RATE) return ticket.label;
      return otherTeam(ticket.label, (draw - MEMORIZED_ANSWER_RATE) / (1 - MEMORIZED_ANSWER_RATE));
    }
    const rates = prompt.teamRates[ticket.pattern];
    let cumulative = 0;
    for (let team = 0; team < TEAM_COUNT; team++) {
      cumulative += rates[team];
      if (draw < cumulative) return team;
    }
    return TEAM_COUNT - 1;
  }

  // One run of the prompt on a set, scored against its labels and against the truth.
  function scoreSet(rng, prompt, tickets) {
    let matchesLabel = 0;
    let matchesTruth = 0;
    for (const ticket of tickets) {
      const team = answer(rng, prompt, ticket);
      if (team === ticket.label) matchesLabel++;
      if (team === ticket.trueTeam) matchesTruth++;
    }
    return { againstLabels: matchesLabel / tickets.length, againstTruth: matchesTruth / tickets.length };
  }

  function moveToward(rates, team, lift) {
    for (let other = 0; other < rates.length; other++) {
      rates[other] = rates[other] * (1 - lift) + (other === team ? lift : 0);
    }
  }

  // ===== 1. the climb on the training set =====
  function climb({ rng, world, training, rounds, generality, reviewer, watchedSets = {} }) {
    requireInteger('rounds', rounds, 0, 50);
    requireNumber('generality', generality, 0, 1);
    if (typeof reviewer !== 'boolean') throw new Error(`reviewer must be true or false, got ${reviewer}`);
    const prompt = startPrompt(world);
    const history = [];
    for (let round = 0; round <= rounds; round++) {
      const answers = training.map(ticket => answer(rng, prompt, ticket));
      const missIndexes = [];
      answers.forEach((team, index) => { if (team !== training[index].label) missIndexes.push(index); });
      const row = {
        round,
        trainingScore: 1 - missIndexes.length / training.length,
        trueAccuracy: trueAccuracy(world, prompt),
        misses: missIndexes.length,
        labelErrorMisses: missIndexes.filter(index => training[index].labelIsWrong).length,
        relabeled: 0,
        rulesFromRightLabels: 0,
        rulesFromWrongLabels: 0,
        watched: {},
      };
      for (const [name, tickets] of Object.entries(watchedSets)) row.watched[name] = scoreSet(rng, prompt, tickets);
      history.push(row);
      if (round === rounds) break;
      for (const index of missIndexes) {
        const ticket = training[index];
        if (reviewer && ticket.labelIsWrong && rng() < REVIEWER_CATCH_RATE) {
          ticket.label = ticket.trueTeam;
          ticket.labelIsWrong = false;
          row.relabeled++;
          if (answers[index] === ticket.label) continue;
        }
        prompt.memorized.add(ticket);
        if (rng() < generality) {
          moveToward(prompt.teamRates[ticket.pattern], ticket.label, RULE_LIFT);
          if (ticket.labelIsWrong) row.rulesFromWrongLabels++;
          else row.rulesFromRightLabels++;
        }
      }
    }
    return { history, prompt };
  }

  // The whole pipeline: draw the three sets, climb on training, and score the
  // prompt on validation and test after every round. Every set has its own noise;
  // sets with the same habit at the same rate were labeled by the same labeler.
  function pipelineRun({ seed, trainingSize, validationSize, testSize, rounds, generality, reviewer, trainingNoise, validationNoise, testNoise }) {
    const setRng = mulberry32(seed);
    const world = makeWorld(setRng);
    const training = drawTickets(setRng, world, trainingSize, trainingNoise);
    const validation = drawTickets(setRng, world, validationSize, validationNoise);
    const test = drawTickets(setRng, world, testSize, testNoise);
    return climb({
      rng: mulberry32(seed + 1), world, training, rounds, generality, reviewer, watchedSets: { validation, test },
    }).history;
  }

  // Round-by-round averages over `runs` pipelines. The same seeds are used for
  // every setting, so every setting sees the same worlds and the same tickets.
  function pipelineSweep({ runs, ...settings }) {
    requireInteger('runs', runs, 1, 100000);
    const histories = Array.from({ length: runs }, (_, run) => pipelineRun({ seed: 1000 + run, ...settings }));
    return histories[0].map((_, round) => {
      const rows = histories.map(history => history[round]);
      const misses = sum(rows.map(row => row.misses));
      return {
        round,
        trainingScore: mean(rows.map(row => row.trainingScore)),
        validationScore: mean(rows.map(row => row.watched.validation.againstLabels)),
        validationScoreIfLabelsWereRight: mean(rows.map(row => row.watched.validation.againstTruth)),
        testScore: mean(rows.map(row => row.watched.test.againstLabels)),
        trueAccuracy: mean(rows.map(row => row.trueAccuracy)),
        misses: misses / runs,
        labelErrorShareOfMisses: misses === 0 ? 0 : sum(rows.map(row => row.labelErrorMisses)) / misses,
        rulesFromRightLabels: mean(rows.map(row => row.rulesFromRightLabels)),
        rulesFromWrongLabels: mean(rows.map(row => row.rulesFromWrongLabels)),
        relabeled: mean(rows.map(row => row.relabeled)),
      };
    });
  }

  // ===== 2. the validation set picks a strategy =====
  // Four climbs on clean training labels, one per strategy; the validation set
  // picks the one with the highest score. The climbs run once; at every rate the
  // validation set has the same tickets and the router gives the same answers,
  // so only the labels change.
  function pickStrategy({ seed, trainingSize, validationSize, rounds, validationRates }) {
    const setRng = mulberry32(seed);
    const world = makeWorld(setRng);
    const training = drawTickets(setRng, world, trainingSize, CLEAN);
    const climbRng = mulberry32(seed + 1);
    const prompts = STRATEGY_GENERALITIES.map(generality =>
      climb({ rng: climbRng, world, training, rounds, generality, reviewer: false }).prompt);
    const trueAccuracies = prompts.map(prompt => trueAccuracy(world, prompt));
    const best = indexOfMax(trueAccuracies);
    const byRate = validationRates.map(rate => {
      const validation = drawTickets(mulberry32(seed + 2), world, validationSize, { rate, kind: 'random' });
      const answerRng = mulberry32(seed + 3);
      const scores = prompts.map(prompt => scoreSet(answerRng, prompt, validation));
      const validationScores = scores.map(scored => scored.againstLabels);
      const picked = indexOfMax(validationScores);
      return {
        rate,
        validationScores,
        validationScoresIfLabelsWereRight: scores.map(scored => scored.againstTruth),
        picked,
        regret: trueAccuracies[best] - trueAccuracies[picked],
      };
    });
    return { trueAccuracies, best, byRate };
  }

  function validationSweep({ runs, rates, trainingSize, validationSize, rounds }) {
    requireInteger('runs', runs, 1, 100000);
    const picks = Array.from({ length: runs }, (_, run) => pickStrategy({
      seed: 5000 + run, trainingSize, validationSize, rounds, validationRates: rates,
    }));
    return rates.map((rate, rateIndex) => {
      const atRate = picks.map(pick => ({ ...pick.byRate[rateIndex], trueAccuracies: pick.trueAccuracies, best: pick.best }));
      return {
        rate,
        shareBestPicked: mean(atRate.map(pick => (pick.picked === pick.best ? 1 : 0))),
        meanRegret: mean(atRate.map(pick => pick.regret)),
        winnerScoreMinusTruth: mean(atRate.map(pick => pick.validationScores[pick.picked] - pick.trueAccuracies[pick.picked])),
      };
    });
  }

  // How much a real difference between two prompts shrinks when it is measured
  // against labels with this rate of random errors (three teams).
  function differenceShrink(rate) {
    requireNumber('rate', rate, 0, 0.5);
    return 1 - 1.5 * rate;
  }

  // ===== 3. one look at the test set =====
  function wilsonInterval(successes, trials, zScore = 1.96) {
    requireInteger('trials', trials, 1, 1e9);
    requireInteger('successes', successes, 0, trials);
    const share = successes / trials;
    const zSquared = zScore * zScore;
    const center = (share + zSquared / (2 * trials)) / (1 + zSquared / trials);
    const halfWidth = zScore * Math.sqrt(share * (1 - share) / trials + zSquared / (4 * trials * trials)) / (1 + zSquared / trials);
    return { low: center - halfWidth, high: center + halfWidth };
  }

  function measuredAccuracy(trueAccuracyValue, rate) {
    requireNumber('trueAccuracy', trueAccuracyValue, 0, 1);
    requireNumber('rate', rate, 0, 0.5);
    return trueAccuracyValue * (1 - rate) + (1 - trueAccuracyValue) * rate / 2;
  }

  // A frozen prompt that is right on a share `trueAccuracy` of new tickets,
  // scored once on `testSize` fresh tickets, `looks` times over.
  function testLooks({ seed, trueAccuracy: trueAccuracyValue, testSize, testNoise, looks }) {
    requireNumber('trueAccuracy', trueAccuracyValue, 0, 1);
    requireInteger('testSize', testSize, 1, 100000);
    requireInteger('looks', looks, 1, 1000000);
    requireNoise(testNoise);
    if (testNoise.kind !== 'random') throw new Error('testLooks models random label errors only');
    const rng = mulberry32(seed);
    const measuredScores = [];
    let covered = 0;
    for (let look = 0; look < looks; look++) {
      let hits = 0;
      for (let index = 0; index < testSize; index++) {
        const labelIsWrong = rng() < testNoise.rate;
        const routerIsRight = rng() < trueAccuracyValue;
        const wrongAnswerDraw = rng();
        if (!labelIsWrong && routerIsRight) hits++;
        if (labelIsWrong && !routerIsRight && wrongAnswerDraw < 0.5) hits++;
      }
      const interval = wilsonInterval(hits, testSize);
      if (interval.low <= trueAccuracyValue && trueAccuracyValue <= interval.high) covered++;
      measuredScores.push(hits / testSize);
    }
    return { measuredScores, coverage: covered / looks, meanMeasured: mean(measuredScores) };
  }

  const api = {
    TEAM_COUNT, PATTERN_COUNT, STRATEGY_GENERALITIES, CLEAN,
    mulberry32, makeWorld, habitPatterns, drawTickets, startPrompt, trueAccuracy, answer, scoreSet,
    climb, pipelineRun, pipelineSweep,
    pickStrategy, validationSweep, differenceShrink,
    wilsonInterval, measuredAccuracy, testLooks,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NoisyLabels = api;
})(typeof window !== 'undefined' ? window : globalThis);
