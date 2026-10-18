// Run with: node --test noisy_labels/tests/sim.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const sim = require('../sim.js');

const CLEAN = { rate: 0, kind: 'random' };
const PIPELINE = { trainingSize: 200, validationSize: 100, testSize: 100, rounds: 8, generality: 0.4, reviewer: false };

function worldAndTickets(seed, count, noise) {
  const rng = sim.mulberry32(seed);
  const world = sim.makeWorld(rng);
  return { world, tickets: sim.drawTickets(rng, world, count, noise) };
}
const share = (items, predicate) => items.filter(predicate).length / items.length;

test('the random numbers repeat for a seed and stay in [0, 1)', () => {
  const first = sim.mulberry32(42), second = sim.mulberry32(42);
  for (let i = 0; i < 1000; i++) {
    const value = first();
    assert.equal(value, second());
    assert.ok(value >= 0 && value < 1);
  }
});

test('the world has 30 long-tailed patterns, and habits point to a wrong team', () => {
  const world = sim.makeWorld(sim.mulberry32(1));
  assert.equal(world.frequencies.length, sim.PATTERN_COUNT);
  assert.ok(Math.abs(world.frequencies.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  for (let p = 1; p < world.frequencies.length; p++) assert.ok(world.frequencies[p] < world.frequencies[p - 1]);
  world.trueTeams.forEach((team, p) => {
    assert.ok([0, 1, 2].includes(team));
    assert.notEqual(world.habitTeams[p], team);
    assert.ok(world.startCorrectRates[p] >= 0.6 && world.startCorrectRates[p] <= 0.95);
  });
});

test('random noise makes about the given share of labels wrong, spread over the other two teams', () => {
  const { tickets } = worldAndTickets(3, 40000, { rate: 0.2, kind: 'random' });
  const wrong = tickets.filter(ticket => ticket.labelIsWrong);
  assert.ok(Math.abs(wrong.length / tickets.length - 0.2) < 0.01);
  wrong.forEach(ticket => assert.notEqual(ticket.label, ticket.trueTeam));
  const nextTeam = share(wrong, ticket => ticket.label === (ticket.trueTeam + 1) % 3);
  assert.ok(Math.abs(nextTeam - 0.5) < 0.02);
  tickets.filter(ticket => !ticket.labelIsWrong).forEach(ticket => assert.equal(ticket.label, ticket.trueTeam));
});

test('a habit always gives the same wrong team on its patterns, about the given share of traffic', () => {
  const { world, tickets } = worldAndTickets(4, 40000, { rate: 0.1, kind: 'habit' });
  const habit = sim.habitPatterns(world, 0.1);
  const habitTraffic = [...habit].reduce((total, p) => total + world.frequencies[p], 0);
  assert.ok(habitTraffic <= 0.105 && habitTraffic >= 0.08, `habit covers ${habitTraffic}`);
  tickets.forEach(ticket => {
    assert.equal(ticket.labelIsWrong, habit.has(ticket.pattern));
    if (ticket.labelIsWrong) assert.equal(ticket.label, world.habitTeams[ticket.pattern]);
  });
  assert.equal(sim.habitPatterns(world, 0).size, 0);
});

test('the same seed draws the same tickets at every rate, and a higher rate only adds wrong labels', () => {
  const low = worldAndTickets(5, 2000, { rate: 0.1, kind: 'random' }).tickets;
  const high = worldAndTickets(5, 2000, { rate: 0.3, kind: 'random' }).tickets;
  low.forEach((ticket, index) => {
    assert.equal(ticket.pattern, high[index].pattern);
    if (ticket.labelIsWrong) assert.equal(ticket.label, high[index].label);
  });
  assert.ok(share(high, t => t.labelIsWrong) > share(low, t => t.labelIsWrong));
});

test('measured accuracy against noisy labels matches a(1-q) + (1-a)q/2', () => {
  const { world, tickets } = worldAndTickets(6, 60000, { rate: 0.2, kind: 'random' });
  const prompt = sim.startPrompt(world);
  const truth = sim.trueAccuracy(world, prompt);
  const scored = sim.scoreSet(sim.mulberry32(7), prompt, tickets);
  assert.ok(Math.abs(scored.againstTruth - truth) < 0.01);
  assert.ok(Math.abs(scored.againstLabels - sim.measuredAccuracy(truth, 0.2)) < 0.01);
  assert.ok(Math.abs(sim.measuredAccuracy(0.8, 0.1) - 0.73) < 1e-12);
});

test('a real difference measured on noisy labels shrinks by 1 - 1.5q', () => {
  const { world, tickets } = worldAndTickets(8, 60000, { rate: 0.2, kind: 'random' });
  const weaker = sim.startPrompt(world);
  const stronger = sim.startPrompt(world);
  stronger.teamRates = stronger.teamRates.map((rates, p) => {
    const correct = Math.min(1, world.startCorrectRates[p] + 0.1);
    return rates.map((_, team) => (team === world.trueTeams[p] ? correct : (1 - correct) / 2));
  });
  const realDifference = sim.trueAccuracy(world, stronger) - sim.trueAccuracy(world, weaker);
  const rng = sim.mulberry32(9);
  const measuredDifference = sim.scoreSet(rng, stronger, tickets).againstLabels - sim.scoreSet(rng, weaker, tickets).againstLabels;
  assert.ok(Math.abs(measuredDifference - realDifference * sim.differenceShrink(0.2)) < 0.01);
});

test('with no generality the climb raises the training score but not the true accuracy', () => {
  const history = sim.pipelineRun({ seed: 10, ...PIPELINE, generality: 0, trainingNoise: CLEAN, validationNoise: CLEAN, testNoise: CLEAN });
  const first = history[0], last = history[history.length - 1];
  assert.equal(last.trueAccuracy, first.trueAccuracy);
  assert.ok(last.trainingScore > first.trainingScore + 0.1);
  assert.equal(history.length, PIPELINE.rounds + 1);
});

test('on clean labels every rule moves toward the true team, so true accuracy never falls', () => {
  const history = sim.pipelineRun({ seed: 11, ...PIPELINE, generality: 1, trainingNoise: CLEAN, validationNoise: CLEAN, testNoise: CLEAN });
  for (let round = 1; round < history.length; round++) assert.ok(history[round].trueAccuracy >= history[round - 1].trueAccuracy);
  history.forEach(row => { assert.equal(row.rulesFromWrongLabels, 0); assert.equal(row.labelErrorMisses, 0); });
});

test('a climb on a habit learns it: the habit patterns end below where they started', () => {
  const rng = sim.mulberry32(12);
  const world = sim.makeWorld(rng);
  const training = sim.drawTickets(rng, world, 400, { rate: 0.2, kind: 'habit' });
  const { prompt } = sim.climb({ rng: sim.mulberry32(13), world, training, rounds: 8, generality: 1, reviewer: false });
  const habit = sim.habitPatterns(world, 0.2);
  const seen = new Set(training.map(ticket => ticket.pattern));
  const habitSeen = [...habit].filter(p => seen.has(p));
  assert.ok(habitSeen.length > 0);
  habitSeen.forEach(p => assert.ok(prompt.teamRates[p][world.trueTeams[p]] < world.startCorrectRates[p]));
});

test('training noise: the training score ends the same, the real gain shrinks, rules are written from wrong labels', () => {
  const settings = { runs: 60, ...PIPELINE, validationNoise: CLEAN, testNoise: CLEAN };
  const clean = sim.pipelineSweep({ ...settings, trainingNoise: CLEAN });
  const noisy = sim.pipelineSweep({ ...settings, trainingNoise: { rate: 0.2, kind: 'random' } });
  const last = rows => rows[rows.length - 1];
  assert.ok(Math.abs(last(clean).trainingScore - last(noisy).trainingScore) < 0.03);
  const cleanGain = last(clean).trueAccuracy - clean[0].trueAccuracy;
  const noisyGain = last(noisy).trueAccuracy - noisy[0].trueAccuracy;
  assert.ok(noisyGain < cleanGain - 0.05, `gains ${cleanGain} vs ${noisyGain}`);
  assert.ok(noisy.reduce((total, row) => total + row.rulesFromWrongLabels, 0) > 10);
  assert.ok(Math.abs(last(noisy).validationScore - last(noisy).trueAccuracy) < 0.02, 'clean validation tracks the truth');
});

test('the reviewer relabels misses and keeps most of the real gain', () => {
  const settings = { runs: 60, ...PIPELINE, trainingNoise: { rate: 0.2, kind: 'random' }, validationNoise: CLEAN, testNoise: CLEAN };
  const without = sim.pipelineSweep(settings);
  const withReviewer = sim.pipelineSweep({ ...settings, reviewer: true });
  const total = (rows, key) => rows.reduce((sum, row) => sum + row[key], 0);
  assert.ok(total(withReviewer, 'relabeled') > 20);
  assert.ok(total(withReviewer, 'rulesFromWrongLabels') < total(without, 'rulesFromWrongLabels') / 3);
  assert.ok(withReviewer[withReviewer.length - 1].trueAccuracy > without[without.length - 1].trueAccuracy + 0.05);
});

test('validation noise: the validation score drops below the truth and the training-validation gap widens', () => {
  const settings = { runs: 60, ...PIPELINE, trainingNoise: CLEAN, testNoise: CLEAN };
  const rows = sim.pipelineSweep({ ...settings, validationNoise: { rate: 0.2, kind: 'random' } });
  const last = rows[rows.length - 1];
  assert.ok(Math.abs(last.validationScore - sim.measuredAccuracy(last.trueAccuracy, 0.2)) < 0.02);
  assert.ok(Math.abs(last.validationScoreIfLabelsWereRight - last.trueAccuracy) < 0.02);
  assert.ok(last.trainingScore - last.validationScore > 0.15);
  assert.ok(last.trainingScore - last.trueAccuracy < 0.04);
});

test('validation noise changes only the labels, and the best strategy is picked less often', () => {
  const pick = sim.pickStrategy({ seed: 5003, trainingSize: 200, validationSize: 100, rounds: 8, validationRates: [0, 0.3] });
  assert.equal(pick.trueAccuracies.length, sim.STRATEGY_GENERALITIES.length);
  assert.deepEqual(pick.byRate[1].validationScoresIfLabelsWereRight, pick.byRate[0].validationScoresIfLabelsWereRight);
  assert.deepEqual(pick.byRate[0].validationScores, pick.byRate[0].validationScoresIfLabelsWereRight);
  const sweep = sim.validationSweep({ runs: 300, rates: [0, 0.3], trainingSize: 200, validationSize: 100, rounds: 8 });
  assert.ok(sweep[1].shareBestPicked < sweep[0].shareBestPicked - 0.05);
  assert.ok(sweep[1].meanRegret > sweep[0].meanRegret);
});

test('the Wilson interval matches known values', () => {
  const interval = sim.wilsonInterval(80, 100);
  assert.ok(Math.abs(interval.low - 0.7112) < 1e-4);
  assert.ok(Math.abs(interval.high - 0.8666) < 1e-4);
});

test('test noise: the reported number is biased and its interval stops containing the truth', () => {
  const settings = { seed: 14, trueAccuracy: 0.85, testSize: 100, looks: 3000 };
  const clean = sim.testLooks({ ...settings, testNoise: CLEAN });
  const noisy = sim.testLooks({ ...settings, testNoise: { rate: 0.2, kind: 'random' } });
  assert.ok(Math.abs(clean.coverage - 0.95) < 0.03);
  assert.ok(Math.abs(noisy.meanMeasured - sim.measuredAccuracy(0.85, 0.2)) < 0.005);
  assert.ok(noisy.coverage < 0.1);
  assert.equal(noisy.measuredScores.length, 3000);
});

test('one labeler with a habit on every set: validation and test read above the truth', () => {
  const habit = { rate: 0.2, kind: 'habit' };
  const rows = sim.pipelineSweep({ runs: 60, ...PIPELINE, trainingNoise: habit, validationNoise: habit, testNoise: habit });
  const last = rows[rows.length - 1];
  assert.ok(last.validationScore > last.trueAccuracy + 0.05);
  assert.ok(last.testScore > last.trueAccuracy + 0.05);
  assert.ok(Math.abs(last.validationScoreIfLabelsWereRight - last.trueAccuracy) < 0.02);
});

test('bad inputs fail loudly', () => {
  const rng = sim.mulberry32(1);
  const world = sim.makeWorld(rng);
  assert.throws(() => sim.drawTickets(rng, world, 10, { rate: 0.6, kind: 'random' }), /noise.rate/);
  assert.throws(() => sim.drawTickets(rng, world, 10, { rate: 0.1, kind: 'sloppy' }), /noise.kind/);
  assert.throws(() => sim.drawTickets(rng, world, 0, CLEAN), /count/);
  assert.throws(() => sim.drawTickets(rng, world, 10, undefined), /noise must be an object/);
  assert.throws(() => sim.mulberry32(-1), /seed/);
  assert.throws(() => sim.climb({ rng, world, training: [], rounds: 3, generality: 1.5, reviewer: false }), /generality/);
  assert.throws(() => sim.climb({ rng, world, training: [], rounds: 3, generality: 0.5, reviewer: 'yes' }), /reviewer/);
  assert.throws(() => sim.testLooks({ seed: 1, trueAccuracy: 0.8, testSize: 100, looks: 10, testNoise: { rate: 0.1, kind: 'habit' } }), /random label errors only/);
  assert.throws(() => sim.wilsonInterval(120, 100), /successes/);
});
