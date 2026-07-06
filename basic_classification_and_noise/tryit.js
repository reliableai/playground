// In-browser "Try it" classifier for the L1 lab — no backend needed.
// Mirrors serve.py exactly: a TF-IDF + logistic-regression decision_function for the
// label/scores, and a full-vocab TF-IDF nearest-example lookup for the 2D placement.
// Model weights come from predictions/models.js (built by build_model_js.py).
//
// sklearn equivalences this file must preserve:
//   - tokenizer: lowercase + token_pattern \b\w\w+\b  (runs of >=2 word chars)
//   - ngrams joined with a single space
//   - tf = raw count, x = tf * idf, then L2-normalised per text
//   - decision_function = x . coef[c] + intercept[c], one score per class
(function (global) {
  "use strict";

  function tokenize(text) {
    return text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
  }

  function grams(tokens, nmax) {
    if (nmax <= 1) return tokens;
    const out = tokens.slice();
    for (let n = 2; n <= nmax; n++)
      for (let i = 0; i + n <= tokens.length; i++)
        out.push(tokens.slice(i, i + n).join(" "));
    return out;
  }

  // sparse L2-normalised tf-idf vector as a Map(featureIndex -> value)
  function vectorize(text, model) {
    const counts = new Map();
    for (const g of grams(tokenize(text), model.ngram_max || 1)) {
      const idx = model.vocab[g];
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    let norm = 0;
    const x = new Map();
    for (const [idx, c] of counts) {
      const v = c * model.idf[idx];
      x.set(idx, v);
      norm += v * v;
    }
    if (norm > 0) {
      norm = Math.sqrt(norm);
      for (const [idx, v] of x) x.set(idx, v / norm);
    }
    return x;
  }

  // -> {label, scores:{class: decisionScore}}  (same shape serve.py returned)
  function classify(clf, text) {
    const x = vectorize(text, clf);
    const scores = {};
    let best = 0;
    for (let c = 0; c < clf.classes.length; c++) {
      let s = clf.intercept[c];
      for (const [idx, v] of x) s += clf.coef[c][idx] * v;
      scores[clf.classes[c]] = s;
      if (s > scores[clf.classes[best]]) best = c;
    }
    return { label: clf.classes[best], scores };
  }

  function cosine(a, b) {
    // both already L2-normalised; iterate the smaller map
    const [s, l] = a.size <= b.size ? [a, b] : [b, a];
    let dot = 0;
    for (const [idx, v] of s) {
      const w = l.get(idx);
      if (w !== undefined) dot += v * w;
    }
    return dot;
  }

  // Place typed text at its single most-similar pool example (serve.py _position).
  // items: the L1_PREDICTIONS items (only those with lex & sem are considered).
  // Pool vectors are computed once and cached on the place model.
  function place(placeModel, items, text) {
    if (!placeModel._pool) {
      placeModel._pool = [];
      for (const it of items)
        if (it.lex && it.sem)
          placeModel._pool.push({ it, vec: vectorize(it.text, placeModel) });
    }
    const v = vectorize(text, placeModel);
    if (v.size === 0) return null;
    let top = null, topSim = 0;
    for (const p of placeModel._pool) {
      const sim = cosine(v, p.vec);
      if (sim > topSim) { topSim = sim; top = p; }
    }
    if (!top || topSim <= 1e-9) return null;   // no shared words -> can't place honestly
    return { lex: top.it.lex, sem: top.it.sem, sim: topSim, nearest: top.it.text };
  }

  global.L1_TRYIT = { tokenize, vectorize, classify, place };
})(typeof window !== "undefined" ? window : globalThis);
