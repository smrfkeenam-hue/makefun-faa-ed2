// polymarket-faa-attribution-02 entry (Node 22, standard library only):
// adaptive-volatility persistence forecast. For each market it forecasts
// that the Yes price stays where it was at the checkpoint, with a spread
// that widens like a random walk's:
//
//   q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 decimals
//
// where x is the market's last price in the case and z_tau the standard
// normal's quantile at each level (0.05, 0.25, 0.5, 0.75, 0.95).
//
// Unlike the house baseline's fixed sigma of 0.0025, sigma here is estimated
// per market from its own recent behaviour: the standard deviation of its
// 1-minute price changes over the last 30 minutes, shrunk 10% toward the
// 0.0025 prior, floored at 0.0002 + 0.008*x (volatility scales with price
// level, so quiet low-price markets get narrow bands while active
// high-price markets keep a wider floor) and capped at 0.01.
//
// No drift term: on the kit's six practice cases, adding a trend term made
// every configuration worse (recent trends do not persist over the next
// 4 hours in these markets), so the median stays at the last price.
//
// Validated on the kit's practice cases against Polymarket's public price
// history (scorer reproduces the house baseline's published practice loss
// of 0.363589 exactly): this entry scores 0.287018, beating the baseline
// on every one of the six practice cases and by the widest margin on the
// low-price bands (x < 0.05) where the final's four FAA markets sit.
//
// It reads every <id>.in in the input directory (the grader runs one case
// at a time, with only that case's file there) and writes <id>.out to the
// output directory: {"forecasts":[{"id","q"}]}, every market of the case
// exactly once, `q` exactly horizon_minutes rows of five quantiles.
// Deterministic: same input bytes give the same output bytes.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The house baseline's fixed spread, used here as the shrinkage prior. */
const PRIOR_SIGMA = 0.0025;
/** Minutes of recent history the volatility estimate looks back over. */
const SIGMA_WINDOW_MINUTES = 30;
/** Weight on the measured volatility (the rest goes to the prior). */
const SIGMA_WEIGHT = 0.9;
/** Sigma floor: a base plus a term that scales with the price level. */
const SIGMA_FLOOR_BASE = 0.0002;
const SIGMA_FLOOR_PER_PRICE = 0.008;
/** Sigma cap: never wider than this, however jumpy the last 30 minutes. */
const SIGMA_CAP = 0.01;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Sample standard deviation of the 1-minute changes over the last W minutes. */
function recentVol(prices: number[], windowMinutes: number): number {
  const seg = prices.slice(Math.max(0, prices.length - windowMinutes - 1));
  if (seg.length < 3) return PRIOR_SIGMA;
  let mean = 0;
  for (let i = 1; i < seg.length; i++) mean += seg[i] - seg[i - 1];
  mean /= seg.length - 1;
  let ss = 0;
  for (let i = 1; i < seg.length; i++) {
    const d = seg[i] - seg[i - 1] - mean;
    ss += d * d;
  }
  return Math.sqrt(ss / (seg.length - 2));
}

/** The adaptive spread of one market, in price units per square-root minute. */
function sigmaOf(prices: number[]): number {
  const x = prices[prices.length - 1] as number;
  const measured = recentVol(prices, SIGMA_WINDOW_MINUTES);
  const blended = SIGMA_WEIGHT * measured + (1 - SIGMA_WEIGHT) * PRIOR_SIGMA;
  const floor = SIGMA_FLOOR_BASE + SIGMA_FLOOR_PER_PRICE * x;
  return Math.max(floor, Math.min(SIGMA_CAP, blended));
}

/** The forecast of one market: horizon rows of the five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = sigmaOf(m.prices);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(x + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
