import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { evaluateServingStep } from '../src/core/engine/sim/run/serving';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// Workload shapes on our node (V4.1, B300 x8 DEP8, DSpark acceptance 0.6,
// skew 1.3, 8192-token budget per rank): the most output per GPU that
// still holds each per-user speed.
//   npm exec vite-node scripts/shapes.ts
const model = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4.1 Flash'))!;
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };
const chip = CHIPS_BY_ID['b300'];
const mesh = enumeratePlacements(deployedAxes(chip.interconnect, { domain: 8, nodes: 1 }), {
  PP: 1,
  TP: 1,
  DPA: 8,
  EP: 8,
  ETP: 1,
})[0];
const usdPerGpuHour = (60 * 1.08) / 8;

const shapes: [string, number, number, number][] = [
  ['InferenceX 1k1k, no cache', 1024, 1024, 0],
  ['InferenceX 8k1k, no cache', 8192, 1024, 0],
  ['InferenceX 1k8k, no cache', 1024, 8192, 0],
  ['chat 4k->600, 50% hit', 4096, 600, 0.5],
  ['AgentX p50 88k->413, 91% hit', 88_000, 413, 0.91],
  ['AgentX p50 88k->413, 73% hit', 88_000, 413, 0.73],
  ['AgentX p90 (256k cap) 200k->2.2k, 91% hit', 200_000, 2200, 0.91],
];
console.log(
  'shape                                       >=tok/s/user  seqs  out tok/s/GPU  in tok/s/GPU  new-prefill tok/rank/step  $/1M out (all cost)',
);
for (const [label, P, G, hit] of shapes)
  for (const slo of [35, 90]) {
    const at = (perRank: number) =>
      evaluateServingStep(
        {
          model,
          deployment: { chip, mesh, moeDispatch: 'coalesced-a2a' },
          workload: { prefillLen: P, generateLen: G, routingSkew: 1.3 },
        },
        perRank * 8,
        {
          speculative: { draftTokens: 5, acceptance: 0.6 },
          mixedPrefill: true,
          prefixHitRate: hit,
          maxBatchedTokens: 8192,
        },
        opts,
      );
    let best: ReturnType<typeof at> | undefined;
    let seqs = 0;
    for (let k = 1; k <= 256; k++) {
      const r = at(k);
      if (!r.ok || 1 / r.tpot < slo) continue;
      if (!best || !best.ok || r.outputTokPerSecPerChip > best.outputTokPerSecPerChip)
        [best, seqs] = [r, k * 8];
    }
    if (!best || !best.ok) {
      console.log(`${label.padEnd(44)} ${String(slo).padStart(4)}          infeasible`);
      continue;
    }
    console.log(
      `${label.padEnd(44)} ${String(slo).padStart(4)}  ${String(seqs).padStart(10)} ${best.outputTokPerSecPerChip.toFixed(0).padStart(13)} ` +
        `${best.inputTokPerSecPerChip.toFixed(0).padStart(13)} ${best.prefillTokensPerGroup.toFixed(0).padStart(26)} ` +
        `${((usdPerGpuHour / 3600 / best.outputTokPerSecPerChip) * 1e6).toFixed(2).padStart(10)}`,
    );
  }
