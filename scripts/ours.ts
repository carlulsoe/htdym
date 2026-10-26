import { searchTuples } from '../src/core/engine/optimizer/search';
import { roleSize, type Deployment, type MoeDispatch } from '../src/core/engine/surface/deploy';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { evaluateServingStep, type ServingOptions } from '../src/core/engine/sim/run/serving';
import { evaluateDisagg } from '../src/core/engine/sim/run/disagg';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID, type ChipSpec } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// DeepSeek V4.1 Flash served the way deploy/ serves it: vLLM 0.32 DEP8 on
// one B300 x8, DSpark k=5, chunked prefill mixed into decode steps,
// max_num_seqs 256 and max_num_batched_tokens 8192 per DP rank.
//   npm exec vite-node scripts/ours.ts

const model = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4.1 Flash'))!;
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };
// Scaleway's B300 x8 list price, about EUR 60/h, at ~1.08 USD/EUR
const SCALEWAY_USD_PER_GPU_HOUR = (60 * 1.08) / 8;
const BUDGET = 8192;
const MAX_SEQS_PER_RANK = 256;

const b300 = CHIPS_BY_ID['b300'];
const dep8 = (moeDispatch: MoeDispatch = 'coalesced-a2a'): Deployment => {
  const axes = deployedAxes(b300.interconnect, { domain: 8, nodes: 1 });
  const mesh = enumeratePlacements(axes, { PP: 1, TP: 1, DPA: 8, EP: 8, ETP: 1 })[0];
  return { chip: b300, mesh, moeDispatch };
};

const usdPerM = (tokPerSecPerChip: number, usdPerHour: number) =>
  (usdPerHour / 3600 / tokPerSecPerChip) * 1e6;

function row(label: string, P: number, G: number, batch: number, s: ServingOptions, skew = 1) {
  const r = evaluateServingStep(
    { model, deployment: dep8(), workload: { prefillLen: P, generateLen: G, routingSkew: skew } },
    batch,
    s,
    opts,
  );
  if (!r.ok)
    return console.log(
      `${label.padEnd(44)} infeasible: ${r.diags.map((d) => d.message).join('; ')}`,
    );
  const busy = (r.cost as { busy?: Record<string, number> }).busy!;
  console.log(
    `${label.padEnd(44)} step ${(r.stepTime * 1e3).toFixed(2).padStart(6)} ms ` +
      `(draft ${((r.draftTime / r.stepTime) * 100).toFixed(0).padStart(2)}%, c/m/n ${(busy.compute * 1e3).toFixed(1)}/${(busy.memory * 1e3).toFixed(1)}/${(busy.comms * 1e3).toFixed(1)}) ` +
      `${String(Math.round(r.batchedTokensPerGroup)).padStart(5)} tok/rank ` +
      `${r.outputTokPerSecPerChip.toFixed(0).padStart(6)} out tok/s/GPU ` +
      `${(1 / r.tpot).toFixed(0).padStart(4)} tok/s/user ` +
      `$${usdPerM(r.outputTokPerSecPerChip, SCALEWAY_USD_PER_GPU_HOUR).toFixed(3)}/1M out`,
  );
}

console.log(
  `${model.name}, B300 x8 DEP8 coalesced-a2a, 2,048 sequences, $${SCALEWAY_USD_PER_GPU_HOUR.toFixed(2)}/GPU-h`,
);
console.log(
  '\n1K -> 1K, the warmctl benchmark shape (measured: ~6,900 out tok/s/GPU, ~35 tok/s/user)',
);
row('plain decode, no prefill (stock HTDYM view)', 1024, 1024, 2048, {});
row('+ DSpark k=5, acceptance 0.2 (random text)', 1024, 1024, 2048, {
  speculative: { draftTokens: 5, acceptance: 0.2 },
});
row('+ mixed prefill', 1024, 1024, 2048, {
  speculative: { draftTokens: 5, acceptance: 0.2 },
  mixedPrefill: true,
  maxBatchedTokens: BUDGET,
});
row(
  '+ routing skew 1.3',
  1024,
  1024,
  2048,
  {
    speculative: { draftTokens: 5, acceptance: 0.2 },
    mixedPrefill: true,
    maxBatchedTokens: BUDGET,
  },
  1.3,
);
row(
  '  acceptance 0 (no draft accepted)',
  1024,
  1024,
  2048,
  { speculative: { draftTokens: 5, acceptance: 0 }, mixedPrefill: true, maxBatchedTokens: BUDGET },
  1.3,
);
row(
  '  acceptance 0.6 (real chat)',
  1024,
  1024,
  2048,
  {
    speculative: { draftTokens: 5, acceptance: 0.6 },
    mixedPrefill: true,
    maxBatchedTokens: BUDGET,
  },
  1.3,
);
row(
  '  acceptance 0.8 (code, tool calls)',
  1024,
  1024,
  2048,
  {
    speculative: { draftTokens: 5, acceptance: 0.8 },
    mixedPrefill: true,
    maxBatchedTokens: BUDGET,
  },
  1.3,
);

console.log('\nOpenRouter-like 8K -> 512, acceptance 0.6, skew 1.3');
for (const hit of [0, 0.5, 0.8])
  for (const batch of [512, 1024, 2048])
    row(
      `prefix hit ${hit}, ${batch} seqs`,
      8192,
      512,
      batch,
      {
        speculative: { draftTokens: 5, acceptance: 0.6 },
        mixedPrefill: true,
        prefixHitRate: hit,
        maxBatchedTokens: BUDGET,
      },
      1.3,
    );

console.log('\nOpenRouter-like 32K -> 1K, acceptance 0.6, skew 1.3');
for (const hit of [0, 0.8])
  for (const batch of [256, 512, 1024])
    row(
      `prefix hit ${hit}, ${batch} seqs`,
      32768,
      1024,
      batch,
      {
        speculative: { draftTokens: 5, acceptance: 0.6 },
        mixedPrefill: true,
        prefixHitRate: hit,
        maxBatchedTokens: BUDGET,
      },
      1.3,
    );

// Aggregated vs disaggregated (phase 2), 8K -> 512, hit 0.5: one prefill
// node and one decode node at the decode batch the aggregated node runs.
console.log('\nPhase 2: P/D disaggregation, 8K -> 512, prefix hit 0.5, acceptance 0.6, skew 1.3');
for (const decodeBatch of [1024, 2048]) {
  const d = evaluateDisagg(
    {
      model,
      prefill: dep8(),
      decode: dep8(),
      workload: { prefillLen: 8192, generateLen: 512, routingSkew: 1.3 },
      decodeBatch,
      prefixHitRate: 0.5,
      speculative: { draftTokens: 5, acceptance: 0.6 },
    },
    opts,
  );
  if (!d.ok) {
    console.log(
      `decode batch ${decodeBatch}: infeasible ${d.diags.map((x) => x.message).join('; ')}`,
    );
    continue;
  }
  const outPerChip = d.reqPerSecPerChip * 512;
  console.log(
    `decode batch ${decodeBatch}: ${d.decodeChipsPerPrefillChip.toFixed(2)} decode GPUs per prefill GPU, ` +
      `KV ${(d.kvBytesPerRequest / 1e6).toFixed(1)} MB/request, transfer ${(d.transferTime * 1e3).toFixed(2)} ms, ` +
      `link util P ${(d.prefill.linkUtilisation * 100).toFixed(1)}% / D ${(d.decode.linkUtilisation * 100).toFixed(1)}%, ` +
      `fleet ${outPerChip.toFixed(0)} out tok/s/GPU, $${usdPerM(outPerChip, SCALEWAY_USD_PER_GPU_HOUR).toFixed(3)}/1M out`,
  );
}

// Which 8-GPU node, under the same serving model: per chip, the layout
// and batch with the most output per dollar holding >= 35 tok/s/user.
console.log(
  '\nOther 8-GPU nodes, 8K -> 512, hit 0.5, acceptance 0.6, >= 35 tok/s/user, HTDYM list prices',
);
const SLO = 35;
const serving: ServingOptions = {
  speculative: { draftTokens: 5, acceptance: 0.6 },
  mixedPrefill: true,
  prefixHitRate: 0.5,
  maxBatchedTokens: BUDGET,
};
for (const skew of [1, 1.3])
  for (const [chipId, nodes] of [
    ['b300', 1],
    ['b200', 1],
    ['h200-sxm', 1],
    ['h200-sxm', 2],
    ['mi355x', 1],
  ] as const) {
    const chip: ChipSpec = CHIPS_BY_ID[chipId];
    const axes = deployedAxes(chip.interconnect, { domain: 8, nodes });
    let best: { tps: number; line: string } | undefined;
    for (const sizes of searchTuples(model, 8 * nodes)) {
      if (sizes.PP > 1) continue; // speculative serving is modeled without PP
      for (const mesh of enumeratePlacements(axes, sizes))
        for (const moeDispatch of (sizes.EP > 1
          ? ['coalesced-a2a', 'ring-of-experts']
          : ['ring-of-experts']) as MoeDispatch[]) {
          const input = {
            model,
            deployment: { chip, mesh, moeDispatch },
            workload: { prefillLen: 8192, generateLen: 512, routingSkew: skew },
          };
          const dpa = roleSize(mesh, 'DPA');
          const at = (perGroup: number) =>
            evaluateServingStep(input, perGroup * dpa, serving, opts);
          const meets = (perGroup: number) => {
            const r = at(perGroup);
            return r.ok && 1 / r.tpot >= SLO;
          };
          if (!meets(1)) continue;
          let [lo, hi] = [1, MAX_SEQS_PER_RANK * ((8 * nodes) / dpa)];
          while (lo < hi) {
            const mid = Math.ceil((lo + hi) / 2);
            if (meets(mid)) lo = mid;
            else hi = mid - 1;
          }
          const r = at(lo);
          if (!r.ok || (best && r.outputTokPerSecPerChip <= best.tps)) continue;
          const layout = ROLES.filter((x) => roleSize(mesh, x) > 1)
            .map((x) => `${x}${roleSize(mesh, x)}`)
            .join(' ');
          best = {
            tps: r.outputTokPerSecPerChip,
            line:
              `skew ${skew} ${`${chipId} x${8 * nodes}`.padEnd(14)} ${layout.padEnd(18)} ${moeDispatch.padEnd(15)} ` +
              `${String(lo * dpa).padStart(5)} seqs ${r.outputTokPerSecPerChip.toFixed(0).padStart(6)} out tok/s/GPU ` +
              `${(1 / r.tpot).toFixed(0).padStart(4)} tok/s/user  $${usdPerM(r.outputTokPerSecPerChip, chip.costPerHour!).toFixed(3)}/1M out at $${chip.costPerHour}/GPU-h`,
          };
        }
    }
    console.log(best?.line ?? `${chipId} x${8 * nodes}: infeasible`);
  }
