import { expect, test } from 'vitest';
import { searchShardings, searchTuples } from '../src/core/engine/optimizer/search';
import { evaluateServingStep, type ServingOptions } from '../src/core/engine/sim/run/serving';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { roleSize, type Deployment, type MoeDispatch } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// Pins sweep results to what the planner and pricer produced before their
// caches were restructured, so a speedup that changes a winner, a batch
// or a score fails here. Each row is a tuple's best (or, for the serving
// sweep, every placement's bisected operating point) as
// [deployment, batch, score].

type Row = [string, number, number];

const describe = (d: Deployment) =>
  [
    ...ROLES.filter((r) => roleSize(d.mesh, r) > 1).map(
      (r) => `${r}${roleSize(d.mesh, r)}[${d.mesh.roles[r].join('')}]`,
    ),
    d.moeDispatch,
    `dcp${d.decodeContextParallel ?? 1}`,
  ].join(' ');

const model = (name: string) => MODEL_PRESETS.find((m) => m.name.startsWith(name))!;

function search(
  modelName: string,
  chipId: string,
  machine: string | { domain: number; nodes: number },
  phase: Parameters<typeof searchShardings>[4]['phase'],
): Row[] {
  const chip = CHIPS_BY_ID[chipId];
  const m =
    typeof machine === 'string'
      ? chip.interconnect.topologies!.find((t) => t.name === machine)!
      : machine;
  return [
    ...searchShardings(
      model(modelName),
      chip,
      m,
      { prefillLen: 4096, generateLen: 1024 },
      { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 }), phase },
    ),
  ].flatMap(({ candidate: c }): Row[] =>
    c ? [[describe(c.deployment), c.batch ?? 0, c.score]] : [],
  );
}

// scripts/rank.ts's serving sweep, on the first `tuples` speculative-
// servable tuples of one machine
function servingSweep(chipId: string, domain: number, nodes: number, tuples: number): Row[] {
  const v41 = model('DeepSeek V4.1 Flash');
  const chip = CHIPS_BY_ID[chipId];
  const opts = {
    costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }),
  };
  const serving: ServingOptions = {
    speculative: { draftTokens: 5, acceptance: 0.6 },
    mixedPrefill: true,
    prefixHitRate: 0.91,
    maxBatchedTokens: 8192,
  };
  const workload = { prefillLen: 88000, generateLen: 413, routingSkew: 1.3 };
  const axes = deployedAxes(chip.interconnect, { domain, nodes });
  const rows: Row[] = [];
  for (const sizes of searchTuples(v41, domain * nodes)
    .filter((s) => s.PP === 1)
    .slice(0, tuples))
    for (const mesh of enumeratePlacements(axes, sizes))
      for (const moeDispatch of (sizes.EP > 1
        ? ['coalesced-a2a', 'ring-of-experts']
        : ['ring-of-experts']) as MoeDispatch[]) {
        const deployment = { chip, mesh, moeDispatch };
        const dpa = roleSize(mesh, 'DPA');
        const at = (g: number) =>
          evaluateServingStep({ model: v41, deployment, workload }, g * dpa, serving, opts);
        const meets = (g: number) => {
          const r = at(g);
          return r.ok && 1 / r.tpot >= 50;
        };
        if (!meets(1)) continue;
        let [lo, hi] = [1, (256 * domain * nodes) / dpa];
        while (lo < hi) {
          const mid = Math.ceil((lo + hi) / 2);
          if (meets(mid)) lo = mid;
          else hi = mid - 1;
        }
        const r = at(lo);
        if (r.ok) rows.push([describe(deployment), lo * dpa, r.outputTokPerSecPerChip]);
      }
  return rows;
}

function expectRows(got: Row[], want: Row[]) {
  expect(got.map(([d, b]) => [d, b])).toEqual(want.map(([d, b]) => [d, b]));
  got.forEach(([, , score], i) =>
    expect(Math.abs(score - want[i][2])).toBeLessThanOrEqual(1e-9 * Math.abs(want[i][2])),
  );
}

// prettier-ignore
const V41_B300_8_DECODE_SLO: Row[] = [
  ["DPA8[D] EP8[D] coalesced-a2a dcp1", 5120, 32795.22396953869],
  ["DPA4[D0] TP2[D1] EP8[D0D1] coalesced-a2a dcp2", 4124, 25833.978363746784],
  ["DPA2[D1] TP4[D0D2] EP8[D0D1D2] ring-of-experts dcp2", 2846, 17791.02236681406],
  ["TP8[D] EP8[D] ring-of-experts dcp8", 1338, 8363.308479724925],
  ["DPA8[D0D1] EP4[D0] ETP2[D1] coalesced-a2a dcp1", 4784, 29904.253827179742],
  ["PP2[D0] DPA4[D1] EP4[D1] coalesced-a2a dcp1", 424, 5301.63432772971],
  ["DPA4[D0] TP2[D1] EP4[D0] ETP2[D1] coalesced-a2a dcp2", 4316, 27026.508244221306],
  ["PP2[D0] DPA2[D1] TP2[D2] EP4[D1D2] coalesced-a2a dcp1", 326, 4075.4501355702173],
  ["DPA2[D0] TP4[D1D2] EP4[D0D1] ETP2[D2] ring-of-experts dcp2", 2816, 17704.67825653679],
  ["PP2[D1] TP4[D0D2] EP4[D0D2] coalesced-a2a dcp2", 256, 3221.5596317377167],
  ["TP8[D0D1D2] EP4[D0D2] ETP2[D1] ring-of-experts dcp4", 1965, 12281.781009551882],
  ["DPA8[D0D1D2] EP2[D1] ETP4[D0D2] coalesced-a2a dcp1", 4344, 27163.19576812013],
  ["PP2[D0] DPA4[D1D2] EP2[D1] ETP2[D2] coalesced-a2a dcp1", 332, 4150.82505391529],
  ["DPA4[D0D2] TP2[D1] EP2[D1] ETP4[D0D2] coalesced-a2a dcp2", 3844, 24026.236230466617],
  ["PP2[D0] DPA2[D1] TP2[D2] EP2[D1] ETP2[D2] coalesced-a2a dcp1", 372, 4651.180533995218],
  ["DPA2[D1] TP4[D0D2] EP2[D1] ETP4[D0D2] coalesced-a2a dcp2", 3166, 19791.175090896577],
  ["PP2[D0] TP4[D1D2] EP2[D1] ETP2[D2] coalesced-a2a dcp2", 256, 3223.3108616128575],
  ["TP8[D0D1D2] EP2[D1] ETP4[D0D2] ring-of-experts dcp4", 1922, 12012.525779314195],
  ["DPA8[D] ETP8[D] ring-of-experts dcp1", 3776, 23608.960046934975],
  ["PP2[D1] DPA4[D0D2] ETP4[D0D2] ring-of-experts dcp1", 316, 3952.3883444126577],
  ["DPA4[D0D2] TP2[D1] ETP8[D0D1D2] ring-of-experts dcp2", 3440, 21500.9921102562],
  ["PP2[D0] DPA2[D1] TP2[D2] ETP4[D1D2] ring-of-experts dcp1", 306, 3825.5452133992644],
  ["DPA2[D1] TP4[D0D2] ETP8[D0D1D2] ring-of-experts dcp2", 2542, 15888.424569656174],
  ["PP2[D1] TP4[D0D2] ETP4[D0D2] ring-of-experts dcp2", 312, 3900.4429625033845],
  ["TP8[D] ETP8[D] ring-of-experts dcp8", 1664, 10441.622410473588],
];
// prettier-ignore
const GLM_V7X_222_PREFILL: Row[] = [
  ["DPA8[XYZ] EP8[XYZ] ring-of-experts dcp1", 0, 15122.397356935373],
  ["DPA4[XY] TP2[Z] EP8[XYZ] ring-of-experts dcp1", 0, 8147.757310083606],
  ["DPA2[X] TP4[YZ] EP8[XYZ] ring-of-experts dcp1", 0, 6588.454030452964],
  ["TP8[XYZ] EP8[XYZ] ring-of-experts dcp1", 0, 5290.990057962591],
  ["DPA8[XYZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 14870.861536658973],
  ["PP2[X] DPA4[YZ] EP4[YZ] ring-of-experts dcp1", 0, 16561.014217547014],
  ["DPA4[XY] TP2[Z] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 8304.266038897522],
  ["PP2[X] DPA2[Y] TP2[Z] EP4[YZ] ring-of-experts dcp1", 0, 8123.471489466869],
  ["DPA2[X] TP4[YZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 6540.256876825166],
  ["PP2[X] TP4[YZ] EP4[YZ] ring-of-experts dcp1", 0, 6148.986310167286],
  ["TP8[XYZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 5259.861833717205],
  ["DPA8[XYZ] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 14376.691477788921],
  ["PP2[X] DPA4[YZ] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 16267.287929842352],
  ["PP4[XY] DPA2[Z] EP2[Z] coalesced-a2a dcp1", 0, 17549.580432991534],
  ["DPA4[XY] TP2[Z] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 7926.2471438465855],
  ["PP2[X] DPA2[Y] TP2[Z] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 8383.97478031594],
  ["PP4[XY] TP2[Z] EP2[Z] coalesced-a2a dcp1", 0, 8504.179231098808],
  ["DPA2[X] TP4[YZ] EP2[Y] ETP4[XZ] ring-of-experts dcp1", 0, 6442.857831498173],
  ["PP2[X] TP4[YZ] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 6108.037062939697],
  ["TP8[XYZ] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 5196.681519184171],
  ["DPA8[XYZ] ETP8[XYZ] ring-of-experts dcp1", 0, 13472.385687016485],
  ["PP2[X] DPA4[YZ] ETP4[YZ] ring-of-experts dcp1", 0, 15698.43411897841],
  ["PP4[XY] DPA2[Z] ETP2[Z] ring-of-experts dcp1", 0, 17165.953421067654],
  ["PP8[XYZ] ring-of-experts dcp1", 0, 24677.891989358395],
  ["DPA4[XY] TP2[Z] ETP8[XYZ] ring-of-experts dcp1", 0, 7643.391049244933],
  ["PP2[X] DPA2[Y] TP2[Z] ETP4[YZ] ring-of-experts dcp1", 0, 7910.270524515716],
  ["PP4[XY] TP2[Z] ETP2[Z] ring-of-experts dcp1", 0, 10306.152471941037],
  ["DPA2[X] TP4[YZ] ETP8[XYZ] ring-of-experts dcp1", 0, 6254.710689268574],
  ["PP2[X] TP4[YZ] ETP4[YZ] ring-of-experts dcp1", 0, 7328.419140798539],
  ["TP8[XYZ] ETP8[XYZ] ring-of-experts dcp1", 0, 6697.35395832914],
];
// prettier-ignore
const V41_B300_8X2_SERVING: Row[] = [
  ["DPA16[DN] EP16[DN] coalesced-a2a dcp1", 944, 2959.3448109065407],
  ["DPA16[DN] EP16[DN] ring-of-experts dcp1", 1232, 3853.572112039495],
  ["DPA8[D0N] TP2[D1] EP16[D0D1N] coalesced-a2a dcp1", 784, 2463.4448810684603],
  ["DPA8[D0N] TP2[D1] EP16[D0D1N] ring-of-experts dcp1", 1184, 3703.3185704600423],
  ["DPA8[D0D2N] TP2[D1] EP16[D0D1D2N] coalesced-a2a dcp1", 784, 2463.4448810684603],
  ["DPA8[D0D2N] TP2[D1] EP16[D0D1D2N] ring-of-experts dcp1", 1184, 3708.0056960538263],
  ["DPA8[D] TP2[N] EP16[DN] coalesced-a2a dcp1", 440, 1393.5055496577766],
  ["DPA8[D] TP2[N] EP16[DN] ring-of-experts dcp1", 584, 1841.952023324227],
  ["DPA4[D0N] TP4[D1] EP16[D0D1N] coalesced-a2a dcp1", 584, 1829.4619504199063],
  ["DPA4[D0N] TP4[D1] EP16[D0D1N] ring-of-experts dcp1", 632, 2855.074088410604],
  ["DPA4[D1N] TP4[D0D2] EP16[D0D1D2N] coalesced-a2a dcp1", 588, 1846.8767832042988],
  ["DPA4[D1N] TP4[D0D2] EP16[D0D1D2N] ring-of-experts dcp1", 632, 2908.5276267047275],
  ["DPA4[D0] TP4[D1N] EP16[D0D1N] coalesced-a2a dcp1", 368, 1156.7133829581142],
  ["DPA4[D0] TP4[D1N] EP16[D0D1N] ring-of-experts dcp1", 520, 1635.3707960173854],
  ["DPA4[D0D2] TP4[D1N] EP16[D0D1D2N] coalesced-a2a dcp1", 368, 1156.7133829581142],
  ["DPA4[D0D2] TP4[D1N] EP16[D0D1D2N] ring-of-experts dcp1", 524, 1641.9889340742616],
];

test('decode search under an SLO pins V4.1 on a B300 x8', () => {
  expectRows(
    search(
      'DeepSeek V4.1 Flash',
      'b300',
      { domain: 8, nodes: 1 },
      { kind: 'decode', policy: { sloTokPerSecPerUser: 50 } },
    ),
    V41_B300_8_DECODE_SLO,
  );
});

test('prefill search pins GLM on a TPU v7x 2x2x2 torus', () => {
  expectRows(
    search('GLM 5.3', 'tpu-v7x', '2x2x2', { kind: 'prefill', seqs: 8, mode: 'throughput' }),
    GLM_V7X_222_PREFILL,
  );
});

test('serving sweep pins V4.1 on two B300 x8 nodes', () => {
  expectRows(servingSweep('b300', 8, 2, 3), V41_B300_8X2_SERVING);
});
