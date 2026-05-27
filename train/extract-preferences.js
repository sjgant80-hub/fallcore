#!/usr/bin/env node
// ◊·κ FallCore · LoRA preference-pair extractor
//
// Pulls (prompt, AI suggestion, human correction) triples out of:
//   1. The proxy's own JSONL logs (when frontier was called after local got it wrong)
//   2. Fall* KCC ledger exports (every "approve/flag" on AI output)
//   3. Manual annotation files (.annotations/*.jsonl)
//
// Outputs:
//   train.jsonl     — chosen/rejected pairs in standard DPO format
//   eval.jsonl      — 10% held out for eval
//   report.md       — what was extracted, from where, quality breakdown
//
// Usage:
//   node train/extract-preferences.js
//   node train/extract-preferences.js --kcc ~/Downloads/fallforce-kcc-proof-*.json
//   node train/extract-preferences.js --out training-2026-Q2/
//
// The output JSONL is ready to feed to Hugging Face TRL, axolotl, unsloth, etc.

const fs = require('fs');
const path = require('path');

const args = (() => {
  const a = { out: 'training-' + new Date().toISOString().slice(0,10) };
  for (let i = 2; i < process.argv.length; i++) {
    const k = process.argv[i];
    if (k === '--kcc')        a.kcc = process.argv[++i];
    if (k === '--proxy-logs') a.proxyLogs = process.argv[++i];
    if (k === '--out')        a.out = process.argv[++i];
    if (k === '--min-quality') a.minQ = parseFloat(process.argv[++i]);
  }
  return a;
})();

const PROXY_LOG_DIR = args.proxyLogs || path.join(__dirname, '..', 'logs');
const MIN_QUALITY = args.minQ ?? 0.3;
const OUT_DIR = args.out;
fs.mkdirSync(OUT_DIR, { recursive: true });

const pairs = [];   // { prompt, chosen, rejected, source, tool, quality, t }

// ─── Source 1: proxy logs · frontier-called-after-local triples
function fromProxyLogs() {
  if (!fs.existsSync(PROXY_LOG_DIR)) return 0;
  let n = 0;
  for (const fn of fs.readdirSync(PROXY_LOG_DIR)) {
    if (!fn.endsWith('.jsonl')) continue;
    const lines = fs.readFileSync(path.join(PROXY_LOG_DIR, fn), 'utf8').trim().split('\n');
    for (const line of lines) {
      try {
        const r = JSON.parse(line);
        if (r.tier !== 'frontier' || !r.local_response || !r.frontier_response) continue;
        if (r.local_response === r.frontier_response) continue;
        // The local got it wrong (low confidence → frontier called). Frontier IS the chosen.
        pairs.push({
          prompt: '(prompt redacted — see internal context)',  // proxy doesn't store full prompt in log to avoid bloat
          chosen: r.frontier_response,
          rejected: r.local_response,
          source: 'proxy-cascade',
          tool: r.tool || 'unknown',
          quality: 1.0 - (r.local_confidence || 0),   // higher when local was VERY wrong
          t: r.t
        });
        n++;
      } catch (_) {}
    }
  }
  return n;
}

// ─── Source 2: KCC export (Fall* tools) · reviewer corrections
function fromKccExport(filePath) {
  if (!fs.existsSync(filePath)) return 0;
  let n = 0;
  try {
    const blob = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const events = blob.events || [];
    // We want pairs where action like "llm:*" is followed shortly by "audit:flag" or similar correction
    // Pair the AI call with the next human correction within 30 minutes
    const sorted = events.slice().sort((a, b) => a.t - b.t);
    for (let i = 0; i < sorted.length; i++) {
      const ev = sorted[i];
      if (!/^(llm|ob:reason|ai)/.test(ev.action || '')) continue;
      // Find the next correction within 30 min
      for (let j = i + 1; j < sorted.length && sorted[j].t - ev.t < 30 * 60 * 1000; j++) {
        const c = sorted[j];
        if (/^(audit:flag|audit:escalation|reviewer:correction|gov:override)/.test(c.action || '')) {
          // Pair: AI output (we don't have full text in KCC, just action) vs human direction
          pairs.push({
            prompt: '(' + (ev.meta?.problem || ev.action) + ')',
            chosen: '(human correction: ' + c.action + (c.meta ? ' · ' + JSON.stringify(c.meta).slice(0, 120) : '') + ')',
            rejected: '(AI suggestion: ' + ev.action + ')',
            source: 'kcc-correction',
            tool: blob.tool || 'unknown',
            quality: 0.5,  // signal-only, not full-text pairs
            t: ev.t
          });
          n++;
          break;
        }
      }
    }
  } catch (e) {
    console.error('failed to parse', filePath, ':', e.message);
  }
  return n;
}

// ─── Source 3: manual annotation files
function fromAnnotations() {
  const dir = path.join(__dirname, '..', '.annotations');
  if (!fs.existsSync(dir)) return 0;
  let n = 0;
  for (const fn of fs.readdirSync(dir)) {
    if (!fn.endsWith('.jsonl')) continue;
    const lines = fs.readFileSync(path.join(dir, fn), 'utf8').trim().split('\n');
    for (const line of lines) {
      try {
        const r = JSON.parse(line);
        if (r.prompt && r.chosen && r.rejected) {
          pairs.push({ ...r, source: 'manual-annotation', quality: r.quality || 0.9, t: r.t || Date.now() });
          n++;
        }
      } catch (_) {}
    }
  }
  return n;
}

console.log('━'.repeat(70));
console.log('◊·κ FallCore · preference extractor');
console.log('━'.repeat(70));

const proxyN = fromProxyLogs();
console.log('proxy cascade triples:    ' + proxyN);

let kccN = 0;
if (args.kcc) {
  const files = args.kcc.includes('*') ? null : [args.kcc];
  // Naive: only handles single file or comma-separated
  if (!files) console.log('(use shell glob, pass one file at a time for now)');
  for (const f of (files || [])) {
    kccN += fromKccExport(f);
  }
  console.log('kcc correction signals:   ' + kccN);
}

const manualN = fromAnnotations();
console.log('manual annotations:       ' + manualN);
console.log('total pairs:              ' + pairs.length);

if (pairs.length === 0) {
  console.log('\nNo training data yet. Run more cascade-firing queries and/or export KCC ledgers.');
  process.exit(0);
}

// Quality filter
const filtered = pairs.filter(p => p.quality >= MIN_QUALITY);
console.log('after quality filter ≥' + MIN_QUALITY + ': ' + filtered.length);

// Shuffle for split
for (let i = filtered.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [filtered[i], filtered[j]] = [filtered[j], filtered[i]];
}

// 90/10 split
const evalN = Math.max(1, Math.floor(filtered.length * 0.1));
const evalSet = filtered.slice(0, evalN);
const trainSet = filtered.slice(evalN);

// Write JSONL
const trainPath = path.join(OUT_DIR, 'train.jsonl');
const evalPath = path.join(OUT_DIR, 'eval.jsonl');
fs.writeFileSync(trainPath, trainSet.map(p => JSON.stringify({
  prompt: p.prompt, chosen: p.chosen, rejected: p.rejected,
  meta: { source: p.source, tool: p.tool, quality: p.quality, t: p.t }
})).join('\n'));
fs.writeFileSync(evalPath, evalSet.map(p => JSON.stringify({
  prompt: p.prompt, chosen: p.chosen, rejected: p.rejected,
  meta: { source: p.source, tool: p.tool, quality: p.quality, t: p.t }
})).join('\n'));

// Report
const bySource = {};
const byTool = {};
filtered.forEach(p => {
  bySource[p.source] = (bySource[p.source] || 0) + 1;
  byTool[p.tool] = (byTool[p.tool] || 0) + 1;
});

const report = [
  '# ◊·κ FallCore preference-extraction report',
  '',
  '_generated ' + new Date().toISOString() + '_',
  '',
  '## Summary',
  '- train.jsonl: ' + trainSet.length + ' pairs',
  '- eval.jsonl:  ' + evalSet.length + ' pairs',
  '- quality threshold: ' + MIN_QUALITY,
  '',
  '## By source',
  ...Object.entries(bySource).map(([k, v]) => '- ' + k + ': ' + v),
  '',
  '## By tool',
  ...Object.entries(byTool).map(([k, v]) => '- ' + k + ': ' + v),
  '',
  '## Next steps',
  '1. Spot-check 10 random pairs in train.jsonl for sanity',
  '2. Run LoRA fine-tune (axolotl / unsloth / TRL DPO trainer)',
  '3. Deploy adapter alongside base model in Ollama',
  '4. Re-run `node eval/replay.js` to measure improvement'
].join('\n');
fs.writeFileSync(path.join(OUT_DIR, 'report.md'), report);

console.log('');
console.log('output:');
console.log('  ' + trainPath);
console.log('  ' + evalPath);
console.log('  ' + path.join(OUT_DIR, 'report.md'));
console.log('━'.repeat(70));
