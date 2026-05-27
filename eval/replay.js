#!/usr/bin/env node
// ◊·κ FallCore · eval harness
//
// Replays past frontier calls (from proxy log JSONL) through the local
// model and scores equivalence + projected savings.
//
// Usage:
//   node eval/replay.js                       # last 7 days
//   node eval/replay.js --days 30             # last 30 days
//   node eval/replay.js --file logs/2026-05-27.jsonl
//   node eval/replay.js --model qwen2.5:72b   # try a heavier model
//   node eval/replay.js --out report.json     # JSON report for the dashboard

const fs = require('fs');
const path = require('path');

const args = (() => {
  const a = { days: 7 };
  for (let i = 2; i < process.argv.length; i++) {
    const k = process.argv[i];
    if (k === '--days')   a.days = parseInt(process.argv[++i], 10);
    if (k === '--file')   a.file = process.argv[++i];
    if (k === '--model')  a.model = process.argv[++i];
    if (k === '--out')    a.out  = process.argv[++i];
    if (k === '--proxy')  a.proxy = process.argv[++i];
  }
  return a;
})();

const PROXY_URL = args.proxy || 'http://localhost:11434';
const LOG_DIR = path.join(__dirname, '..', 'logs');

function collectLogs() {
  const records = [];
  if (args.file) {
    const lines = fs.readFileSync(args.file, 'utf8').trim().split('\n');
    for (const l of lines) try { records.push(JSON.parse(l)); } catch (_) {}
    return records;
  }
  if (!fs.existsSync(LOG_DIR)) return [];
  const now = Date.now();
  const cutoff = now - args.days * 24 * 60 * 60 * 1000;
  for (const fn of fs.readdirSync(LOG_DIR)) {
    if (!fn.endsWith('.jsonl')) continue;
    const lines = fs.readFileSync(path.join(LOG_DIR, fn), 'utf8').trim().split('\n');
    for (const l of lines) {
      try {
        const r = JSON.parse(l);
        if (r.t && r.t > cutoff) records.push(r);
      } catch (_) {}
    }
  }
  return records;
}

// Simple ROUGE-ish overlap score · tokens shared / max(tokens)
function similarity(a, b) {
  if (!a || !b) return 0;
  const ta = new Set(String(a).toLowerCase().match(/\b\w+\b/g) || []);
  const tb = new Set(String(b).toLowerCase().match(/\b\w+\b/g) || []);
  if (ta.size === 0 || tb.size === 0) return 0;
  let shared = 0;
  for (const t of ta) if (tb.has(t)) shared++;
  return shared / Math.max(ta.size, tb.size);
}

// Length ratio bonus (penalises wildly different responses)
function lengthScore(a, b) {
  const la = String(a || '').length, lb = String(b || '').length;
  if (la === 0 || lb === 0) return 0;
  const ratio = Math.min(la, lb) / Math.max(la, lb);
  return ratio;
}

function equivalence(local, frontier) {
  const sim = similarity(local, frontier);
  const len = lengthScore(local, frontier);
  return (sim * 0.7 + len * 0.3);
}

(async () => {
  const all = collectLogs();
  const frontier = all.filter(r => r.tier === 'frontier' && r.local_response && r.frontier_response);
  const localOnly = all.filter(r => r.tier === 'local');

  console.log('━'.repeat(70));
  console.log('◊·κ FallCore eval · replay analysis');
  console.log('━'.repeat(70));
  console.log('window:           last ' + args.days + ' days');
  console.log('records loaded:   ' + all.length);
  console.log('  - local calls:    ' + localOnly.length);
  console.log('  - frontier calls: ' + frontier.length);
  console.log('  - with both:      ' + frontier.length + ' (cascade fired)');
  console.log('');

  if (frontier.length === 0) {
    console.log('No cascade-fired records yet (low-confidence local + frontier responses both present).');
    console.log('Run some queries through the proxy first, then re-run this eval.');
    process.exit(0);
  }

  let totalEq = 0, totalSpend = 0, replayableSaved = 0;
  const sample = frontier.slice(0, Math.min(50, frontier.length));
  const breakdown = [];

  for (const r of sample) {
    const eq = equivalence(r.local_response, r.frontier_response);
    totalEq += eq;
    totalSpend += r.cost_usd || 0;
    if (eq >= 0.6) replayableSaved += r.cost_usd || 0;
    breakdown.push({
      id: r.id, t: r.t, tool: r.tool,
      cost_usd: r.cost_usd,
      local_confidence: r.local_confidence,
      equivalence: eq,
      would_save: eq >= 0.6
    });
  }

  const avgEq = totalEq / sample.length;
  const couldHaveStayedLocal = breakdown.filter(b => b.would_save).length;

  console.log('SCORED ' + sample.length + ' frontier responses against their local counterparts:');
  console.log('  avg equivalence:        ' + (avgEq * 100).toFixed(1) + '%');
  console.log('  would-have-stayed-local: ' + couldHaveStayedLocal + ' of ' + sample.length + ' (' + ((couldHaveStayedLocal/sample.length)*100).toFixed(1) + '%)');
  console.log('');
  console.log('COST:');
  console.log('  frontier spend (sampled): $' + totalSpend.toFixed(4));
  console.log('  replayable savings:       $' + replayableSaved.toFixed(4));
  console.log('  projected if confidence raised: ~' + ((replayableSaved / Math.max(0.001, totalSpend)) * 100).toFixed(1) + '% of frontier spend was unnecessary');
  console.log('');
  console.log('PER-TOOL:');
  const byTool = {};
  breakdown.forEach(b => {
    const t = b.tool || 'unknown';
    byTool[t] = byTool[t] || { n: 0, save: 0, spend: 0 };
    byTool[t].n++;
    byTool[t].spend += b.cost_usd || 0;
    if (b.would_save) byTool[t].save += b.cost_usd || 0;
  });
  for (const [t, s] of Object.entries(byTool)) {
    console.log('  ' + t.padEnd(20) + ' n=' + String(s.n).padStart(3) + ' spend=$' + s.spend.toFixed(4).padStart(8) + ' replayable=$' + s.save.toFixed(4));
  }
  console.log('');
  console.log('RECOMMENDATION:');
  if (avgEq > 0.75) {
    console.log('  ✓ Local model is highly equivalent. Lower CONFIDENCE_THRESHOLD to 0.5 — more queries stay local.');
  } else if (avgEq > 0.55) {
    console.log('  ◐ Local model is acceptable for ~half. Consider a bigger model OR run a LoRA fine-tune on your reviewer corrections.');
  } else {
    console.log('  ✗ Local model is too far from frontier. Try a larger model (qwen2.5:72b) or wait for first LoRA cycle.');
  }
  console.log('━'.repeat(70));

  const report = {
    window_days: args.days,
    sampled: sample.length,
    avg_equivalence: avgEq,
    would_have_stayed_local_count: couldHaveStayedLocal,
    frontier_spend_usd: totalSpend,
    replayable_savings_usd: replayableSaved,
    saving_percentage: (replayableSaved / Math.max(0.001, totalSpend)) * 100,
    by_tool: byTool,
    recommendation: avgEq > 0.75 ? 'lower_threshold' : avgEq > 0.55 ? 'try_lora' : 'upgrade_model',
    generated_at: new Date().toISOString()
  };
  if (args.out) {
    fs.writeFileSync(args.out, JSON.stringify(report, null, 2));
    console.log('Report written to ' + args.out);
  }
})();
