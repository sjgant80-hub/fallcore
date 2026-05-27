// ═══════════════════════════════════════════════════════════════
// ◊·κ=1 · FallCore Proxy · Anthropic-API-compatible local endpoint
//
// Drop-in replacement for https://api.anthropic.com/v1/messages.
// Every Fall* tool (and any third-party app that hits Anthropic's API)
// can point at FallCore instead and Just Work — but now the cognition
// runs on the customer's own GPU, the data never leaves, and every
// query becomes training data for the next fine-tune.
//
// Routes:
//   POST  /v1/messages          · Anthropic Messages API shape
//   POST  /v1/chat/completions  · OpenAI Chat shape (bonus, same backend)
//   GET   /v1/models            · list available local + frontier models
//   GET   /health               · status + uptime + token-spend dashboard
//   GET   /v1/stats             · usage by tool, by tier (T1 local / T3 frontier), $ saved
//   GET   /v1/log               · recent prompt/response pairs (admin only)
//
// Cascade decision:
//   1. Try local Ollama (configured model)
//   2. If confidence < threshold OR explicit `force_frontier: true`
//      → fall through to real Anthropic with customer's key
//   3. Log everything to disk for the LoRA pipeline
//
// Env vars:
//   PORT                = 11434 (Ollama-friendly default)
//   OLLAMA_URL          = http://localhost:11434
//   OLLAMA_MODEL        = qwen2.5:32b (default · 70b also recommended)
//   ANTHROPIC_FALLBACK_KEY = customer's real Anthropic key (optional)
//   CONFIDENCE_THRESHOLD = 0.6 (local stays if score >= this)
//   ADMIN_KEY           = generated on first boot, written to .admin-key
//   LOG_DIR             = ./logs (prompt/response pairs · for training)
// ═══════════════════════════════════════════════════════════════

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = process.env.PORT || 11434;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:32b';
const ANTHROPIC_FALLBACK_KEY = process.env.ANTHROPIC_FALLBACK_KEY || '';
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD || '0.6');
const LOG_DIR = process.env.LOG_DIR || path.join(__dirname, '..', 'logs');
const STATE_FILE = path.join(__dirname, '..', '.state.json');
const ADMIN_FILE = path.join(__dirname, '..', '.admin-key');

let ADMIN_KEY;
if (fs.existsSync(ADMIN_FILE)) {
  ADMIN_KEY = fs.readFileSync(ADMIN_FILE, 'utf8').trim();
} else {
  ADMIN_KEY = 'fc_admin_' + crypto.randomBytes(16).toString('hex');
  fs.writeFileSync(ADMIN_FILE, ADMIN_KEY);
  console.log('◊·κ admin key generated → .admin-key');
}

fs.mkdirSync(LOG_DIR, { recursive: true });

let STATS = {
  started: Date.now(),
  local_calls: 0,
  frontier_calls: 0,
  frontier_tokens_in: 0,
  frontier_tokens_out: 0,
  local_tokens_in: 0,
  local_tokens_out: 0,
  by_tool: {},                     // { 'fallforce': { local: 12, frontier: 1 }, ... }
  by_model: {},
  errors: 0,
  last_calls: []                   // ring buffer of last 50
};
try {
  if (fs.existsSync(STATE_FILE)) STATS = Object.assign(STATS, JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')));
} catch (_) {}
function persistStats() { try { fs.writeFileSync(STATE_FILE, JSON.stringify(STATS, null, 2)); } catch (_) {} }
// Persist every 30s
setInterval(persistStats, 30000);

const app = express();
app.use(cors({
  origin: '*',
  exposedHeaders: ['x-fallcore-tier', 'x-fallcore-model', 'x-fallcore-ms', 'x-fallcore-saved-usd']
}));
app.use(express.json({ limit: '10mb' }));

// Rolling cost estimate (anthropic sonnet-4 list price · approx)
const FRONTIER_COST_PER_MIL_IN  = 3.0;   // USD per 1M input tokens (sonnet)
const FRONTIER_COST_PER_MIL_OUT = 15.0;
function estFrontierCost(inT, outT) {
  return (inT / 1e6) * FRONTIER_COST_PER_MIL_IN + (outT / 1e6) * FRONTIER_COST_PER_MIL_OUT;
}

// Heuristic local-confidence scorer
// Tiny LLMs return short, hedged, or refusal-shaped answers when uncertain.
// Real confidence requires logprobs (which Ollama does support, future patch).
// For v0: length + refusal-pattern check.
const REFUSAL_PATTERNS = [
  /I('| a)m (just |only |a )?(an )?(AI|language model|assistant)/i,
  /I don't (have|know|understand)/i,
  /I cannot/i,
  /Without more (context|information|details)/i,
  /(I('| a)m )?unable to/i,
  /As an (AI|assistant)/i
];
function scoreLocalConfidence(answerText) {
  if (!answerText) return 0;
  const len = answerText.length;
  if (len < 20) return 0.3;
  let pen = 0;
  for (const re of REFUSAL_PATTERNS) if (re.test(answerText)) pen += 0.25;
  let base = Math.min(0.95, 0.4 + Math.log10(Math.max(1, len)) * 0.18);
  return Math.max(0.05, base - pen);
}

// Log every call to disk · one JSONL per day
function logCall(record) {
  STATS.last_calls.unshift(record);
  if (STATS.last_calls.length > 50) STATS.last_calls.length = 50;
  if (record.tier === 'local') {
    STATS.local_calls++;
    STATS.local_tokens_in += record.tokens_in || 0;
    STATS.local_tokens_out += record.tokens_out || 0;
  } else if (record.tier === 'frontier') {
    STATS.frontier_calls++;
    STATS.frontier_tokens_in += record.tokens_in || 0;
    STATS.frontier_tokens_out += record.tokens_out || 0;
  }
  const tool = record.tool || 'unknown';
  STATS.by_tool[tool] = STATS.by_tool[tool] || { local: 0, frontier: 0 };
  STATS.by_tool[tool][record.tier]++;
  STATS.by_model[record.model] = (STATS.by_model[record.model] || 0) + 1;
  try {
    const day = new Date().toISOString().slice(0, 10);
    fs.appendFileSync(path.join(LOG_DIR, day + '.jsonl'), JSON.stringify(record) + '\n');
  } catch (e) { /* swallow */ }
}

// ─── Ollama backend ─────────────────────────────────────
async function callOllama(model, system, messages, opts) {
  const t0 = Date.now();
  const body = {
    model: model || OLLAMA_MODEL,
    messages: [
      ...(system ? [{ role: 'system', content: system }] : []),
      ...messages
    ],
    stream: false,
    options: {
      temperature: opts?.temperature ?? 0.7,
      num_predict: opts?.max_tokens || 1024
    }
  };
  const r = await fetch(OLLAMA_URL + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error('ollama ' + r.status + ' ' + (await r.text()).slice(0, 200));
  const j = await r.json();
  return {
    text: j.message?.content || '',
    tokens_in: j.prompt_eval_count || 0,
    tokens_out: j.eval_count || 0,
    ms: Date.now() - t0,
    model: body.model
  };
}

// ─── Anthropic frontier fallthrough ─────────────────────
async function callAnthropic(model, system, messages, opts, apiKey) {
  const t0 = Date.now();
  const r = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey || ANTHROPIC_FALLBACK_KEY,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: model || 'claude-sonnet-4-20250514',
      max_tokens: opts?.max_tokens || 2048,
      system: system,
      messages: messages
    })
  });
  if (!r.ok) throw new Error('anthropic ' + r.status + ' ' + (await r.text()).slice(0, 200));
  const j = await r.json();
  return {
    text: j.content[0].text,
    tokens_in: j.usage?.input_tokens || 0,
    tokens_out: j.usage?.output_tokens || 0,
    ms: Date.now() - t0,
    model: j.model || model
  };
}

// ═══════════════════════════════════════════════════════════
// POST /v1/messages — Anthropic Messages API shape
// ═══════════════════════════════════════════════════════════
app.post('/v1/messages', async (req, res) => {
  const { model, max_tokens, system, messages, fallcore_options = {} } = req.body;
  const tool = req.headers['x-fallcore-tool'] || req.body?.metadata?.tool || 'unknown';
  const userKey = req.headers['x-api-key'] || '';
  const forceFrontier = fallcore_options.force_frontier || req.headers['x-fallcore-force-frontier'] === '1';
  const callId = crypto.randomBytes(6).toString('hex');

  // Step 1: try local Ollama (unless forced frontier)
  let localResult = null, localError = null;
  if (!forceFrontier) {
    try {
      localResult = await callOllama(
        fallcore_options.local_model,
        system,
        messages,
        { temperature: req.body.temperature, max_tokens: max_tokens || 1024 }
      );
    } catch (e) {
      localError = e.message;
    }
  }

  // Step 2: decide if local is good enough
  let useLocal = false;
  if (localResult) {
    const score = scoreLocalConfidence(localResult.text);
    if (score >= CONFIDENCE_THRESHOLD) useLocal = true;
    localResult.confidence = score;
  }

  if (useLocal) {
    const record = {
      id: callId, t: Date.now(), tier: 'local', tool, model: localResult.model,
      tokens_in: localResult.tokens_in, tokens_out: localResult.tokens_out, ms: localResult.ms,
      confidence: localResult.confidence,
      saved_usd: estFrontierCost(localResult.tokens_in, localResult.tokens_out),
      system_chars: (system || '').length, prompt_chars: JSON.stringify(messages).length,
      response_chars: localResult.text.length
    };
    logCall(record);
    // Anthropic-shaped response
    res.setHeader('x-fallcore-tier', 'local');
    res.setHeader('x-fallcore-model', localResult.model);
    res.setHeader('x-fallcore-ms', String(localResult.ms));
    res.setHeader('x-fallcore-saved-usd', record.saved_usd.toFixed(6));
    res.json({
      id: 'msg_fc_' + callId,
      type: 'message', role: 'assistant',
      model: localResult.model,
      content: [{ type: 'text', text: localResult.text }],
      stop_reason: 'end_turn',
      usage: { input_tokens: localResult.tokens_in, output_tokens: localResult.tokens_out },
      fallcore: { tier: 'local', confidence: localResult.confidence, ms: localResult.ms, saved_usd: record.saved_usd }
    });
    return;
  }

  // Step 3: fall through to frontier
  if (!userKey && !ANTHROPIC_FALLBACK_KEY) {
    STATS.errors++;
    res.status(503).json({
      error: 'no frontier key',
      detail: 'Local model confidence below threshold' + (localResult ? ' (' + localResult.confidence.toFixed(2) + ')' : ' (local error: ' + localError + ')') + '. Provide x-api-key header or set ANTHROPIC_FALLBACK_KEY env.',
      local_attempted: !!localResult,
      local_error: localError
    });
    return;
  }
  try {
    const frontierResult = await callAnthropic(model, system, messages, { max_tokens }, userKey || ANTHROPIC_FALLBACK_KEY);
    const record = {
      id: callId, t: Date.now(), tier: 'frontier', tool, model: frontierResult.model,
      tokens_in: frontierResult.tokens_in, tokens_out: frontierResult.tokens_out, ms: frontierResult.ms,
      cost_usd: estFrontierCost(frontierResult.tokens_in, frontierResult.tokens_out),
      local_attempted: !!localResult,
      local_confidence: localResult?.confidence,
      local_response: localResult?.text?.slice(0, 4000),
      frontier_response: frontierResult.text?.slice(0, 4000),
      system_chars: (system || '').length, prompt_chars: JSON.stringify(messages).length
    };
    logCall(record);
    res.setHeader('x-fallcore-tier', 'frontier');
    res.setHeader('x-fallcore-model', frontierResult.model);
    res.setHeader('x-fallcore-ms', String(frontierResult.ms));
    res.json({
      id: 'msg_fc_' + callId,
      type: 'message', role: 'assistant',
      model: frontierResult.model,
      content: [{ type: 'text', text: frontierResult.text }],
      stop_reason: 'end_turn',
      usage: { input_tokens: frontierResult.tokens_in, output_tokens: frontierResult.tokens_out },
      fallcore: { tier: 'frontier', cost_usd: record.cost_usd, ms: frontierResult.ms, local_confidence: localResult?.confidence }
    });
  } catch (e) {
    STATS.errors++;
    res.status(502).json({ error: 'frontier failed', detail: e.message, local_error: localError });
  }
});

// ═══════════════════════════════════════════════════════════
// POST /v1/chat/completions — OpenAI shape (bonus)
// ═══════════════════════════════════════════════════════════
app.post('/v1/chat/completions', async (req, res) => {
  // Normalise OpenAI shape into Anthropic shape, then re-route through /v1/messages
  const { model, messages = [], max_tokens, temperature } = req.body;
  const systemMsg = messages.find(m => m.role === 'system');
  const userMsgs = messages.filter(m => m.role !== 'system');
  // Forward
  req.body = {
    model, max_tokens, temperature,
    system: systemMsg?.content,
    messages: userMsgs,
    fallcore_options: req.body.fallcore_options || {}
  };
  req.url = '/v1/messages';
  app._router.handle(req, res, () => {});
});

// ═══════════════════════════════════════════════════════════
// GET /v1/models
// ═══════════════════════════════════════════════════════════
app.get('/v1/models', async (req, res) => {
  try {
    const r = await fetch(OLLAMA_URL + '/api/tags');
    const j = await r.json();
    const local = (j.models || []).map(m => ({ id: m.name, owned_by: 'local', tier: 'T1' }));
    const frontier = [
      { id: 'claude-sonnet-4-20250514', owned_by: 'anthropic', tier: 'T3' },
      { id: 'claude-opus-4-20250514', owned_by: 'anthropic', tier: 'T3' }
    ];
    res.json({ object: 'list', data: [...local, ...frontier] });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════════════
// GET /health
// ═══════════════════════════════════════════════════════════
app.get('/health', async (req, res) => {
  let ollamaUp = false, ollamaModels = [];
  try {
    const r = await fetch(OLLAMA_URL + '/api/tags', { signal: AbortSignal.timeout(2000) });
    if (r.ok) { ollamaUp = true; ollamaModels = (await r.json()).models?.map(m => m.name) || []; }
  } catch (_) {}
  const uptimeMs = Date.now() - STATS.started;
  const ratio = STATS.local_calls + STATS.frontier_calls > 0
    ? STATS.local_calls / (STATS.local_calls + STATS.frontier_calls) : 0;
  const totalSaved = STATS.last_calls.filter(c => c.tier === 'local').reduce((a, c) => a + (c.saved_usd || 0), 0);
  res.json({
    status: ollamaUp ? 'ok' : 'degraded',
    version: '0.1.0',
    uptime_ms: uptimeMs,
    ollama: { up: ollamaUp, url: OLLAMA_URL, models: ollamaModels, default_model: OLLAMA_MODEL },
    frontier: { configured: !!ANTHROPIC_FALLBACK_KEY },
    confidence_threshold: CONFIDENCE_THRESHOLD,
    stats: {
      local_calls: STATS.local_calls,
      frontier_calls: STATS.frontier_calls,
      local_ratio: ratio.toFixed(3),
      usd_saved_recent: totalSaved.toFixed(4),
      errors: STATS.errors
    }
  });
});

// ═══════════════════════════════════════════════════════════
// GET /v1/stats — full dashboard data
// ═══════════════════════════════════════════════════════════
app.get('/v1/stats', (req, res) => {
  const total = STATS.local_calls + STATS.frontier_calls;
  const localRatio = total > 0 ? STATS.local_calls / total : 0;
  const totalSavedRecent = STATS.last_calls.filter(c => c.tier === 'local').reduce((a, c) => a + (c.saved_usd || 0), 0);
  const totalSpentRecent = STATS.last_calls.filter(c => c.tier === 'frontier').reduce((a, c) => a + (c.cost_usd || 0), 0);
  res.json({
    uptime_ms: Date.now() - STATS.started,
    calls: { local: STATS.local_calls, frontier: STATS.frontier_calls, total, local_ratio: localRatio },
    tokens: {
      local: { in: STATS.local_tokens_in, out: STATS.local_tokens_out },
      frontier: { in: STATS.frontier_tokens_in, out: STATS.frontier_tokens_out }
    },
    by_tool: STATS.by_tool,
    by_model: STATS.by_model,
    cost: {
      saved_usd_recent: totalSavedRecent,
      spent_usd_recent: totalSpentRecent,
      projection_annual: localRatio > 0 ? `at ${(localRatio*100).toFixed(1)}% local, projecting ~$${((totalSavedRecent / Math.max(1, total)) * 100000).toFixed(0)} saved per 100k calls` : 'insufficient data'
    },
    errors: STATS.errors
  });
});

// ═══════════════════════════════════════════════════════════
// GET /v1/log — recent prompts (admin only)
// ═══════════════════════════════════════════════════════════
app.get('/v1/log', (req, res) => {
  const key = req.headers.authorization?.replace('Bearer ', '') || req.headers['x-admin-key'];
  if (key !== ADMIN_KEY) return res.status(403).json({ error: 'admin only' });
  res.json({ last_50: STATS.last_calls });
});

// Static landing
app.use(express.static(path.join(__dirname, '..', 'public')));

app.listen(PORT, () => {
  console.log('━'.repeat(60));
  console.log('◊·κ=1 FallCore Proxy v0.1.0');
  console.log('━'.repeat(60));
  console.log('Listening on  http://localhost:' + PORT);
  console.log('Ollama URL    ' + OLLAMA_URL);
  console.log('Default model ' + OLLAMA_MODEL);
  console.log('Confidence    >= ' + CONFIDENCE_THRESHOLD + ' → local stays · else fall to frontier');
  console.log('Frontier key  ' + (ANTHROPIC_FALLBACK_KEY ? 'set' : 'NOT SET (will 503 on cascade)'));
  console.log('Logs dir      ' + LOG_DIR);
  console.log('Admin key     ' + ADMIN_KEY.slice(0, 16) + '… (full in .admin-key)');
  console.log('');
  console.log('Point any Anthropic-API client at this URL · drop-in replacement.');
  console.log('GET  /health    · status + dashboard');
  console.log('GET  /v1/stats  · ROI + token-spend data');
  console.log('━'.repeat(60));
});
