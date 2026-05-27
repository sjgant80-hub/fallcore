# ◊·κ=1 · FallCore

**The on-prem brain that learns from your work.**

Anthropic-API-compatible local proxy + RAG + LoRA fine-tune loop. Drop-in replacement for `api.anthropic.com` that routes most queries to a local Ollama-served model (Qwen2.5-32B / 72B / Llama-3.1-8B), falls back to frontier only when local confidence is insufficient, and logs every cascade-fire for the next fine-tune cycle. Frontier-spend ratio drops over time as the local model learns your work.

Part of the [Fall* estate](https://github.com/sjgant80-hub). 30+ sovereign tools all aligned with the same Konomi / KCC / mesh architecture.

---

## Why this exists

Every major AI vendor's business model is **token-metered cognition**. Their growth requires your usage growing.

Enterprises burning £50k–£500k/yr on frontier APIs hit three walls:

1. **Compliance.** Their data legally can't be sent to a US-based vendor (HIPAA, GDPR, financial-services PII, SOX).
2. **Cost.** The bill grows with adoption. Successful AI deployment makes the bill worse.
3. **Lock-in.** Every reviewer correction trains the vendor's model, not theirs. Switching costs are infinite.

FallCore inverts all three:

- **Compliance:** Model + data live on the customer's hardware. Nothing leaves the network.
- **Cost:** Flat fee, decoupled from usage. Cheaper as the local model learns and frontier ratio drops.
- **Ownership:** The fine-tuned adapter IS the customer's IP. They can fire us and it still runs.

---

## Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Customer's tools (Fall* estate, custom apps, anything      │
   │  that hits Anthropic's API shape)                            │
   └────────────────────────────┬────────────────────────────────┘
                                │  POST /v1/messages
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  FallCore Proxy (this repo · port 11434)                    │
   │                                                              │
   │   1. Try local Ollama                                        │
   │   2. Score response confidence                               │
   │   3. If confidence >= threshold → return local               │
   │      Else → fall through to frontier (real Anthropic)        │
   │   4. Log every call to JSONL (training data)                 │
   └────────────┬───────────────────────────────┬───────────────┘
                │                                │
                ▼                                ▼
   ┌────────────────────────┐    ┌──────────────────────────────┐
   │  Ollama (local · GPU)  │    │  Anthropic frontier (fallback)│
   │  Qwen2.5-32B / 72B     │    │  ONLY when confidence low     │
   └────────────────────────┘    └──────────────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │  Logs → LoRA fine-tune cycle    │
                              │  Adapter deployed into Ollama   │
                              │  Local confidence rises         │
                              │  Frontier ratio falls           │
                              └─────────────────────────────────┘
```

---

## Quickstart

### Pre-reqs

- Docker + Docker Compose
- Optional: NVIDIA GPU (24GB+ for 32B model, 48GB+ for 72B). CPU-only works but slow.
- Optional: Anthropic API key for fallthrough on low-confidence queries.

### 1 · Clone and configure

```bash
git clone https://github.com/sjgant80-hub/fallcore
cd fallcore
cp .env.example .env
# edit .env:
#   OLLAMA_MODEL=qwen2.5:32b        (or llama3.1:8b for lighter hardware)
#   ANTHROPIC_FALLBACK_KEY=sk-ant-… (optional; leave blank for local-only)
#   CONFIDENCE_THRESHOLD=0.6        (start here; lower as trust grows)
```

### 2 · Bring up the stack

```bash
docker compose up -d
```

This launches three services:
- **ollama** (port 11435 on host) — local model runtime
- **qdrant** (ports 6333, 6334) — vector DB for RAG (optional, you can leave it idle)
- **proxy** (port 11434) — the Anthropic-shape front-end

### 3 · Pull the model (one-time)

```bash
docker compose exec ollama ollama pull qwen2.5:32b
```

### 4 · Point your existing apps at FallCore

For any Anthropic client (their official SDK, raw curl, third-party libraries):

```bash
export ANTHROPIC_BASE_URL=http://your-host:11434
```

That's it. Your apps now use local-first cascade. No code changes.

### 5 · Verify

```bash
curl http://localhost:11434/health
# { "status": "ok", "ollama": {...}, "stats": {...} }

curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: anything" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [{"role":"user","content":"hello"}]
  }'
```

Response will have `x-fallcore-tier: local` header if the local model handled it.

---

## After a week · run the eval

```bash
node eval/replay.js --days 7
```

This replays past frontier calls through the local model and scores equivalence. Output tells you:

- What % of frontier calls were unnecessary
- £ saved by the cascade
- Whether to lower the confidence threshold for more savings
- Whether to upgrade the model OR train a LoRA

---

## After a month · run the LoRA cycle

```bash
# 1. Extract preference pairs from logs
node train/extract-preferences.js --days 30 --out training-2026-Q2/

# 2. Train (outside this repo — use axolotl, unsloth, or TRL DPO)
#    axolotl train training-2026-Q2/train.jsonl

# 3. Deploy the adapter into Ollama
ollama create my-qwen2.5-32b-tuned -f Modelfile-with-adapter

# 4. Re-run eval to measure improvement
node eval/replay.js --days 7
```

Repeat monthly (or weekly for high-volume customers). Each cycle raises local accuracy → lowers frontier ratio.

---

## Hardware reference

| Tier | Model | VRAM | RAM | Disk | Typical latency |
|---|---|---:|---:|---:|---:|
| Lite | Llama-3.1-8B | 8GB | 16GB | 20GB | 1-3s |
| Pro | Qwen2.5-32B | 24GB | 32GB | 50GB | 3-8s |
| Sovereign | Qwen2.5-72B | 48GB | 64GB | 100GB | 5-15s |
| Sovereign+ | Llama-3.1-405B (rare) | 250GB+ | 512GB | 800GB | 10-30s |

For CPU-only:
- Lite (8B Q4): 16GB RAM, ~10-20s latency, viable for low-volume
- Pro+ on CPU not recommended.

---

## Endpoint reference

```
POST /v1/messages          · Anthropic Messages API · drop-in
POST /v1/chat/completions  · OpenAI Chat shape (same backend)
GET  /v1/models            · list local + frontier models
GET  /health               · status + ollama health + stats
GET  /v1/stats             · ROI dashboard (calls, tokens, USD saved)
GET  /v1/log               · recent prompts (admin key required)
```

Custom headers:
- `x-fallcore-tool: <tool-name>` — for per-tool stats breakdown
- `x-fallcore-force-frontier: 1` — bypass local, force frontier (debugging)

Response headers:
- `x-fallcore-tier: local | frontier` — which path served this call
- `x-fallcore-model: <model-id>`
- `x-fallcore-ms: <latency>`
- `x-fallcore-saved-usd: <amount>` (on local responses)

---

## Tiers (by hardware)

| Tier | Model | VRAM | Frontier-call reduction | Notes |
|---|---|---|---:|---|
| **Lite** | Llama-3.1-8B | 8GB | ~40% | Laptops + small teams · CPU also viable |
| **Pro** | Qwen2.5-32B | 24GB | ~75% | Departmental · weekly LoRA cadence |
| **Sovereign** | Qwen2.5-72B | 48GB | ~95% | Regulated industries · server-class |
| **Enterprise** | Qwen2.5-72B | 48GB+ | ~99% | Multi-region · custom certifications |

**Free during launch.** Open source, MIT-licensed. Forge a branded stack at the [FallCore Factory](https://sjgant80-hub.github.io/fallcore-factory/) — no card, no commitment. Commercial managed-tier pricing will be set once volume + feedback are in.

---

## What's open source vs. proprietary

**MIT-licensed (this repo):**
- The proxy server (`proxy/server.js`)
- Docker compose stack
- Eval harness (`eval/replay.js`)
- LoRA preference extractor (`train/extract-preferences.js`)
- Landing page

**Proprietary / commercial:**
- Managed deployment service
- Custom LoRA training (we do the training; you keep the adapter)
- Compliance certifications (SOC2, HIPAA, ISO 27001) at Sovereign tier+
- 24/7 ops + bespoke SLAs at Enterprise tier
- Cross-customer LoRA adapter marketplace (via KCC) — opt-in

---

## Integration with the Fall* estate

Every Fall* tool ships with the LLM cascade shim. To point any Fall* tool at your FallCore deployment:

1. Open the tool in your browser.
2. Click the `● AI · T3` badge bottom-left.
3. Paste your FallCore endpoint (e.g. `https://ai.acme.internal` or `http://localhost:11434`) into the "FallCore (on-prem)" field.
4. Save.

The tool now routes through your local brain. Tier indicator changes to `T1·FallCore`.

When the cascade fails (local server down), it falls through automatically to direct API keys → OnlyBrains → WebLLM → mesh → T0 — same sovereign resilience as before.

---

## Roadmap

- [ ] **vLLM backend option** — for high-throughput / multi-user production deployments
- [ ] **TensorRT-LLM backend** — NVIDIA-optimised, batched inference
- [ ] **Built-in LoRA training loop** — currently external (axolotl/unsloth); will add managed training
- [ ] **RAG ingestion pipelines** — current Qdrant container is empty; add adapters for Confluence, SharePoint, Notion, Gmail, Outlook
- [ ] **Cross-customer adapter marketplace** — KCC-based discovery and licensing
- [ ] **Dashboard UI** — currently CLI + JSON; add a real-time dashboard at `/`

---

## Licence

MIT for the code in this repo. Konomi licence (Ed25519) for commercial managed-tier features. See `LICENCE`.

---

## Contact

- GitHub: https://github.com/sjgant80-hub/fallcore
- Estate: https://github.com/sjgant80-hub
- Email: simon@ai-nativesolutions.com
- Site: https://ai-nativesolutions.com

◊·κ=1
