# `h3ir` — a prompt compiler for MiniMax-H3

MiniMax's own model card says the quiet part out loud:

> **H3-Context-IR** deeply understands and refines the input multimodal instructions, then converts
> them into a form that H3 can readily understand. […] **H3-Context-IR is critical to the quality of
> the final output.**

…and then says it is **not part of the open-source release**. So whatever a caller sends to SGLang
*is* the IR. `../PROMPT_IR.md` measured what that costs: the same request, same weights, same seed,
as one Chinese sentence versus as documented IR, is the difference between prosodically-plausible
babble and the exact line coming out of the model's mouth. At 25 **and** 50 steps.

This directory reproduces the part of H3-Context-IR that can be reproduced. It is a **text-to-text
compiler with a hard validator**, which is the cheapest possible thing to build against a closed
system whose output format is fully documented:

```
user request  ──►  Bedrock (Haiku 4.5)  ──►  validator  ──►  prompt
   (Chinese,          system prompt =            36 rules         │
    1 sentence,       official guides verbatim   with IDs         │ fail closed
    + assets)       + measured rules                             │
                    + gold few-shots                             ▼
                            ▲                             violations fed back
                            └───────── repair loop ──────────────┘
```

The validator is the load-bearing part. **Anyone can ask an LLM to "write it in MiniMax's format";
the difference between that and a compiler is whether a prompt that got the format wrong can
possibly reach the GPU.** Here it cannot: `CompileResult.prompt` is `None` unless the last attempt
is error-free. Three seconds of retry against a 105–180 s render is a trade with no downside.

## Measured: does it work, and which model

`python -m h3ir.cli bench --models haiku45,opus5` — four requests chosen because each one can fail a
different rule.

**These numbers are cited, not reproducible from this repo as it stands.** Two of the four bench
requests described a customer's shot and have been replaced with neutral equivalents of the same
difficulty, and the raw output logs were withdrawn with the rest of that material. The table below is
what was measured on the previous set; re-running `bench` on the current set is one command per model
and is the honest way to restore it. `logs/repair_trace_gpt6.json` is still here because that case (the
headphone advert) was never customer material.

| model | first-pass validator-clean | validated after repair | mean repairs | mean latency | slowest case |
|---|---|---|---|---|---|
| `us.anthropic.claude-haiku-4-5` | **4 / 4** | 4 / 4 | 0.00 | **8.1 s** | 11.3 s |
| `us.anthropic.claude-sonnet-4-6` | 4 / 4 | 4 / 4 | 0.00 | 10.1 s | 13.2 s |
| `us.anthropic.claude-sonnet-5` | 4 / 4 | 4 / 4 | 0.00 | 18.7 s | 29.7 s |
| `us.anthropic.claude-opus-5` | 4 / 4 | 4 / 4 | 0.00 | 19.5 s | 22.4 s |
| `us.openai.gpt-6-astra` | 3 / 4 | 4 / 4 | 0.25 | 55.5 s | 98.4 s |

**Default to Haiku 4.5.** It is 2.4× faster than Opus 5 at the same 4/4 first-pass rate, and the
reason a cheap model is safe here is the validator: format compliance is the failure mode a small
model has, and the failure mode a mechanical checker catches for free. Every model above clears the
bar the closed system sets — MiniMax's own documented H3-Context-IR call took **29 s** of server
time for one 5 s clip (see *The oracle* below), so 8 s is not a compromise, it is the fast path.

The compile is **output-bound** (~900 tokens out against a fixed ~8k-token system prompt), so the
model choice is the whole latency lever. `cache_system=True` (default; `--no-cache` to disable) puts
a Bedrock `cachePoint` after the system prompt: that is a cost lever and a help on the repair call,
not an answer to the latency budget.

GPT-6's one failure is
worth naming because it is exactly the kind of thing a human reviewer does not catch: on the
headphone advert — a product film with **nobody on camera** — it wrote a voiceover `<d>` and omitted
the mandatory clause stating that no visible mouth is speaking. `E024` named it, the model fixed
that one clause, and the second attempt validated. `logs/repair_trace_gpt6.json` is the full trace.

The compiled prompts were not merely legal — e.g. the market scene laid down two speakers with
distinct voice descriptions, timed cuts at `00:03.400` and `00:06.800`, on-screen text as
`"西红柿 2.5元/斤"`, and 112 BPM non-diegetic music the characters cannot hear. That request is one of
the two still in the bench set, so it is the cheapest one to re-verify.

## The validator, and why it is calibrated against renders rather than against the guides

`h3ir/validate.py`, **36 rules**, each carrying an ID and pointing at the normative sentence or the
experiment behind it. Two severities: `error` fails closed; `warn` (word counts, syllable rate) is
reported and never blocks.

**Calibration corpus** — `ir/tests/test_validate.py`, 45 tests:

* `golds/nvlabs_solh3_13.json` — NVlabs' 13 validated Sol-H3 showcase prompts. All 13 rendered.
  All 13 must pass with **zero** errors and zero warnings, and they do.
* `golds/minimax_official_api.json` + `golds/minimax_official_harvest_cn.json` — **real output from
  the closed H3-Context-IR compiler**: MiniMax's documented example plus live API calls. Ground
  truth; a rule that fires here is a validator bug unless the case says otherwise. See *The oracle*.
  Two of the four harvested calls are **not** in the repo: they probed the compiler with a customer's
  shot. The fixture's `note` records what they showed and how to re-harvest an equivalent.
* `../case/demo_ir.txt` — a synthetic pair in the format of this repo's render grid. The `ref2va` arm
  must pass; the `t2va` arm must **fail on exactly one rule**, `E030-D-MIXED-SCRIPT-TOKEN`, standing
  in for the arm whose audio came out as noise (`../PROMPT_IR.md` §3.1.1). A validator that passed it
  would be useless. The repaired arm — the same line with the digit-letter token spelled out, which is
  also what `guides.golds()` hands the model as a few-shot — must pass.

  Synthetic because the measured prompts were a customer's and are not in this repo. The *rules* they
  taught us are all here; the prompts are not, so treat `demo_ir.txt` as a shape and every number in
  `../PROMPT_IR.md` as having been measured on text you cannot see.

Writing the rules from the guides and then running them against that corpus is what found the
three places where **the guides are stricter than the practice that actually renders.** Each one
would have made the compiler reject known-good output:

1. **Field separator.** Base guide §2.2 shows fields separated by a blank line. All 13 validated
   prompts use a single `\n`. Both are accepted; what is rejected is a field name appearing
   mid-line (`E004`), which is what a model emits when it treats the format as prose.
2. **Speaker IDs.** `../PROMPT_IR.md` §4.1 assumed, from the guide's short examples, that *every*
   `<d>` is preceded by a speaker ID. It is not: the validated prompts bind `(S1)` where the
   speaker is **introduced**, together with pitch/timbre/rate, and attribute later lines by name
   (`the tea master says:`). Requiring adjacency would have rejected all 13. What is enforceable
   and was kept: IDs exist, are dense from `S1`, never appear inside `<d>` (`E026`), the first one
   precedes the first `<d>` (`E025`), and each `<d>` follows an attribution ending in `:` or `,`
   (`E027` — true of all 38 dialogue blocks in the corpus).
3. **The voiceover clause has (at least) three legal forms.** The guide gives only "state that the
   on-screen character's lips remain closed". Case `16-liquid-metal-chronograph` has no on-screen
   character at all and discharges the obligation with `while no speaking face appears on screen`.
   MiniMax's own API writes a third: `with her unseen mouth remaining completely still throughout`.
   All three are accepted; the real obligation is "assert that no visible mouth is speaking", and
   three independent phrasings for one requirement is the signal that the requirement, not the
   wording, is what to check.

The rules that come from measurement rather than from either guide are in `h3ir/guides.py`
(`MEASURED_RULES`, given to the model) and enforced in `validate.py`:

| rule | what it catches | evidence |
|---|---|---|
| `E030-D-MIXED-SCRIPT-TOKEN` | a digit welded to a letter inside `<d>` | `2x等于4` is voiced as **"rx"** while `等于4` and `等于2` in the same sentence transcribe correctly. A digit-letter token is a formula and has no grapheme→phoneme path. Fix: `二x`. §3.1.1 |
| `E031-D-MATH-OPERATOR` | `= + * / ^ %` inside `<d>` | same mechanism. `<scenetrans>`/`<cutoff>` are stripped first — their brackets are markup, not math, and forgetting that was a real false positive |
| `W032-SYLLABLE-BUDGET` | a script that cannot be said in the duration | ~20 unhurried Mandarin syllables per 5.04 s. Over budget means buy frames, not talk faster |
| `E024` (widened) | a voiceover with no "no visible mouth" assertion | see above |

## The oracle: MiniMax ships H3-Context-IR as an API

`POST /v2/h3_context_ir` on `api.minimaxi.com` (CN) or `api.minimax.io` (global) —
`h3ir/oracle.py`, `cli oracle` / `cli harvest`.
The endpoint's own description says it plainly: *"H3-Context-IR is a complex system and its
implementation is not open sourced. This API can be used both to validate the official Full
2K-Workflow results and in production workflows."* It returns **only the enhanced prompt**, no video.

| | |
|---|---|
| shape | async — create returns `task_id`, poll `GET /v2/query/video_generation/{task_id}` until `succeeded` |
| auth | `Authorization: Bearer $MINIMAX_API_KEY`, pay-as-you-go video must be enabled |
| input | `model: MiniMax-H3`, `content[]` of `text` / `image_url` / `video_url` / `audio_url` with `role` ∈ `first_frame`, `last_frame`, `reference_image`, `reference_video`, `reference_audio`; `duration` 4–15 **integer** seconds; `ratio` (required and non-`adaptive` for text-only) |
| output | `task.content.prompt` — the same `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music` fields, single `\n` separated |
| price | $0.90 / M tokens in, $3.60 / M out → **$0.006–0.020 per call**, measured across 5 live calls |
| latency | **20–35 s server-side, mean 24.2 s** across 5 live calls (29 s on the documented example) |

**The host is a trap.** MiniMax runs two independent platforms with separate accounts and
non-interchangeable keys, and the OpenAPI spec documents only the international one. A CN-console key
against `api.minimax.io` returns HTTP 401 `invalid api key (2049)` — indistinguishable from a bad key.
Measured 2026-09-18: the same key that 401s on `.io` returns 200 on **`api.minimaxi.com`** (and on the
legacy `api.minimax.chat`). `oracle.py` defaults to the CN host, takes `MINIMAX_REGION=cn|global`, and
says so in the 401 message. The CN create response is also a flat `{"task_id": "..."}`, not the
`VideoGenerationV2Resp` the spec types it as.

Three things follow, and they matter more than anything else in this README:

**1. It is ground truth, and the validator now passes it — after the ground truth found two bugs.**
`golds/minimax_official_api.json` (the documented example) and `golds/minimax_official_harvest_cn.json`
(4 live calls, $0.05) are asserted in `tests/`. On first contact, two rules fired on correct official
output and both were **our** bugs:

* `E031-D-MATH-OPERATOR` fired four times on a two-shot voiceover, because its character class
  included `<` and `>` and so counted the brackets of a legal, *mandatory* in-`<d>` `<scenetrans>`
  marker as math. Fixed by stripping `<scenetrans>`/`<cutoff>` before the operator scan.
* `E024-VOICEOVER-NO-LIPS-CLAUSE` knew two phrasings for "no visible mouth is speaking"; the official
  compiler writes a third, *"with her unseen mouth remaining completely still throughout"*. Three
  independent phrasings for one requirement is the signal that the requirement, not the wording, is
  what to check. Widened accordingly.

This is the third time the calibrate-against-output discipline has paid: guides describe, corpora
decide.

**2. `E030` is a defect the official IR does not fix — which is the case for keeping this layer.**
Asked for a subject to explain an equation out loud with **no script supplied**, the official compiler
invented a spoken line containing `2x` entirely on its own. `2x` renders as **"rx"** (PROMPT_IR.md
§3.1.1, measured). Our Haiku compile of the byte-identical request wrote `二x等于四`. Matched request,
matched task, matched duration:

| | official H3-Context-IR | `h3ir` + Haiku 4.5 |
|---|---|---|
| latency | 20 s | **6.8 s** |
| cost | $0.0063 | ≈$0.013 uncached, ≈$0.005 cached |
| spoken line | `2x等于4` → voiced "rx" | `二x等于四` |
| validator | 1 error | clean, first pass |

Not a claim that our IR is better *cinematically* — theirs is richer, and it is the system the model
was trained alongside. The claim is narrower and stands on a measurement: it does not guard the one
audio failure we have actually observed on this model.

The two API responses behind that table were harvested with the customer's own request and are no
longer in the repo, so this row is a **cited measurement, not a reproducible fixture**. Re-running the
probe on a scene of your own is one call and about $0.014 — see the fixture's `note`.

**3. It is a teacher, not a production path.** 24 s mean is 3.5× our Haiku compile, and it is a
metered network hop in front of a render. But `(request → official IR)` pairs at ~$0.014 each make
distillation arithmetic cheap: 10k pairs ≈ $140, against a day of 8×B300. The ladder, in increasing
cost:

* **Few-shot / retrieval from harvested pairs** — no training. Swap `guides.golds()` from our four
  hand-picked exemplars to nearest-neighbour retrieval over harvested official IR. Almost certainly
  the best value, and it makes the system prompt *smaller*, not larger.
* **Rejection-sampled SFT** — harvest, keep only pairs our validator passes, fine-tune Haiku (Bedrock
  custom model) or a local Qwen2.5-7B. Buys latency and unit cost, not correctness.
* **Preference/RL on render outcomes** — needs the GPU loop that does not exist yet (see below).

Before harvesting at scale, check MiniMax's terms on using API output as training data; that is a
contract question for whoever owns the account, not a technical one.

## Prior art

Eleven public repos mention `H3-Context-IR`. One is serious: **`ruashots/open-h3-ir`** (Apache-2.0,
also named `h3ir` by coincidence) — a contract/acceptance validator, an eval loop, golden files, a
bundled Qwen2.5 tokenizer vocab, ComfyUI nodes and an HTTP service, and the thing this package
lacks: **A/B render evidence**. Its build log reports 3.9–13.1 s compose times, which brackets our
Haiku/Sonnet range. Three of its findings are worth carrying regardless of whose code you use:

1. **Reference labels are emitted by the runtime, not by you.** `comfy/text_encoders/minimax.py`
   writes `"<Picture 1>: "` ahead of each vision block, so IR that says `<Image 1>` is a dangling
   pointer. Matches this package's label rules (`E050`/`E051`).
2. **`<d>` is not a special token.** H3's tokenizer carries the same 26 added tokens as stock
   Qwen2.5, so `<d>` BPE-splits into `['<d', '>']` — it works because the model learned the byte
   sequence, which is exactly why it has to be byte-exact. Good justification for `E020`.
3. **IR text is 0.5–1.6 % of the packed sequence**, while one max-resolution reference image is 7,296
   rows and a 5 s reference video is 37,296. Write long, precise IR; be ruthless about attachments.

Others (`oufeixinxinren/ComfyUI-MiniMax-ContextIR`, `XINGSHEN2/minimax-H3-context-IR`,
`Yi-Biao/MiniMax-H3-Context-IR-Skill`, `leoleelxh/ComfyUI-M3-IRContext`, `gavin2us/open-h3-context-ir`,
`Git-Nicolai/h3-prompt-writing`) are prompt-template or ComfyUI wrappers with no validator and no
calibration corpus.

## Layout

```
h3ir/validate.py    the rules. Pure Python, no network, no model. Report.ok is the gate
h3ir/guides.py      system-prompt assembly: guides verbatim + MEASURED_RULES + gold few-shots
h3ir/compile.py     Bedrock converse + the repair loop + fail-closed CompileResult; route_task()
h3ir/oracle.py      MiniMax's own /v2/h3_context_ir: ground truth, and the distillation teacher
h3ir/cli.py         check / compile / bench / oracle / harvest
golds/              NVlabs' 13 validated showcase prompts, MiniMax's documented API response,
                    4 live-harvested official IR outputs, and the archived OpenAPI spec
tests/              the calibration suite
logs/               bench output and one full repair trace
```

## Use

```bash
# validate an existing prompt -- no model call, no credentials, exit 1 if it would be rejected
python -m h3ir.cli check --task t2va --frames 121 some_prompt.txt

# compile. --script is the caller's OWN words, used verbatim: never let the model write the 台词
python -m h3ir.cli compile --frames 241 --model haiku45 \
    --script '今天的豆子是云南的，牛奶用两升装的。' \
    '固定镜头，中景，一位年轻男咖啡师站在吧台后面，身后小黑板已经写好"TODAY: YUNNAN"，他指着黑板介绍今天的豆子。'

python -m h3ir.cli bench --models haiku45,opus5 --out logs/ir_bench.json
python -m pytest tests -q

# the real thing, for ground truth and for distillation data (metered: ~$0.017 a call)
export MINIMAX_API_KEY=...        # platform.minimax.io -> Account Management -> API Keys
python -m h3ir.cli oracle --duration 5 --ratio 16:9 'A boy playing basketball by the sea'
python -m h3ir.cli harvest --in requests.jsonl --out golds/harvest.jsonl
```

Needs `boto3` and Bedrock access in a region with the `us.`-prefixed inference profiles. Two
Bedrock gotchas are baked in rather than left as traps: the **bare** model ids
(`anthropic.claude-opus-5`) fail with "on-demand throughput isn't supported" — only the `us.`
profiles work; and `us.anthropic.claude-opus-5` rejects `temperature` outright, so
`inferenceConfig` carries only `maxTokens`. `bedrock_client()` also overrides botocore's 60 s read
timeout, which otherwise turns one slow reasoning call into two silent retries and then a
`ReadTimeoutError` — GPT-6 took 176 s on one prompt.

## What this does not do, and what it costs

* **`frames` is chosen by the caller, not inferred.** `route_task()` implements the task routing in
  `../PROMPT_IR.md` §3.3 (an image of the *finished* state of the action is a last keyframe →
  `l2va`/`fl2va`, **not** `ref2va`, which means "keep this subject and scene", not "arrive at this
  frame"), but nothing here decides duration from the length of the requested action. That is
  §4.5 step 3 and it is the next thing worth building — measured rule 5 says an action that cannot
  finish in the duration makes the model compress time and smear.
* **The audio-driven route is not wired.** `../PROMPT_IR.md` §4.3: `ref2va` accepts
  `reference/audio` with `duration_from_audio_reference=True`, which collapses the model's job from
  *invent intelligible Mandarin* to *align lips to this track*. TTS the approved line and send it.
  Note `t2va` has `condition_rules=()`, so that route needs `ref2va` or a keyframe task.
* **No rule here has been validated by a render.** The validator is calibrated so that it *accepts*
  prompts that provably rendered and *rejects* the specific defects that were measured — but the
  compiled prompts the bench produced have not themselves been through the GPU. Doing that on
  the 8-card box is the honest completion of this work, and until then the claim is "these prompts
  are in the documented distribution", not "these prompts render better".
* **Only 5 official samples exist so far**, all `t2va` and all short. The reference (`ref2va`),
  keyframe (`i2va`/`l2va`/`fl2va`) and audio-driven routes have never been put through the real API,
  so the validator's rules for those are still calibrated only against the guides plus this repo's
  own renders.
  Harvesting those is the cheapest remaining correctness win in this package: it needs public asset
  URLs and about $0.05 for four more calls.
* Cost per compile is one to two calls of a ~29 KB system prompt plus ~2 KB out — cents, against a
  105–180 s render. Haiku 4.5 at ~8k in / ~900 out is ≈$0.013 uncached and ≈$0.005 with the system
  prompt cached, which is in the same range as MiniMax's own ≈$0.017 endpoint at a quarter the wait.
