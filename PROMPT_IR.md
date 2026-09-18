# The prompt *is* the IR: why the first two renders were bad, and what to build

The customer's report on the first two renders was blunt and correct, and worth stating as they put
it: the ref2va render had no sense of the writing being formed stroke by stroke, the t2va render's
speech was simply unintelligible, and they suspected the missing piece was the prompt layer — that
you cannot hand a request straight to the model, because MiniMax's own product puts an IR in front of
it and that IR is not in the release.

**THE PROMPTS AND RENDERS THIS DOCUMENT MEASURES ARE NOT IN THIS REPO.** They are a customer's shot
description and a photograph of a real person, so they are untracked (see `.gitignore`) and the contact
sheets made from them are gone. What is here is every rule they taught us, plus synthetic stand-ins in
the same format (`case/demo_ir.txt`, `case/demo_raw.txt`) that the test suite and the compiler's
few-shots run against. Where a finding below rests on a frame you cannot look at, it says so.

**That diagnosis is right, and it is now measured.** Neither failure is the weights, the precision,
the step count or the GPU. Both are the **missing prompt layer**. Rewriting the same two cases into
MiniMax's own documented prompt format — same weights, same seed 42, same 768p / 121 f, same 25
steps, nothing changed but the text — turns unintelligible babble into the exact scripted line and
turns a smeared macro shot into a clean, static, correctly-framed take. (It does **not** fix the
handwriting itself — see §3.4, which is a model limit and not a prompt one.)

| what changed | latency | result |
|---|---:|---|
| nothing (raw 62-char Chinese prompt, 25 steps) | 105.37 s | ASR: `山瞪聰智的李时年好备喜第三号都是进西南价卡` |
| **50 steps** instead of 25, same raw prompt | 213.36 s | ASR: `深邓寵食 研岛歌词 第三行活 景气蓝价格` — *still gibberish* |
| **IR-format prompt** (1036 chars), 25 steps | 105.92 s | ASR: **`…两边同时减3,rx等于4,所以x等于2`** |
| IR-format prompt, 50 steps | 214.33 s | same line, same intelligibility |
| IR prompt with **`二x`** instead of `2x`, 25 steps | 106.74 s | ASR: **`…两边同时减3,2x等于4,所以x等于2`** — clean (§3.1.1) |

The transcripts are elided (`…`) where the line is the customer's own script. What is left is the part
the experiment turns on: the arithmetic, which is where the defect lives.

Read the second row twice. **Doubling the step count does not buy one intelligible syllable, and it
costs 108 seconds.** The prompt buys all of it and costs 0.55 s. Every latency knob in
[`G7.md`](../minimax_h3_g7/G7.md) is worth less than this one text change.

---

## 1. Why raw text is out of distribution

SGLang sends the prompt to the text tower **verbatim**:

```
runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py:109
    def minimax_h3_text_only_ids(tokenizer, prompt):
        """t2va presentation: verbatim prompt, no special tokens."""
```

`grep -r apply_chat_template runtime/.../model_specific_stages/minimax_h3/` returns **nothing**. No
chat template, no system prompt, no rewriter, no length normalisation. The string the client puts in
`"prompt"` is the string Qwen3-VL tokenizes.

MiniMax's product does not do that. The model card describes a separate hosted system:

* `README.md:80` — **H3-Context-IR**, "a dedicated multi-stage system" of models that "deeply
  understand and refine the input multimodal instructions, then convert them into … the Context
  Intermediate Representation", with the flat statement **"H3-Context-IR is critical to the quality
  of the final output."**
* `README.md:94` — it is **not** part of the open-source release. An API reproduces it; everyone else
  is pointed at the *Prompting Guidance* to build their own.

The scale gap is visible in MiniMax's own API examples, which report `prompt_tokens` of **5650 /
12800 / 33323**. The customer's t2va case is **62 characters**. So:

> **Whatever a caller sends to SGLang IS the IR.** A one-sentence Chinese instruction is not a short
> prompt — it is an input from a distribution the DiT never saw, and the model fills the gap by
> inventing. Prosodically plausible babble is exactly what "no phonemes were supplied" looks like.

**Do not confuse this with `refine_prompt_embeds`.** `_precompute_refined_prompt_embeds`
(`minimax_h3/stages/denoising.py:385-422`, key `_minimax_h3_refined_prompt_embeds`) is an in-model
*embedding* refiner that runs on whatever text you already sent. It is not a prompt IR and it cannot
recover content that was never in the text.

## 2. The format is documented even though the system is not

Copied out of the model repo's `docs/` into [`docs/h3official/`](docs/h3official/):

* `VIDEO_PROMPT_WRITING_GUIDE_base_en.md` — T2VA / I2VA / FL2VA / L2VA
* `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` — the full-reference (ref2va) format

### 2.1 T2VA / I2VA / FL2VA / L2VA — three fields

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

Blank-line separated. The keyframe tasks prepend one fixed instruction line before them, e.g. for
L2VA — the shape that matters below —

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

### 2.2 ref2va — **six** sections, not three

```text
subject_definitions / summary / retention_analysis / detailed_description /
overall_soundscape / non_diegetic_music
```

with every referenced asset given an explicit label (`<Subject N>` for reusable visible content,
`<Picture N>` / `<Video N>` / `<Audio N>` for concrete anchors), a bracketed task-type prefix on the
summary (`[reference generation]`, `[video editing + audio reuse]`, …), a **fixed relationship marker
per label** in `retention_analysis` (`fully_preserved` / `partially_preserved` / `attribute_transfer`
/ `weak_reference`; for audio `fully_copy` / `partially_copy` / `reference` / `weak_reference`), and a
`detailed_description` of normally **350–500 English words**.

### 2.3 The four rules the customer's prompts broke

1. **The three (or six) named fields are mandatory.** The audio branch is conditioned by the *same
   text* as the video branch. A prompt with no `overall_soundscape` and no `non_diegetic_music` field
   leaves both audio fields to be invented.
2. **Spoken lines go inside `<d>` with a language tag, and the speaker gets a stable ID** (base guide
   §4.4): `... (S1) says: <d>[Chinese] 今天的豆子是云南的，牛奶用两升装的。</d>`. Inside `<d>` goes
   *only* the language tag and the verbatim line; delivery, timbre and speaking rate go **outside**.
   A clause like "the subject is explaining the problem" asks for speech and supplies no words —
   hence the babble.
3. **The body is English.** Only dialogue, lyrics and on-screen text keep the source language, and
   on-screen text goes in English double quotes, verbatim, never translated (§4.5: `A red neon sign
   reading "营业中"`). Every worked example in both guides is English prose with foreign-language
   content quoted. An all-Chinese description is not the shape the IR emits.
4. **Cuts are explicit and timed** (§4.2): `[Shot 1]` carries no timestamp, later shots start
   `[Shot 2] At 00:03.500, the camera cuts to …` with strictly increasing times inside the duration;
   camera motion is written as motion type + amplitude + speed inside the shot (§4.3, `pushes in with
   small amplitude at slow speed`). The original ref2va line asks for 中景背影 → 推进手部特写 → 定格
   with no markers and no times, so the model guessed where the cuts were inside 5.04 s. **Multi-shot
   is supported in one generation — it just has to be marked.**

## 3. The rewrite, and what it fixed

The two cases were rewritten into the documented format and run at the same seed, resolution, frame
count and step count as the raw originals. Only the prompt text differs. Neither file is in the repo;
[`case/demo_ir.txt`](case/demo_ir.txt) and [`case/demo_raw.txt`](case/demo_raw.txt) are a synthetic
pair in exactly that before/after relationship, so the commands in §5 still run. `sglang_case.py` decodes `\n` escapes so a multi-field prompt still fits the
one-case-per-line file (`parse()`, after the ref2va image token is split off).

### 3.1 t2va — the audio

Transcribed with faster-whisper `small`, `language="zh"`, on the pod. The table at the top of this
file is the whole result: **the raw prompt is unintelligible at 25 *and* 50 steps; the IR prompt is
the scripted line at 25 *and* 50 steps.** The failure was 100 % prompt-layer. The t2va *visual* was
already acceptable in the original render, which matches the customer's report — they complained
about the audio only.

The mechanism is not subtle: put the actual 台词 in `<d>[Chinese] …</d>` and the model has phonemes to
render. Leave it out and it has a prosody target and no content.

### 3.1.1 `2x` is read as "rx": no token inside `<d>` may mix scripts

Once the line was intelligible the customer heard the one remaining defect: **`2x` comes out as "rx".**
faster-whisper agrees — it transcribes that syllable as `rx` while getting every other word right — so
what is rendered really is one unclear syllable in front of the letter.

Neither guide has a rule about this. `grep -iE 'numeral|digit|number|formula|equation|math'` over both
files finds nothing, so how to spell `2x` for speech is not specified anywhere and had to be measured.
The ablation was four arms differing **only** in the bytes between `<d>` and `</d>` — one server
process, fp8 TP=4 × U=2, 768p / 121 f / seed 42, 25 steps, and the control arm is byte-identical to
the IR prompt's line so the defect is *reproduced* rather than reconstructed. Driver:
[`scripts/aud.sh`](scripts/aud.sh), whose case file is untracked along with the rest of the prompts.

| arm | dialogue | inference | ASR (`small`, `language="zh"`) |
|---|---|---:|---|
| **mix** (control) | …**2x**等于**4**，所以x等于**2** | 106.08 s | `…减3,`**`rx`**`等于4,所以x等于2` |
| **cn** | …**二x**等于四，所以x等于二 | 106.74 s | `…减3,`**`2x`**`等于4,所以x等于2` ✅ |
| **aike** | …**二艾克斯**等于四 | 106.66 s | `…两边同时减3`**`2x`**`等于4 所以x等于2` ✅ |
| **short** | …**艾克斯**等于二 (12 syllables) | 106.54 s | `…减3`**`X`**`等于2` ✅ |

Whisper normalises `二`→`2` and `艾克斯`→`x` on output, so the transcripts cannot show *which* spelling
was spoken — but that is not what they are being asked. Same model, same settings, same beam: the
control produces `rx` and the other three produce a clean `x`, so the difference is real and it is
caused by the 1–3 characters that changed.

**The digit is not the problem; the digit-letter juncture is.** Read the control row again — in the very
same sentence, `等于4` and `等于2` transcribe correctly. Bare Arabic digits are read fine. Only `2x`
breaks, and it breaks because it is not a word in any script: two writing systems fused into one token
with no separator, i.e. a *formula*, and a formula has no grapheme→phoneme path. `二x` fixes it with one
character changed while keeping the algebraic letter a letter.

So the rule, which belongs in the compiler's validator (§4.1):

> **Inside `<d>…</d>`, no token may mix scripts, and every mathematical expression must be written the
> way a person would say it out loud.** Latin letters used as variables are fine on their own — `x`
> read as a letter is in distribution. What is out of distribution is a digit welded to a letter.
> Expand `2x` → `二x` (or `二艾克斯`), `x²` → `x的平方`, `3/4` → `四分之三`, `y=kx+b` → `y等于kx加b`.

Two secondary observations, both weaker than the main result and recorded as such:

* **`二x` beats `二艾克斯`.** Both read cleanly, but the `aike` transcript loses its comma boundaries
  (`…两边同时减32x等于4` — the words run together) while `cn` keeps them. `aike` is 24 syllables against `cn`'s 20 in the same
  ~4.3 s of speech that fits a 5.04 s clip, i.e. ~5.6 vs ~4.7 syllables/second. Prefer the shorter
  spelling when both are speakable.
* **Budget the syllables.** `short` (12 syllables) is the most relaxed read of the four. There is no
  measurement here isolating rate from spelling, but the arithmetic is worth stating for the compiler:
  a 5.04 s clip has room for roughly 20 Mandarin syllables at an unhurried pace, and a
  request whose script is longer than that wants 241 frames, not a faster delivery.

**And the visual did not move.** All four arms hold the written equation pre-existing and unchanged at
0.2 / 2.5 / 4.9 s, so the dialogue rewrite is free of visual side effects — which is what makes it safe to iterate
on the `<d>` content alone.

### 3.1.2 The static equation was only half of it — the pose is the other half

The customer's first complaint was that the equation should already be on the board but the render
shows the subject writing it. The prompt they were given genuinely does say that the writing is
*already on the board*, and the file they watched was the **raw** prompt, where the subject chalks the
equation stroke by stroke across the whole clip. **The IR prompt already fixed the literal defect:**
the equation is on the board at frame 0 and untouched at frame 120, in every arm of the grid. `Chalk
handwriting reading "2x+3=7" is already on the board beside the subject` plus `The camera holds a
static shot` is what does it.

But the audio grid's montage showed why that is not yet a *pass*: in all four arms the subject holds a
piece of chalk **pressed against the board**, arm raised, for the entire take. Nothing is being written
and yet it reads as writing. That was my own prompt's fault — `taps the chalk twice on the board just
beneath the equation` asks for exactly that pose. The request was for something to be *explained*, and
the gesture for explaining something already written is *pointing at* the board, not touching it.

Same for the framing: the original case asked for a medium shot (中景), the IR prompt said `a static
medium shot`, and what came back was a medium **close-up**. Shot-size nouns turn out to be weak
instructions.

Three arms tested both fixes, dialogue held byte-identical to `@cn` (driver
[`scripts/v2.sh`](scripts/v2.sh), same config, 25 steps; the case file is untracked with the rest of
the prompts, and the contact sheet showed a real person so it is not in the repo either):

| arm | change | inference | result |
|---|---|---:|---|
| **cn** (control) | — | 106.74 s | chalk against the board in every frame; medium close-up |
| **pose** | hands start empty at their sides; `points at the equation with an open palm, never touching the board`; `No writing takes place and the handwriting … is unchanged from the first frame to the last` | 106.20 s | **writing pose gone.** Open-palm point, chalk absent. Framing unchanged |
| **wide** | `@pose` + `the camera is about three metres away, so the frame cuts at the waist`, `space above the head`, `the full width of the board panel is visible`, `small in the frame and legible in full` | 107.02 s | **the medium shot the case asked for.** The board's top frame is in shot, headroom above the head, the equation small and fully legible |

The two fixes are **independent** — `@pose` did not widen the shot on its own, so the framing sentences
are load-bearing and both stay in the production prompt. And the audio was unperturbed in both (the
same clean transcript as the `cn` arm), which is exactly why the dialogue was held byte-identical:
it demonstrates that prose edits to the description do not disturb the speech, so the two halves of the
prompt can be iterated separately.

Two transferable rules for the compiler, both of the same kind:

> **Negate the default explicitly.** `already on the board` is not enough on its own, because the
> action verbs elsewhere in the sentence reintroduce it. Add the negative clause: *no writing takes
> place, the handwriting is unchanged from the first frame to the last.* Same shape as the guide's own
> `lips remain closed` requirement for off-screen voiceover (§4.4) — the model needs to be told what
> does **not** happen.
>
> **Describe the frame, not the shot size.** `medium shot` bought a close-up; `the frame cuts at the
> waist / space above the head / the full board panel is visible / the equation is small in the
> frame` bought the medium shot. Every shot-size noun in a generated prompt should be accompanied by
> what is in frame at its edges.

**Cost of all of this: zero.** 106.20 / 107.02 s against the 106.08 s control — inside noise, on a
prompt that grew from 1036 to 1407 characters.

### 3.2 ref2va — the "no sense of stroke-by-stroke writing" complaint

The comparison was the six-section IR prompt at 0.5 / 2.5 / 4.9 s against the customer's original
prompt at 1.0 / 3.0 / 4.9 s. Same weights (fp8, TP=4×Ulysses=2), same seed, same 25 steps. The contact
sheet is not in the repo: both rows show the customer's reference subject.

**The raw prompt.** The model took the requested push-in to a hand close-up literally and immediately:
by 1.0 s the framing is already a macro of the board, by 3.0 s the phrase is essentially complete, and
the last frame is a hand and five characters with **no person in shot at all**. There is no
stroke-by-stroke reading because there is no shot in which strokes are the visible event — the writing
happened during the push-in, off the readable part of the frame. The reference image's identity is
also mostly wasted: the subject is out of frame for the second half.

**The IR prompt.** A static medium shot for the whole take. The board starts with the first three of
the five characters already on it, the fourth appears through the first second, the fifth lands after
it, and at 4.9 s the subject has stepped back, the complete phrase is unobstructed, and they have
turned to camera with the closed-lip smile the prompt asked for. Identity, clothing, the board's
frame, the chalk tray, the eraser and a notice sheet on the wall are all preserved from the reference
image. That is the *shot* the customer wanted.

**Correction to an earlier version of this file, which claimed the fourth character "lands stroke by
stroke".** It does not, and neither does the fifth. Sampling every frame instead of every 2.5 s shows
the characters do not accumulate strokes at all — see §3.4. The IR prompt fixed the framing, the camera, the pacing and the
reference adherence; it did not make the handwriting behave like handwriting, and no prompt will.

Cost of the six-section prompt: **177.18 s inference vs the baseline's 174.43 s** — 1.6 %, i.e. free.
A 3337-character prompt is 0 % of a video-diffusion bill. **Prompt length is not a latency knob** — and
it stays free in the fast config: the same IR prompt through the 8-step LoRA arm is **63.74 s** against
63.26 s on the 126-character original, and it produces the same shot progression and the same closing
frame as the 25-step render. So the prompt fix and the 2.8× speed-up in [`G7.md`](../minimax_h3_g7/G7.md) compose;
neither costs the other anything.

Three things in the rewrite did the work, in rough order of importance:

* **A static shot, declared.** `The camera holds a static shot for the entire duration.` A cut or a
  push-in inside 5 s costs the model the only frames in which strokes are legible.
* **An action that fits 5.04 s.** Five chalk characters stroke by stroke is not a five-second action.
  The board is pre-loaded with the first three and the shot covers the last two. Guide §4.1: every detail must
  correspond to something actually visible; an action that cannot finish makes the model compress
  time, and compressed time is the smear.
* **The reference told what it is *for*.** `subject_definitions` + `retention_analysis` name
  `<Subject 1>` (the person) and `<Subject 2>` (the room) and mark both `fully_preserved`. The
  customer's original 200-character line never mentions the image at all, so nothing in the text told
  the model what the reference image was for and adherence was left to the image tower alone.

### 3.3 One thing no prompt can fix: ref2va is the wrong task for this case

The reference image is the **finished** board — the whole phrase already written, the subject already
turned to camera smiling. It is the **end state** of the action the prompt asks for. ref2va means *keep this
subject and this scene*; it cannot pin the ending.

A request whose whole point is to end on the completed text wants the image as a **last keyframe** — task `fl2va`, which
admits 1–2 keyframes with an explicit `frame_index` (`minimax_h3/task_profiles.py:163,
requires_frame_index=True`) and whose instruction line is fixed by the base guide as the L2VA form in
§2.1 above. That is the next experiment and it is a task change, not a prompt change.

### 3.4 The one defect that is NOT the prompt: stroke incoherence

The customer's next report, after the framing was fixed, was that the strokes were wrong in a very
specific way: it looks as though a stroke on the *left* of the character is being written, and yet a
stroke on the *right* is what appears.

Correct, and this one is **not** a prompt problem. It is a representation limit, and it is worth being
precise about why, because the conclusion is "stop asking the model to write" rather than "write a
better prompt".

The evidence was the final character's region of `rir25` (25-step fp8, the IR prompt), cropped and
sampled every second frame across frames 44–110 — a nine-stroke character in two side-by-side
components. That contact sheet is not in the repo (it is a crop of a customer render), so the three
observations are recorded here in words; each is impossible for real handwriting:

1. **Ink appears where the chalk is not.** The chalk tip is at the top-left of the LEFT component while
   the entire RIGHT component is already fully formed. This is the customer's exact observation.
2. **Ink appears and then disappears.** A vertical stroke beside the left component persists for three
   sampled frames and is simply gone in the next. Chalk does not un-write.
3. **The intermediate glyphs are not prefixes of the target.** The first thing to appear is a different,
   simpler character that is not any initial subsequence of the target's stroke order. It is a
   plausible-looking chalk shape, not a partially-written character.

The hand never traces a stroke path either — it hovers in roughly the right area while the glyph
resolves behind and around it. **The model is not simulating writing. It is interpolating between "no
glyph here" and "this glyph here", and synthesising plausible chalk texture at every frame
independently.** Note that the model was *given* the correct final glyph — the reference image is the
finished board — so this is not ignorance of the character. It is that nothing constrains the intermediate
states.

**Why it cannot be prompted away.** From the checkpoint's own configs:

```
Ref2VA/video_vae/source/config.json    vae_ratio: 16,  vae_ratio_t: 4
                                       space_down [2,2,2,2,1,1], time_down [1,2,2,1,1,1]
Ref2VA/video_vae/config.json           latent_channels: 24
```

So at 1344×768 the DiT works on an **84×48** spatial grid, and 121 frames become **31** latent frames.
Measured off the render, one character occupies about **117×167 px ≈ 7×10 latent cells**, and a chalk stroke is
about **8–10 px wide ≈ 0.5 of one latent cell**.

> **A single stroke is smaller than the smallest thing the DiT can address.** It has no token for
> "stroke"; it has ~70 cells that say "chalk-glyph, roughly this dense, roughly this shape". The actual
> stroke texture is invented by the VAE decoder, per frame. Frame-to-frame the invention differs, and
> that difference *is* the artifact.

Temporally it is just as tight: that character is written across frames 44–110, i.e. ~16 latent frames
for 9 strokes — under two latent frames per stroke. And the schedule is non-causal: all 31 latent frames are
denoised jointly, so there is no mechanism that enforces "ink at frame *t* ⊇ ink at frame *t−1*".
Temporal attention is a smoothness prior, not an ordering constraint. Monotone accumulation is a hard
constraint the architecture cannot represent.

Two confirmations that it is not a knob:

* **Neither guide has any stroke vocabulary.** `grep -i stroke` over both files returns exactly one hit
  — `strokes the dog's thick white fur`. There is nothing to write even if the model could obey it.
* **Step count does not touch it.** `rir8` (8-step LoRA) shows the same sequence of plausible-but-wrong
  intermediate glyphs with the same class of error. Same as the audio: more steps buy nothing here.

**What actually helps, in descending order of effect:**

1. **Don't generate the handwriting.** Generate the person — which H3 is genuinely good at: identity,
   hands, lip sync, a held static frame — with the board **empty or already written**, and composite the
   write-on as a deterministic layer (SVG stroke-path animation, a write-on effect, or real footage).
   Correct stroke order, correct glyphs, repeatable, and re-editable without a re-render. A static camera
   makes the composite trivial because there is nothing to track — which is exactly the shape the t2va
   prompt in §3.1.2 was validated at.
2. **Give the glyph more latent cells.** Every mitigation is really this one: raise the short edge, put
   fewer characters in frame, or frame tighter on the board. A character at 7×10 cells is hopeless; at 3×
   the linear size it is ~21×30 with strokes ~1.5 cells wide, which is at least representable. The cost of
   framing tighter is the person, which is what the customer's original raw prompt accidentally traded
   away (§3.2) — so the honest version of this lever is **resolution**, not framing.
3. **Ask for fewer strokes.** Nine strokes in three components is close to the worst case. One
   simple character, or "adds the final stroke to an almost-complete phrase", is one stroke's worth
   of coherence to get right instead of nine.
4. **More time per stroke.** 241 frames instead of 121 doubles the latent frames per stroke. Helps the
   temporal half; does nothing for the spatial half, which is the binding one.

## 4. What to build: a prompt compiler, not a prompt

H3-Context-IR is closed, but it is a **text-to-text** system with a documented output format and a
validator-shaped spec. That is the cheapest possible thing to reproduce.

### 4.1 An LLM compiler with a hard validator

Take the user's request (Chinese, one sentence, plus whatever assets) → emit the exact format.

* **Prompt the compiler with both guides verbatim** plus 3–5 gold few-shot pairs per task. The guides
  are 15.8 KB and 23.6 KB; that is a normal system prompt, not a fine-tune.
* **Then validate mechanically and retry on failure.** This is the part that makes it a compiler
  instead of a wish. All of these are regex-checkable:
  - the required fields are present, in order, blank-line separated, and *only* those fields
    — **corrected while building `ir/`:** the separator is a single `\n` in all 13 of NVlabs'
    validated showcase prompts, so a blank-line rule rejects known-good output. Accept either;
    reject only a field name that is not at the start of a line
  - `[Shot 1]` has no timestamp; every later `[Shot N] At MM:SS.mmm` is strictly increasing and
    inside `duration_seconds`
  - `<d>` tags are balanced, each opens with a `[Language]` tag, and every `<d>` is preceded by a
    speaker ID; speaker IDs are consistent across shots and dense from `(S1)`
    — **corrected while building `ir/`:** "preceded by a speaker ID" was read off the guide's short
    examples and is wrong. The validated corpus binds `(S1)` where the speaker is *introduced*,
    with its pitch/timbre/rate, and attributes later lines by name (`the tea master says:`);
    requiring adjacency rejects all 13. Enforce instead: IDs exist, dense from `S1`, never inside
    `<d>`, the first one precedes the first `<d>`, and each `<d>` follows an attribution ending in
    `:` or `,` (true of all 38 dialogue blocks in the corpus)
  - every voiceover `<d>` is followed by an explicit "lips remain closed" clause (§4.4)
    — **corrected while building `ir/`:** there is a second legal form. NVlabs' product-film case
    has nobody on camera and discharges it with `while no speaking face appears on screen`. The
    real obligation is "assert that no visible mouth is speaking"
  - **no token inside `<d>` mixes scripts** — reject `\d[A-Za-z]` and `[A-Za-z]\d` adjacency, plus
    `^*/=²³` and the rest of the math operators, and expand them to spoken form (§3.1.1). This one is
    worth failing hard on: it is the difference between `2x` and `rx`
  - **`<d>` syllable count fits the duration** — roughly 20 Mandarin syllables per 5.04 s clip at an
    unhurried pace; over budget means raise `duration_seconds`, not talk faster (§3.1.1)
  - on-screen text appears in English double quotes and is byte-identical to the source
  - ref2va: every `<Subject N>` / `<Picture N>` defined in `subject_definitions` has exactly one
    marker line in `retention_analysis`, and the marker is one of the four legal strings
  - `detailed_description` word count in [350, 500]; `overall_soundscape` 1–4 sentences;
    `non_diegetic_music` 1–3 sentences or `N/A`
  - the body contains no CJK outside `<d>…</d>` and outside `"…"`
* **Fail closed.** A prompt that does not validate is not sent. This is worth 3 seconds of retry
  against a 105–180 s render.

### 4.2 Never let the model invent the 台词

For anything with a script — which is every education/marketing case — the business already has the
words. Put them verbatim in `<d>[Chinese] …</d>`. This alone is the difference between the two ASR
rows above. If the caller supplies no line, the compiler should **write one and return it for
approval**, not leave the field to the DiT.

### 4.3 For maximum speech robustness, drive the audio instead of describing it

`ref2va` accepts an **audio reference** as a first-class condition:

```
minimax_h3/task_profiles.py — ref2va condition_rules
    keyframe/image, reference/image (image.reference_preserve),
    reference/video (its soundtrack becomes the audio reference), reference/video_audio,
    reference/audio  (material_chain="audio", audio_tokenizer_encode=True)
  + duration_from_audio_reference=True, video_reference_supported=True
```

So: TTS the approved line, send it as `reference/audio`, and the model's job collapses from *invent
intelligible Mandarin* to *align lips and timing to this track*. `duration_from_audio_reference=True`
means the clip length follows the track, which is also how you stop cramming a 7-second sentence into
5.04 s. **Note that `t2va` has `condition_rules=()` — no conditions at all** — so this route requires
`ref2va` or a keyframe task, not t2va.

### 4.4 Multi-shot, and stop cramming

Two legal routes, and they are not equivalent:

* **One generation, explicit timed `[Shot N]` markers.** Cheapest. Correct when the shots share a
  scene and the whole thing fits the duration.
* **Chained keyframe requests.** Render shot 1, take its last frame as shot 2's first keyframe
  (`fl2va`). Correct when the shots are genuinely different setups, and the only route that gives
  per-shot control.

And in both cases: **prefer 10 s (241 f) over three shots in 5 s.** Frames must be ≡ 1 mod 8 and
duration ∈ [4, 15] s, so the real choices are 121 f / 241 f / 345 f. The customer's ref2va case is a
three-beat request; at 5.04 s each beat gets 1.7 s and the model compresses. 10 s doubles the bill and
is still cheaper than a re-render.

### 4.5 The order to do it in

1. **`<d>` tags + the three/six named fields.** Days of work, and it is the whole audio fix.
2. **The validator**, starting with the script-mixing rule in §3.1.1 and the two negation/framing rules
   in §3.1.2. Turns 1 into something that survives contact with real users, and the three cheapest
   checks in it are the three defects the customer actually reported.
3. **Task routing** — end-state image ⇒ `fl2va` last keyframe, not `ref2va`; scripted speech ⇒
   consider `reference/audio`; duration from the length of the action, not from a default.
4. **The rest of the guides** — camera-motion vocabulary, `<scenetrans>` / `<cutoff>`, styles. Polish.

**Steps 1, 2 and 4 are built: [`ir/`](ir/README.md).** `ir/h3ir/validate.py` is the validator, 36
rules, calibrated so that it accepts NVlabs' 13 validated showcase prompts *and* a ref2va prompt in
this repo's own measured format with zero errors, while rejecting a t2va arm carrying a digit-letter
token on exactly `E030-D-MIXED-SCRIPT-TOKEN` (`case/demo_ir.txt` is that pair). `ir/h3ir/compile.py` is the Bedrock loop, fail closed. Measured:
**Opus 5 is 4/4 first-pass validator-clean at 19.5 s mean; GPT-6 is 3/4, 4/4 after one repair, at
55.5 s.** Step 3's routing predicate exists (`route_task()`) but nothing yet chooses `duration` from
the length of the requested action, and **no compiled prompt has been through the GPU yet** — that
render is what would turn "in-distribution" into "better".

## 5. Reproducing

```bash
# on the pod, after a t2va server is up (see ../minimax_h3_g7/G7.md for the serve line)
python /data/h3/sglang_case.py case=/data/h3/demo_ir.txt  task=t2va tag=ir25   768:25:121
python /data/h3/sglang_case.py case=/data/h3/demo_raw.txt task=t2va tag=orig25 768:25:121

# ASR, installed into its own tree so it cannot shadow the server's packages
pip install -q --target /data/h3/py faster-whisper
PYTHONPATH=/data/h3/py python /data/h3/asr.py '/data/h3/pull/case/t2va_*/*.mp4'
```

`asr.py` is 8 lines around `WhisperModel("small", device="cpu", compute_type="int8")` with
`language="zh"`. CPU is fine — the clips are 5 seconds.

Those two commands run the synthetic pair, not the prompts the numbers above came from, so they will
reproduce the *shape* of the result — babble versus the scripted line — and not the transcripts. The
measured prompts are in the working copy but not in the repo; `scripts/sync.sh` ships whatever is in
`case/` to the pod, so if you have them, substituting their filenames is the only change needed.

---

**Bottom line.** The IR is not optional and it is not in the release, but it is a
text-to-text layer with a published output format, so it is ours to build and it is the highest-return
work available: it fixed intelligible speech outright and it fixed the ref2va shot, at zero latency
cost, while doubling the step count fixed neither and cost 2×.
