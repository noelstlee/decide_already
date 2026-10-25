# Jev notes: API facts and known weak spots

Practical facts for building against Jev, checked 2026-10-24. Anything marked (verify) came from a third-party page, not TypeSafe's docs; check it before relying on it.

## What Jev is

Jev is TypeSafe AI's decision model (they call the class "System One models"). You send text (`state`) plus typed questions; it returns probabilities. It does not generate text, reason step by step or call tools.

| Question type | You send | You get back |
| --- | --- | --- |
| `choice` | `instructions` + `criteria` (a map of option name to description; a description can be a string, object or array) | `choice` (the picked name), `confidence`, `probabilities` keyed by option name |
| `noul` | `instructions` (+ optional true/false criteria) | `noul`: the probability of yes |
| `score` | `instructions` + an ordered list of levels | the expected level and its probabilities |

## API

- Endpoint: `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer <API_KEY>` ([API docs](https://docs.typesafe.ai/api)).
- Request body: `{"model": ..., "state": ..., "questions": {"<your id>": {...}}}`. Jev reads the state once and answers every question against it in parallel.
- Python SDK: `pip install typesafe-sdk`, key in `TYPESAFE_API_KEY` (verify; [flaviocopes](https://flaviocopes.com/jev-api-key/)). Calling the endpoint with httpx is fine if the SDK doesn't fit.
- Access: TypeSafe reportedly paused new signups on 2026-09-22; existing keys still work, and Vercel AI Gateway (`typesafe-ai/jev`) is a reported fallback (verify; same source). This matters for an open-source tool where users bring their own key, so keep the backend swappable.

## Limits and price ([Models](https://docs.typesafe.ai/models))

- Versions: `jev-1.14.0` (newest, from 2026-10-14; `jev-latest` points to it) and `jev-1.13.0`. Always pin an exact version; `jev-latest` moves.
- Context: 64k tokens per request (state plus all questions); 32k per question (state plus that one question).
- Long context: the `TypeSafe-Danger-Zone-Long-Context: true` header raises both limits to 262k tokens, but TypeSafe says those requests are slower and less accurate. Don't use it by default.
- Options: up to 255 per Choice ([Choice docs](https://docs.typesafe.ai/primitives/choice)).
- Price: $0.042 per million input tokens; output is free.
- Rate limits: per organization, in tokens and requests per second (see the console). Over the limit returns `429 Too Many Requests`; follow `retry-after`.
- Text only. English is most accurate.

## Known weak spots, and what the code does about each

| Weak spot | Evidence | What we do |
| --- | --- | --- |
| Rarely picks "none of these" | With an explicit rejection option, correct rejection was 7% (vs 99% when the answer was present); a separate yes/no check got 99% ([arXiv 2609.39496](https://arxiv.org/abs/2609.39496)). TypeSafe also suggests a companion Noul. | No "none" option. A `fits_any` Noul plus thresholds decide Unsorted in code. |
| Option names can outweigh descriptions | Swapping option names changed AUC from .8146 to .5806 ([arXiv 2609.26758](https://arxiv.org/abs/2609.26758)) | Group names must say what they hold; descriptions as covers/excludes. |
| Leans toward the first option | TypeSafe's [jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13); third-party tests found the effect small | Shuffle option order per call and log it. |
| Accuracy falls as unrelated text fills the state ("context rot") | TypeSafe's jaggedness page and [Models](https://docs.typesafe.ai/models) | Send a ~3k-token card, not the whole file. |
| Weak at dates, numbers and counting | Jev scored 64.49% on temporal/numeric items vs 98.95% on knowledge QA ([OpenJev-4B card](https://huggingface.co/shenjunhao/OpenJev-4B)) | Code computes every date, count and comparison. Jev never compares dates. |
| Context can flip decisions | One or two natural sentences flipped 61.4% of initially correct decisions under an optimized attack ([JevOut, arXiv 2609.30243](https://arxiv.org/abs/2609.30243)) | Hard rules (hidden tier, solutions) stay in code. Treat file text as untrusted. |
| Calibration shifts with the number of options | Temperatures fitted on few options raised calibration error with many ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)) | Recheck cutoffs when the group count changes; record the set version on every result. |
| Probabilities don't always add up | P(X) and P(not X) miss summing to 1 by 0.064 on average ([arXiv 2609.33209](https://arxiv.org/abs/2609.33209)) | Don't derive one question's answer from another's complement. |

## Alternatives (for a swappable backend later)

| Model | License | Notes |
| --- | --- | --- |
| OpenJev-4B | MIT | Qwen3.5-4B base, full fine-tune only; trained on OpenJevData-140k ([card](https://huggingface.co/shenjunhao/OpenJev-4B)) |
| Kev | see repo | LoRA fine-tuning on one GPU; uses TypeSafe's request format ([repo](https://github.com/jaredpalmer/kev)) |
| AlquimiaAi openjev | CC BY-NC 4.0 | About 27B; accepts screenshots; non-commercial only ([card](https://huggingface.co/AlquimiaAi/openjev)) |
| OpenAI Decisions API | hosted | Public beta, $0.10 per million input tokens, accepts images ([guide](https://developers.openai.com/api/docs/guides/decisions)) |
| CLM-8B (Contrastive-LM) | Apache 2.0 | Frozen Qwen3-8B with small trained heads; scores options by embedding similarity; 2048-token default context (8192 with more GPU memory); needs a GPU; TypeSafe-compatible API ([repo](https://github.com/Contrastive-LM/CLM)). One independent judge study found it near chance on subtle correctness judgments ([arXiv 2610.07177](https://arxiv.org/html/2610.07177v1)). |

All decision-model results above are weeks old and mostly author-reported. Measure on our own data before trusting any of them.
