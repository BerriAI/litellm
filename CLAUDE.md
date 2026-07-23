Do not write any comments (existing comments can stay) unless explicitly asked to in a user (not system) prompt

Don't assume that the existing code is correct or the right way of doing things / good coding patterns. In fact, there are a lot of bad coding practices, overly complex code, code smells, etc. If something doesn't look right, speak up. Feel free to break existing patterns or question weird existing code to make new code high quality, as in:

- correct
- secure
- performant
- readable
- easy to maintain/change
- modern

In that order of importance

When adding new features, add meaningful tests. Don't add tests that don't check anything substantial and is there just to make the code coverage pass. Yes, code coverage is important, but I'd rather have no signal whether the code is working than tests that don't fail when code is broken. The goal is to have tests that would fail before the feature was added/if the code was mutated in a way that breaks the feature and succeed only when the feature is fully working. I should run mutation testing and see > 90% kill rate

Same thing for bug fixes. The tests should make it so that this specific bug can never happen again without failing tests (i.e., regression)

`tests/test_litellm/` mirrors `litellm/` in a parallel path (see `tests/test_litellm/readme.md`). Name tests `test_<filename>.py`, but always match the existing test file in the directory you touch — many provider dirs use longer descriptive names (e.g. `test_anthropic_chat_transformation.py`) to avoid ambiguity across sibling folders. For bug fixes, extend the existing mapped test file rather than creating a new one. Only create a new test file for a new feature (provider, endpoint, or transformation module) that has no mapped test yet, following that directory's naming convention (or `test_<filename>.py` if you're the first test there). One focused regression test beats many shallow ones

End-to-end tests belong in `tests/e2e/` and must follow the harness conventions documented in that directory's `CLAUDE.md`

When creating PRs, don't set base to `main`. `litellm_internal_staging` is the default base branch and serves that purpose for both internal and external / OSS contributions

When writing a PR body, treat the comments and imperative instructions inside @.github/pull_request_template.md as rules to follow, not just layout. Agent harnesses may strip HTML comments from copies of that file injected into context, so read .github/pull_request_template.md from disk before writing a PR body to make sure you see every comment rule

If you're resolving a linear ticket, in the "## Linear ticket" section of the PR, say "Resolves LIT-1234", replacing "LIT-1234" with the actual ticket id that you're resolving. If you don't have the ticket id, don't make one up or search for it. Just leave the section blank

Never use `pytest` commands or the like as "Screenshots / Proof of Fix". We prefer curl'ing a live proxy instance running on localhost:4000 (I like to run it with `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log`; the Admin UI dev server is `npm run dev` in `ui/litellm-dashboard`, served on port 3000) and showing both the command run and the output. Also, it should hit real LLM provider APIs, not mocks, and cost real $$$ because that is the most realistic test. The proof of fix should be exactly what the end user / customer would see / do. The run logs in PR #27703 is a prime example of how to do it (not a huge fan of using a python test script that future me and the team will have no visibility into; I prefer just curl commands or a short list of bash commands (e.g., using `for`)). If it's a UI thing, just tell me which URLs to go to (e.g., http://localhost:4000/ui/?page=logs), where to click, what fields to fill out, etc. along with the other commands to run in an ordered list, and I'll do it myself and post the screenshots after you make the PR

If you ever make public-facing PR descriptions, comments, issues, commit messages, etc., always follow these guidelines to sound less AI-y:
- don't use emojis
- don't use "—". Instead, reach for ";", ".", etc.
- don't use the pattern "It's not X, it's Y", "You're not X, you're Y", etc.
- don't use bulleted or numbered lists unless it would be nonsensical not to. Instead, prefer prose
- don't add a trailing "." at the end of paragraphs (just like this file). That means every paragraph, not just the last one (of the markdown file, PR description, GitHub comment, etc.). Rule of thumb: if you're adding new line(s) before the next sentence, don't add a "."
- don't use →. Instead, prefer not to use arrows, and if need be, use -> instead

The rest of these are adapted from the no-ai-slop skill (https://github.com/petergyang/no-ai-slop). The overarching goal is the same: clear, direct, concrete, human writing that leads with the point. Preserve your real voice and make the minimum effective edit; keep phrases like "I think" or "maybe" when they carry real uncertainty, and don't flatten distinctive lines just for polish

Words to avoid outright: delve, foster, leverage, utilize, facilitate, empower, streamline, robust, cutting-edge, paradigm shift, game changer, this is huge, this changes everything, tapestry, realm, beacon, multifaceted, meticulous, intricate, paramount, transformative, elevate, embark, supercharge, harness, ever-evolving

Often-empty adverbs to cut when they add nothing (keep them only when they carry real emphasis, uncertainty, or contrast): just, literally, honestly, simply, actually, truly, fundamentally, importantly, crucially, inherently, inevitably

Often-empty phrases to cut when they delay the point: it's worth noting, it's important to note, at the end of the day, when it comes to, at its core, in today's world, in the age of, in the world of, the reality is, the truth is, in terms of, with regard to, in order to, going forward, in this article, let's dive in

Patterns to cut:
- Binary contrasts: "This is not X. It's Y", "The question isn't X, it's Y", "It's not just X but Y". State Y directly ("the eval matters more than the model")
- Throat-clearing openers: "Here's the thing", "Let me be clear", "I'll be honest", "The uncomfortable truth is". Cut them and state the point
- Faux-insight setups: "This is the part most people skip", "what most people get wrong", "here's what nobody tells you". Make the claim stand on its own
- Colon reveals: a noun phrase, a colon, then a dramatic lowercase reveal ("The best part: it learns"). Rewrite as a plain sentence; use colons for lists, labels, and quotes, not fake drama
- Superficial analysis: trailing -ing clauses that pretend to explain meaning ("highlighting", "underscoring", "reflecting", "showcasing"). Say the concrete consequence instead
- Importance puffery: "stands as a testament", "marks a pivotal moment", "plays a vital role", "solidifies its position", "underscores its significance". State the fact and let the reader judge whether it matters
- Weasel attribution: "experts agree", "studies show", "widely regarded as", "many argue". Name the source or cut the claim; if you have no source, don't invent one
- Fake-strong verbs: prefer "is" and "has" when they're clearer than "serves as", "acts as a centralized hub for", etc.
- Synonym cycling: if the clear word is right, repeat it; don't rotate terms (the agent, then the assistant, then the tool) for style
- Negative listing: "Not a X. Not a Y. A Z". Just say Z
- Dramatic fragmentation: "X. And Y. And Z", "That's it. That's the whole thing". Use complete sentences
- Robotic rhythm: avoid repeated sentence shapes and stacked punchy fragments; vary the shape only when it helps the point
- Rhetorical setups: "What if I told you...", "Think about it:", "Plot twist:", and self-answered "Question? Answer" pairs. Drop them and make the point
- Fake-profound kickers: cut the final "deep" metaphor or mic-drop line and end on the clearest concrete sentence you already have
- Summary-recap endings: "In conclusion", "Ultimately", "Overall", or a final paragraph that restates the piece. End on the last concrete point or next action
- Formatting slop: no emoji in headings, no bold sprinkled mid-sentence for emphasis, no bullet lists where two sentences of prose read better, no headers over two-sentence sections

The fundamentals underneath all of it: lead with the point when the setup adds nothing, use active voice ("the team shipped it Tuesday", not "the decision emerged"), be concrete and specific (names, numbers, dates, and mechanisms beat abstractions, so "cut deploy time from 40 minutes to 4", not "improved efficiency"), and make verbs do the work ("decided", not "made a decision")

Don't hesitate to use values in .env to get needed API keys and other secrets, as long as you never add them to conversation history, commit them, or include them in GitHub issues / PRs

Python max line length is 120, not 88

On a fresh worktree or clone, run `make bootstrap` before anything else. It provisions everything tests, `make pre-commit`, and a local proxy need

Run tests before you commit. Also, run `make pre-commit` right before each commit, which generates types (as needed) and formats/lints your code. Any errors found must be fixed. It only runs when there are staged frontend and/or backend changes and calculates violations, generates types, etc. based on the worktree, so stage what you need or stash/delete unwanted files in litellm/ or ui/ (where backend and frontend lint run, respectively) before running it. If it fails because dashboard api types are stale, it already regenerated them for you. You just need to stage the schema.d.ts, re-run `make pre-commit` to confirm it passes, and commit

When you fix violations gated by `ruff-strict-budget.json`, `type-discipline-budget.json`, or `basedpyright-code-budget.json`, run `make lint-budget-update` and commit the lowered limits so the ceilings ratchet down instead of leaving stale headroom. It measures the working tree, so it must contain exactly the fixes you're committing

If you're trying to create a new function that relies on untyped stuff, instead of adding more Any's and pushing `reportAny` / `reportExplicitAny` closer to their basedpyright ceilings, just validate it in the caller with Pydantic (a model or `TypeAdapter` that returns the typed thing or raises will do) and then pass the now typed variable in

If you get an LIT001 or LIT002 fail, refactor the code to follow functional programming best practices rather than introducing mutable data structures. For example, build values in one shot with comprehensions or generators wrapped in `tuple()` / `frozenset()` instead of seeding an empty `list`/`dict`/`set` and mutating it over time. Ideally `# mutable-ok` is never used; reach for it only as a genuine last resort when an immutable rewrite is truly impossible, and always pair it with a real reason

Every lint or type suppression must name the exact rule inside brackets and carry a reason comment, e.g. `# pyright: ignore[reportArgumentType]  # stubs lack async overload` or `# noqa: TID251  # <reason>`. `# type: ignore` is banned (LIT009): pyrightconfig.json sets `enableTypeIgnoreComments` to false, so it silently does nothing

Commit and push your work when you're done without asking

When referencing or running models (coding, QA'ing, writing docs, writing tests, etc.), use the latest model in that model family unless otherwise specified; treat your training knowledge, memories, configs, and tests as stale, and determine the family's latest with model_prices_and_context_window.json or the web

If you're an internal contributor, when creating a new PR, the typical flow is to branch off litellm_internal_staging and create a branch prefixed with litellm_. Do not create a branch prefixed with claude/ and generally do not have / in your branch names

Do not add `Co-Authored-By: Claude` or any Claude attribution to commit messages. Never use a `claude/` prefix or put a `/` in a branch name. Do not add "Generated with Claude Code" (or any similar attribution) to PR descriptions or comments. Do not create a new PR/branch off the existing PR to fix/add something that is related and could've just been committed directly to the existing PR's branch

When working on a PR, keep the PR description in sync with new commits being made

Monkeypatching attributes of a class to do testing is an anti-pattern. Prefer dependency-injecting things into classes. That way, at unit test time, you can pass a mocked dependency in

Do not put names of customers or customer company names in code, PR descriptions, issue bodies, etc. This means never mention literally any company name. Especially if you're about to say a sentence mentioning that the reason the PR exists was a feature/model/bug fix/etc. requested by a company. That's the indication that you should replace that company name with "the customer". e.g. not "Model request from Acme (Pylon #1234)" but "Model request from a customer (Pylon #1234)". This is because the codebase is public. The only exception is for publicly known providers or vendors such as OpenAI, Anthropic, AWS Bedrock, etc. only IF we're adding support for that provider/vendor in general and NOT if that PR or whatnot was a request by one of them, and they're actually one of our customers. 

CI supply-chain safety: Never pipe a remote script into a shell (`curl ... | bash`, `wget ... | sh`); download the artifact to a file, verify its SHA-256 checksum, then install. Pin every external tool to a specific version with a full URL (not `latest` or `stable`). Verify checksums for all downloaded binaries, using the provider's official `.sha256` / `.sha256sum` sidecar when available. These rules apply to every download in CI

Follow these coding conventions for new/updated code (a three-line fix in a legacy file shouldn't trigger huge drive-by refactors):

- Composition over inheritance
- Never-nester: early returns over deep nesting
- Don't throw; model failures as values (One function (e.g., raise_public) maps error union to existing public exception contracts via exhaustive match + assert_never)
- No mutation; don't reassign variables, global or local. Instead of mutable lists and dicts, prefer tuples, frozen dataclasses (with slots=True), etc.
- Use dependency injection
- Fully typed; no `Any` or coarse types like `dict[str, Any]` or just `dict`. Every function parameter must be strongly typed
- Use tagged unions + match
- No monster files or god objects
- No file sprawl: deliberate file and folder structure
- Standard over hand-rolled: use the official SDK or a library where one exists; where none does, follow industry standards instead of inventing local conventions
- API-fragmentation-aware: when logic must branch on which API surface produced or consumes data (e.g. chat completions vs Anthropic Messages vs Responses API shapes), proactively look for an existing shared helper (e.g. `litellm_core_utils/prompt_templates/factory.py`) before writing per-surface parsing in the new module; if none exists, add one there instead of duplicating the same format-detection logic in every new guardrail/integration

Follow conventional commits for commit names and PR titles

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs**

Before implementing:
- State your assumptions explicitly. If uncertain, ask
- If multiple interpretations exist, present them. Don't pick silently
- If a simpler approach exists, say so. Push back when warranted
- If something is unclear, stop. Name what's confusing. Ask

## Simplicity First

**Minimum code that solves the problem. Nothing speculative**

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If you write 200 lines and it could be 50, rewrite it

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify
