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

When creating PRs, don't set base to `main`. `litellm_internal_staging` serves that purpose

Always use @.github/pull_request_template.md as a guide for your PR body

Never use `pytest` commands or the like as "Screenshots / Proof of Fix". We prefer curl'ing a live proxy instance running on localhost:4000 (I like to run it with `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log`) and showing both the command run and the output. Also, it should hit real LLM provider APIs, not mocks, and cost real $$$ because that is the most realistic test. The proof of fix should be exactly what the end user / customer would see / do. The run logs in PR #27703 is a prime example of how to do it (not a huge fan of using a python test script that future me and the team will have no visibility into; I prefer just curl commands or a short list of bash commands (e.g., using `for`)). If it's a UI thing, just tell me which URLs to go to (e.g., http://localhost:4000/ui/?page=logs), where to click, what fields to fill out, etc. along with the other commands to run in an ordered list, and I'll do it myself and post the screenshots after you make the PR

If you ever make public-facing PR descriptions, comments, issues, commit messages, etc., always follow these guidelines to sound less AI-y:
- don't use emojis
- don't use "—". Instead, reach for ";", ".", etc.
- don't use the pattern "It's not X, it's Y", "You're not X, you're Y", etc.
- don't use bulleted or numbered lists unless it would be nonsensical not to. Instead, prefer prose
- don't add a trailing "." at the end of paragraphs (just like this file). That means every paragraph, not just the last one (of the markdown file, PR description, GitHub comment, etc.). Rule of thumb: if you're adding new line(s) before the next sentence, don't add a "."
- don't use →. Instead, prefer not to use arrows, and if need be, use -> instead

Don't hesitate to use values in .env to get needed API keys and other secrets, as long as you never add them to conversation history, commit them, or include them in GitHub issues / PRs

Python max line length is 120, not 88

Run tests before you commit. Also, run `make pre-commit` right before each commit, which generates types (as needed) and formats/lints your code. Any errors found must be fixed. It only runs when there are staged frontend and/or backend changes and calculates violations, generates types, etc. based on the worktree, so stage what you need or stash/delete unwanted files in litellm/ or ui/ (where backend and frontend lint run, respectively) before running it. If it fails because dashboard api types are stale, it already regenerated them for you. You just need to stage the schema.d.ts, re-run `make pre-commit` to confirm it passes, and commit

When you fix violations gated by `ruff-strict-budget.json`, `type-discipline-budget.json`, or `basedpyright-code-budget.json`, run `make lint-budget-update` and commit the lowered limits so the ceilings ratchet down instead of leaving stale headroom. It measures the working tree, so it must contain exactly the fixes you're committing

If you're trying to create a new function that relies on untyped stuff, instead of adding more Any's and pushing `reportAny` / `reportExplicitAny` closer to their basedpyright ceilings, just validate it in the caller with Pydantic (a model or `TypeAdapter` that returns the typed thing or raises will do) and then pass the now typed variable in

If you get an LIT001 or LIT002 fail, refactor the code to follow functional programming best practices rather than introducing mutable data structures. For example, build values in one shot with comprehensions or generators wrapped in `tuple()` / `frozenset()` instead of seeding an empty `list`/`dict`/`set` and mutating it over time. Ideally `# mutable-ok` is never used; reach for it only as a genuine last resort when an immutable rewrite is truly impossible, and always pair it with a real reason

Commit and push your work when you're done without asking

When you must use real LLM models to, for example, write e2e tests, write a QA runbook, etc., make sure to use the latest models (doesn't have to be smartest, can also be a modern small, fast one. No strong preference for smart vs fast here, just use something modern) as of the year and month of the current date. Do a web search as necessary to figure that out

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
