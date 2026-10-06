---
name: agent-docs
description: Add, move, or edit agent guidance in litellm-rust (AGENTS.md files and .agents/skills), including choosing between a directory AGENTS.md and a skill, where the file goes, and how to write it
---

# Agent docs

## AGENTS.md or skill

An `AGENTS.md` holds rules for the code under its directory and is read whenever an agent works there. Ownership, boundaries, invariants, and design constraints of a crate or module go in the `AGENTS.md` of the directory that owns them. Guidance specific to one crate always goes there, however long it gets

A skill holds a technique that applies anywhere in the workspace and is loaded by task rather than by path: how to use a library, a coding pattern, or a cross-cutting concern such as tracing or string enums. Write one only when the same guidance would otherwise be copied into several unrelated `AGENTS.md` files

A feature that spans a few crates is not cross-cutting. Its rules go in the `AGENTS.md` of the crate that owns the contract, and the other crates' `AGENTS.md` point there in one line

`litellm-rust/AGENTS.md` is the index of skills, one line each saying when to follow it. Workspace conventions tied to a task, such as writing tests or defining errors, are skills too. Only a rule that applies to every Rust change regardless of task belongs in that file directly

## Placement

- Crate rules: `crates/<crate>/AGENTS.md`
- Module rules: `crates/<crate>/src/<module>/AGENTS.md`, only when the module has its own boundary. Otherwise extend the crate file
- Skills: `.agents/skills/<name>/SKILL.md`, with `name` matching the directory. Link each skill from `litellm-rust/AGENTS.md` in one line saying when to follow it, because agents that don't index skills only find it through that link
- Material a skill references, such as long examples or tables, sits next to its `SKILL.md` and is linked from it

## Content

- State what code must or must not do, and why when the reason is not obvious. Don't describe what the code already shows
- Write current rules, not history: no changelogs, migration stories, or "previously"
- Name the owner of each concern and the neighbor that owns what this one does not
- Reference types, functions, and files by path so a reader can verify them. Delete a rule when the code it names is gone
- Prefer a lint, type, or test that enforces a rule over prose. When one exists, name it instead of restating the rule, as `clippy.toml` does for HTTP clients and SQL
- Don't repeat a parent `AGENTS.md`; nested files add to it. Link between files instead of copying
- Link external protocol references, such as provider API docs, instead of paraphrasing them

## Style

Match the existing files: short imperative prose paragraphs, bullets only for lists of independent rules, no trailing period on a paragraph or bullet, no emojis, no "—"

Skill frontmatter has `name` and a `description` naming the task that triggers it, specific enough that an agent can decide relevance from that line alone. When a skill grows past about 100 lines, split it by task
