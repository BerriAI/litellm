# ADR 000: Record architecture decisions

Status: Accepted

## Context

Decisions about the Rust core get made in PR threads, Slack, and people's heads. Six months later nobody remembers why a crate boundary sits where it does or why an obvious alternative was rejected, so the same debate happens again or someone undoes a choice without knowing what it was protecting

## Decision

We write a short ADR for any decision that is hard to reverse or that a new contributor would reasonably question. Each ADR lives in this folder as `adr-NNN-short-title.md`, numbered in order, and has three sections: Context, Decision, and Consequences. If writing one takes more than a few minutes, it is too long

ADRs are never edited after they are accepted, apart from the status line. To change course, write a new ADR and mark the old one `Superseded by ADR NNN`

## Consequences

Reviewers and future contributors can find the reasoning behind a choice without archaeology. The cost is a few minutes per significant decision and the discipline to actually write them
