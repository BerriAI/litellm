# ADR 000: Record architecture decisions

Status: Accepted

## Context

Decisions about the Rust core get made in various places, and in synchronous conversations or on whiteboards.
Knowing the context of when and how the decision was made helps greatly to contextualize code for a reader, as well as informing agents about how things should be done.
Especially with the rust core, there are a great many decisions that need to be made in a large solution space, and the reasons for such decisions may not be obvious.

## Decision

We write a short ADR for any decision that is hard to reverse or that a new contributor would reasonably question. Each ADR lives in this folder as `adr_NNN_short_title.md`, numbered in order, and has four sections: Context, Decision, Alternatives Considered, and Consequences. If writing one takes more than a few minutes, it is too long.

ADRs are never edited after they are accepted, apart from the status line. To change course, write a new ADR and mark the old one `Superseded by ADR NNN`.

ADRs should be hand-written as much as possible, with the intention of being
extremely intention-dense. Agents have a habit of making statements stronger
than they otherwise should be, which makes it challenging to interpret
reasoning, which is the entire point of ADRs.

ADR filenames should ideally make it clear what decision was made. Their
contents should explain (briefly) alternatives considered, as well as
consequences of the decision in pro/con format.

## Alternatives Considered

**Not using ADRs**: This would make us able to move a bit faster, but it makes decisionmaking less clear. Additionally, given the lack of other code documentation, there is no living document other than the code.

## Consequences

Benefits:

- Reviewers and future contributors can find reasoning behind a choice easily

Costs:

- A few minutes of thought and writing per significant decision
