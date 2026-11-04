# Architecture

This folder explains how the parts of LiteLLM work inside, for the people who
change them. It covers one component so far, the router. More will follow as the
Rust migration moves each module behind a public interface and into Rust.

Each component's docs follow the [Diataxis](https://diataxis.fr/) layout, so you
can choose the kind of page that fits what you need right now.

![A two-by-two grid of the four kinds of pages: tutorials, how-to guides,
explanation and reference, placed on a studying-to-working axis and a
theory-to-practice axis](img/diataxis-quadrants.svg)

| Kind | Read it when | Router |
| --- | --- | --- |
| Tutorial | you are new to the component and want to see it work | [Route your first requests](router/tutorial/route-your-first-requests.md) |
| How-to guide | you have a specific task to get done | [router/how-to](router/how-to/) |
| Reference | you need an exact name, default, key or signature | [router/reference](router/reference/) |
| Explanation | you want to know why it works the way it does | [router/explanation](router/explanation/) |

Decisions that are hard to reverse are recorded separately as ADRs in
[`docs/adrs/`](../adrs/AGENTS.md). These pages describe the code as it is on the
branch they ship with, and they link to the source so you can check them. When
the code and a page disagree, the code is right, and the page needs a fix in the
same PR that changed the behavior.
