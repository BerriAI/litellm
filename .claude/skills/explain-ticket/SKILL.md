---
name: explain-ticket
description: Comprehensively explain a Linear ticket by reading the ticket plus every explicitly attached Slack, Pylon, email, and GitHub resource, and by inspecting the code in two parallel worktrees (the version reported in the ticket, if any, and the latest litellm_internal_staging). Use when asked to explain, investigate, or get up to speed on a Linear ticket.
argument-hint: <ticket id or Linear URL>
---

# Explain a Linear ticket

The deliverable is an explanation of the ticket, not a fix. Do not change code, do not open PRs, and do not comment on the ticket unless explicitly asked. Read comprehensively, then report

## 1. Read the ticket itself

Resolve the ticket id or URL from the arguments. If none was given, ask for it instead of guessing

Using the Linear MCP tools, fetch the issue (get_issue), all comments (list_comments), and its attachments/links. Note the reporter, assignee, status, labels, customer needs, parent/child issues, and any linked PRs or commits. Extract from the ticket text: the symptom, the expected behavior, any error messages or stack traces, and the LiteLLM version the reporter was running (if stated anywhere in the ticket, comments, or attached threads)

## 2. Follow explicitly attached context only

Hard rule: never use search tools against Slack, Pylon, or Gmail to find threads that "might" be related. Only open resources that are explicitly linked or attached, starting from the ticket

Chaining is allowed and encouraged, as long as every hop derives from an explicit attachment. Examples of the intended traversal:

- The ticket links a Pylon issue: fetch it (get_issue, get_issue_messages) and read the full conversation
- That Pylon issue has a Slack thread attached: read it with slack_read_thread
- The ticket or a thread links an email thread: fetch that specific thread with the Gmail get_thread/get_message tools
- A thread links a GitHub issue, PR, or gist: fetch it with the GitHub MCP tools
- Any of those link further resources (docs, screenshots, config files, follow-up threads): keep going

The distinction is navigation vs discovery. Following a link someone attached is navigation and is always fine, to any depth. Querying Slack/Pylon/Gmail search endpoints to discover unlinked context is forbidden, even if you are confident a related thread exists

While reading, collect concrete reproduction details: config snippets, model names, request payloads, curl commands, versions, and timestamps. Quote error messages verbatim in your notes; they anchor the code investigation

## 3. Investigate the code in two parallel worktrees

Create fresh worktrees outside the main checkout so the primary working tree is untouched (a scratch directory is fine):

1. If the ticket reports a version (e.g. "v1.80.5" or "1.80.5-stable"), find the matching tag (`git tag --list '*<version>*'`) and add a worktree at it: `git worktree add <scratch>/litellm-reported <tag>`
2. Always fetch and add a worktree at the latest staging head: `git fetch origin litellm_internal_staging && git worktree add <scratch>/litellm-staging origin/litellm_internal_staging`

If no version is reported, skip the first worktree and investigate staging only

Launch one Explore subagent per worktree, in parallel (a single message with multiple Agent calls). Give each agent the symptom, verbatim error messages, and repro details from steps 1-2, and ask it to locate the code paths involved, explain the relevant behavior at that revision, and assess whether the reported behavior is plausible there. Then compare the two answers: does the bug exist at the reported version, does it still exist on staging, was it introduced or fixed in between (use `git log`/`git blame` on the implicated files to find the commit if so)

Clean up when done: `git worktree remove <path>` for each worktree you added

## 4. Report

Lead with a short plain-language summary of what the ticket is about and what is actually going on. Then cover, in prose:

- Who reported it and through which channel(s), with what they were trying to do
- The symptom and the exact error, plus repro details gathered from the threads
- What the code does today: the root cause (or best-supported hypothesis, clearly labeled as such) with `file:line` references
- The reported version vs staging comparison: still broken, already fixed (by which commit/PR), or behavior changed
- Anything the threads reveal that the ticket text omits: workarounds already suggested, customer urgency, related tickets, constraints on the fix
- Open questions that block a confident diagnosis, and what would resolve them

Cite where each key fact came from (ticket, specific comment, Slack thread, Pylon message, email) so the reader can verify without redoing the traversal
