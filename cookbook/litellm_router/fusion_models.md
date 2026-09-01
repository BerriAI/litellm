# Fusion models (beta)

Fusion models expose several LiteLLM model groups as one model. For each client call, every panel model receives the
same canonical conversation and runs in parallel. An aggregator then synthesizes their work into the sole response
returned to the client

Fusion operates at the model layer. Coding agents, research loops, chat applications, and tool-using workflows keep
their current control flow and use the Fusion model name anywhere they would name one model

```text
client or harness
      |
      | one model request
      v
panel A ----\
panel B -----+--> aggregator --> one response or tool call
panel C ----/
```

If the aggregator returns a tool call, the existing harness executes it, appends the result to its conversation, and
calls the Fusion model again. That next call starts a new panel round. Fusion does not execute tools, retain private
panel transcripts, create subagents, or replace the harness

## Create a Fusion model

In the dashboard, open **Models & Endpoints**, select **Fusion Models (Beta)**, and choose **Add Fusion Model**. Select
two to six existing model groups for the panel and one existing model group as the aggregator. LiteLLM rejects nested
Fusion models

The equivalent `config.yaml` entry is:

```yaml
model_list:
  - model_name: panel/one
    litellm_params:
      model: openai/gpt-5

  - model_name: panel/two
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250929

  - model_name: fusion-aggregator
    litellm_params:
      model: openai/gpt-5

  - model_name: fusion/coding
    litellm_params:
      model: fusion_router
      fusion_router_config:
        panel_models:
          - panel/one
          - panel/two
        aggregator_model: fusion-aggregator
        min_successful_panelists: 2
        panel_timeout_seconds: 120
        max_candidate_chars: 12000
        on_quorum_failure: fail
```

Clients call `fusion/coding` like a regular model. The beta supports Chat Completions, Responses, and Anthropic
Messages, including async streaming. Panels finish before the aggregator starts, so the first streamed token arrives
during aggregation

## Presets and settings

The dashboard offers two behavior presets:

- **Quality First** sets `on_quorum_failure: fail`. The request fails when fewer than `min_successful_panelists` panel
  calls succeed, preserving the configured quality floor
- **High Availability** sets `on_quorum_failure: aggregator_only`. When the panel misses quorum, the aggregator receives
  the original request without partial candidates and answers alone

Advanced settings stay limited to the controls that affect one Fusion round:

| Setting | Meaning | Bounds |
| --- | --- | --- |
| `min_successful_panelists` | Successful panel responses required before synthesis | 1 to panel size |
| `panel_timeout_seconds` | Deadline applied to each panel call | More than 0, at most 600 seconds |
| `max_candidate_chars` | Text copied from each candidate into the synthesis request | 1,000 to 50,000 characters |

The default uses Quality First with a quorum of two, a 120-second panel deadline, and 12,000 characters per candidate.
LiteLLM runs every configured panel member because this feature optimizes answer quality rather than call cost

## Tools and active work

LiteLLM sends client-defined function schemas to every panel member. A panel can reason about available actions and
propose a function name and arguments. LiteLLM serializes those proposals as untrusted advice, discards their call IDs,
and never returns or executes them

The aggregator receives the original tools and has sole authority to emit a tool call. It creates the call and arguments
after considering the panel. LiteLLM withholds provider-hosted tools such as hosted web search from panel members because
those tools execute inside the provider. The aggregator retains them

A coding or active-task harness follows this loop:

1. The harness sends its transcript, context, and tool schemas to `fusion/coding`
2. Panel members propose answers, edits, commands, or tool use in isolation
3. The aggregator synthesizes one response or tool call
4. The harness executes the call, records the result, and invokes `fusion/coding` again

Research applications use the same loop. Fusion improves the model decision on each call while the application owns
browsing, citation collection, retries, approvals, and its completion criteria

## Conversation history and compaction

Every panel member receives the complete message list supplied on that call. The aggregator receives that list plus one
developer message containing bounded panel candidates. LiteLLM inserts the developer message after leading system and
developer instructions so the insertion preserves assistant/tool adjacency in the transcript

Only the aggregator output enters client-visible history. Panel outputs last for one Fusion round and are not replayed
on later turns. The canonical conversation therefore matches the transcript a client would retain for one model, and
every later panel round sees it. Replaying panel reasoning would multiply context use and create conflicting histories

Fusion does not compact across turns. If the client or harness summarizes or truncates the canonical conversation,
every panel member and the aggregator see that compacted transcript on the next call. Anthropic Messages context
management runs before the same Fusion core

## Failure, health, and observability

Panel calls fail independently. A provider error, timeout, streaming response where Fusion expected a complete
candidate, or empty response counts as one failed panelist. Quality First requires a healthy aggregator and enough
healthy panel dependencies to meet quorum. High Availability requires a healthy aggregator and still reports each
panel's health without taking the virtual model down

Each child provider call keeps its LiteLLM logging and spend record. Panel calls include
`internal_call_origin: fusion_panel`; the aggregator remains the authoritative call for the parent request. A successful
Fusion round makes one billable call per panelist plus one aggregator call

## Beta boundaries

- Fusion supports `n=1`. LiteLLM rejects multiple returned choices because Fusion must produce one authoritative result
- A Fusion model cannot serve as a panel member or aggregator for another Fusion model
- Responses background jobs are unsupported
- Synchronous Python streaming through `Router.completion` and `Router.responses` is unsupported. Use their async
  counterparts. Proxy streaming uses the async paths
- Embeddings, image generation, audio, batch jobs, and other non-conversational endpoints bypass Fusion. The feature
  covers conversational model calls and tool loops rather than every LiteLLM API type

These boundaries keep each model call deterministic: one parallel panel round runs before one aggregation, and the
caller retains task lifecycle control

## Design assumptions

The implementation starts with five testable assumptions:

1. The aggregator should synthesize the panel's work. Its prompt permits combining, correcting, rejecting, or replacing
   candidates and preserving supported minority observations
2. Aggregating on every model call gives operators one predictable policy. The first beta has no hidden cadence or
   conductor-owned state
3. The canonical client transcript is the sole durable history. Private panel histories would create divergent agents,
   which belongs in a harness
4. Function-tool awareness helps coding and active work, while one aggregator retains execution authority
5. Quality is the primary optimization target. LiteLLM exposes cost and latency as consequences rather than routing
   inputs

Evaluations can vary panel composition, aggregator choice, quorum, and failure preset without changing the API contract
or the harness under test
