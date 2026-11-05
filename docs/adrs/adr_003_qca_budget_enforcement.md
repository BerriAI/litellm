# Qoder Cloud Agents budget enforcement

## Status

Accepted

## Context

Qoder Cloud Agents reports cumulative provider credits for a session. LiteLLM monetary budgets require a cost in a
configured currency for each request. Qoder does not publish a stable currency conversion, and charging the cumulative
session credits on every continuation would bill earlier turns more than once

## Decision

The proxy rejects Qoder Cloud Agents requests from callers with an enforced key, user, team, organization, end-user, or
model monetary budget. The rejection happens before creating or continuing a provider session. Callers without monetary
budgets can use Qoder Cloud Agents, and the response keeps the cumulative Qoder credits as provider usage fields

## Alternatives Considered

Converting credits with a fixed rate was rejected because LiteLLM does not own a stable Qoder credit price

Charging cumulative credits on every interaction was rejected because session continuations would repeatedly charge
earlier turns

Tracking session deltas in proxy memory was rejected because it would not be reliable across workers or restarts

## Consequences

Monetary budgets cannot silently admit unpriced Qoder usage. Operators that need budgeted access must wait for a stable,
deployment-owned conversion and durable per-session delta accounting. Existing unbudgeted integrations continue to
receive the provider credit fields without a guessed monetary cost
