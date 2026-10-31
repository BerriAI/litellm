# ADR 002: Virtual routing calls settle usage independently

Status: Proposed

## Context

Switchyard can call an efficient model, judge its response, and either return that response or call a capable model. The existing pre-routing hook only selects a deployment. Repeating an accepted response through another completion would generate a second answer and charge twice

## Decision

Dispatch Switchyard chat requests through an execution strategy before ordinary deployment selection. The strategy calls configured models through the owning Python router, with separate call IDs and inherited caller attribution. Each provider call records its own usage and cost. The virtual request does not record an additional provider charge

Child calls do not own the virtual request's admission reservation. The router host releases that reservation after execution finishes, fails, or is cancelled. When the final provider call streams, release waits for stream completion or closure

Return an accepted provider response unchanged, including usage extras described in ADR 001. Judged responses are buffered. A caller requesting streaming receives the accepted response through the existing response-to-stream conversion, without an additional billing wrapper. Ordinary final calls can stream directly

## Alternatives Considered

Returning only a model name through the pre-routing hook cannot reuse an answer already generated during escalation

Combining every internal call into a fabricated parent usage object would lose model-specific pricing and make provider usage fields ambiguous

## Consequences

Provider execution and billing remain in LiteLLM, while Switchyard owns routing decisions. Logs contain separate provider calls linked to the caller's session

Judged streaming has additional time to first token. Cancellation, reservation cleanup, and the sum of child charges require end-to-end verification. The initial execution strategy supports async Chat Completions only and rejects other endpoint paths until their adapters are implemented
