`litellm-runtime` is an OCR-first reusable execution layer. It depends on core and must not depend on ai-gateway or python-bridge.

Runtime OCR owns provider/model resolution, runtime config selection, environment and credential resolution, URL and headers, document materialization, HTTP, Azure polling, and lifecycle ordering. Provider-specific request and decoded-response transformations remain in core. Public request types and lifecycle seams must stay host-neutral so ai-gateway can adapt its logger, guardrail, and metadata types without creating a reverse dependency.

Do not migrate audio, realtime, Messages, Chat, Responses, or generic lifecycle code into this crate as part of OCR work.
