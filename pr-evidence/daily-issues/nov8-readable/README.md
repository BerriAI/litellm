# Readable Alerts inbox

UI source `8168a596c08d838662aa413b9ffa23dd69f4b1a1`. The before capture uses `4ac2619a113` on the same isolated sample history. Provider calls are not part of this UI check.

The inbox presents each alert as a sentence, its actual and baseline values, and affected resources. Each entry opens the existing investigation. Filters, status tabs and bounded pagination remain available. The same readable heading appears in the investigation.

Open `/alerts` as a gateway administrator. Read the spend, request, latency, error-rate and retirement entries. Select the Research agent spend entry to inspect saved comparisons and Activity. Return to Alerts, filter by type, and open Settings. Repeat at 1440px and 390px widths in light and dark mode.

The proxy uses two API workers, dedicated Redis and PostgreSQL. Daily history is synthetic and screenshots are illustrative product evidence, not throughput or memory measurements. No request to an external model was made for this capture.

37 focused UI tests pass. Production TypeScript and the scoped UI lint gate pass. Live navigation renders the saved comparison table. There are no browser exceptions or horizontal document overflow in the captured viewports. CI and automated reviewer results remain separate merge gates.
