import type { JsonValue } from "./contracts";

export function TracePayload({ label, value }: { label: string; value: JsonValue }) {
  const type = value === null ? "null" : typeof value;
  const typeLabel = Array.isArray(value) ? "array" : type;
  return (
    <section aria-label={label} className="min-w-0 overflow-hidden rounded-lg border bg-background">
      <div className="flex items-center justify-between border-b px-4 py-2.5">
        <h4 className="text-sm font-medium">{label}</h4>
        <span className="font-mono text-xs text-muted-foreground">{typeLabel}</span>
      </div>
      <pre className="max-h-72 overflow-auto p-4 text-xs leading-6" tabIndex={0} aria-label={`${label} JSON`}>
        {JSON.stringify(value, null, 2)}
      </pre>
    </section>
  );
}
