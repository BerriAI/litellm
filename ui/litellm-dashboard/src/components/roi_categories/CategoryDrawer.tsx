import { ExternalLink } from "lucide-react";
import type { UseQueryResult } from "@tanstack/react-query";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { uiHref } from "@/utils/uiHref";
import { money, requestQuote, type Example, type categorySummary } from "./data";

type Summary = ReturnType<typeof categorySummary>[number];

function ExampleTraffic({ query }: { query: UseQueryResult<Example[], Error> }) {
  if (query.isPending) return <p role="status">Loading examples…</p>;
  if (query.error) return <p role="alert">{query.error.message}</p>;
  if (!query.data?.length) return <p className="text-sm text-muted-foreground">No examples available.</p>;
  return query.data.map((example) => (
    <article
      key={example.request_id}
      className="mb-4 space-y-3 rounded-lg border border-l-4 border-l-teal-600 bg-muted/20 p-4"
    >
      <p className="text-xs text-muted-foreground">
        {example.model} · {example.spend === null ? "Cost unavailable" : money(example.spend)}
      </p>
      <blockquote className="whitespace-pre-wrap break-words text-sm">
        {example.excerpt ? requestQuote(example.excerpt) : "Content logging was unavailable for this request."}
      </blockquote>
      <a
        className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
        href={uiHref(`/logs?log_id=${encodeURIComponent(example.request_id)}`)}
        target="_blank"
        rel="noreferrer"
      >
        View source request <ExternalLink className="size-3" />
      </a>
    </article>
  ));
}

export function CategoryDrawer({
  selected,
  total,
  examples,
  onClose,
}: {
  selected: Summary | undefined;
  total: number;
  examples: UseQueryResult<Example[], Error>;
  onClose: () => void;
}) {
  if (!selected) return null;
  const metrics = [
    { label: "Selected spend", value: money(selected.spend) },
    { label: "Requests", value: selected.requests.toLocaleString() },
    { label: "Selected share", value: `${total ? ((100 * selected.spend) / total).toFixed(1) : "0"}%` },
  ];
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto sm:max-w-xl">
        <SheetHeader>
          <SheetTitle>{selected.name}</SheetTitle>
          <SheetDescription>{selected.description}</SheetDescription>
        </SheetHeader>
        <div className="space-y-7 p-5">
          <div className="grid grid-cols-3 rounded-lg border p-4">
            {metrics.map((metric) => (
              <div key={metric.label}>
                <p className="text-xs text-muted-foreground">{metric.label}</p>
                <p className="mt-1 font-semibold">{metric.value}</p>
              </div>
            ))}
          </div>
          <section>
            <h3 className="mb-3 font-medium">Models used</h3>
            {selected.models.map((item) => (
              <div key={item.model} className="flex justify-between gap-4 border-b py-2 text-sm">
                <span className="truncate">{item.model}</span>
                <span className="tabular-nums">{money(item.spend)}</span>
              </div>
            ))}
          </section>
          <section>
            <h3 className="mb-3 font-medium">Example traffic</h3>
            <p className="mb-4 text-xs text-muted-foreground">
              Highest-spend requests in the selected view. Excerpts may be truncated.
            </p>
            <ExampleTraffic query={examples} />
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
