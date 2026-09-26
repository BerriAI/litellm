import { useRoutingUsage } from "../useRoutingUsage";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { routingSources, routingSpend, trafficShare, type RoutingUsageScope } from "../routingUsage";
import { ApiError } from "@/lib/http/client";

export default function ModelRoutingShare({ model, scope }: { model: string; scope: RoutingUsageScope }) {
  const { data, isPending, isError, error } = useRoutingUsage({ ...scope, destination_model: model });
  if (isPending) return <p className="text-xs text-muted-foreground">Loading traffic sources...</p>;
  if (isError)
    return (
      <p className="text-xs text-muted-foreground">
        {error instanceof ApiError && error.status === 400 ? error.message : "Traffic sources unavailable"}
      </p>
    );
  const sources = routingSources(data ?? []);
  const total = sources.reduce((sum, source) => sum + source.requests, 0);
  if (total === 0) return <p className="text-xs text-muted-foreground">No retained requests for traffic sources</p>;
  const routed = sources.filter((source) => source.name !== null).reduce((sum, source) => sum + source.requests, 0);
  return (
    <Popover>
      <PopoverTrigger className="text-xs text-muted-foreground underline decoration-dotted underline-offset-4">
        {trafficShare(routed, total)} via auto-router
      </PopoverTrigger>
      <PopoverContent align="start" className="w-96 max-w-[calc(100vw-2rem)]">
        <PopoverTitle>Traffic sources</PopoverTitle>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Source</TableHead>
              <TableHead className="text-right">Requests</TableHead>
              <TableHead className="text-right">Share</TableHead>
              <TableHead className="text-right">Spend</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sources.map((source) => (
              <TableRow key={source.name ?? "__direct__"}>
                <TableCell className="max-w-40 break-words whitespace-normal">{source.name ?? "Direct"}</TableCell>
                <TableCell className="text-right tabular-nums">{source.requests.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums">{trafficShare(source.requests, total)}</TableCell>
                <TableCell className="text-right tabular-nums">{routingSpend(source.spend)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <p className="text-xs text-muted-foreground">
          Based on {total.toLocaleString()} retained requests in this period. Model spend excludes internal classifier
          and shadow-evaluation calls. Usage totals can include requests whose logs have expired.
        </p>
      </PopoverContent>
    </Popover>
  );
}
