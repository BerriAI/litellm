import { z } from "zod";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const upstreamResponseFields = {
  attempt_id: z.string(),
  status_code: z.number().nullable(),
  headers: z.array(z.tuple([z.string(), z.string()])),
  truncated: z.boolean(),
};
const upstreamResponses = z.array(z.object(upstreamResponseFields));

export function UpstreamResponseHeaders({ data }: { data: unknown }) {
  const parsed = upstreamResponses.safeParse(data);
  if (!parsed.success || parsed.data.length === 0) return null;

  return (
    <Card size="sm" className="mb-6">
      <CardHeader>
        <CardTitle>Upstream Response Headers</CardTitle>
      </CardHeader>
      <CardContent>
        {parsed.data.map((response, index) => (
          <section key={`${response.attempt_id}-${index}`} aria-label={`Upstream response ${index + 1}`}>
            <div className="mb-2 text-sm font-medium">
              Response {index + 1}
              {response.status_code !== null ? ` · HTTP ${response.status_code}` : ""}
            </div>
            <table className="mb-4 w-full table-fixed text-left text-xs">
              <thead>
                <tr>
                  <th className="w-1/3 p-2">Header</th>
                  <th className="p-2">Value</th>
                </tr>
              </thead>
              <tbody>
                {response.headers.map(([name, value], headerIndex) => (
                  <tr key={`${name}-${headerIndex}`} className="border-t">
                    <td className="break-all p-2 font-mono align-top">{name}</td>
                    <td className="break-all whitespace-pre-wrap p-2 font-mono">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {response.truncated && (
              <p className="text-xs text-muted-foreground">
                Some headers or earlier responses were omitted because of logging limits
              </p>
            )}
          </section>
        ))}
      </CardContent>
    </Card>
  );
}
