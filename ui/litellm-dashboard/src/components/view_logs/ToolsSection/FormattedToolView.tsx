import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ParsedTool, ParameterRow } from "./types";

interface FormattedToolViewProps {
  tool: ParsedTool;
}

export function FormattedToolView({ tool }: FormattedToolViewProps) {
  const parameterRows: ParameterRow[] = Object.entries(
    tool.parameters?.properties || {},
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ).map(([name, schema]: [string, any]) => ({
    key: name,
    name: name,
    type: schema.type || "any",
    description: schema.description || "-",
    required: tool.parameters?.required?.includes(name) || false,
  }));

  return (
    <div>
      {tool.description && (
        <div className="mb-4">
          <span className="leading-relaxed whitespace-pre-wrap">
            {tool.description}
          </span>
        </div>
      )}

      {parameterRows.length > 0 && (
        <div>
          <span className="text-xs text-muted-foreground block mb-2">
            Parameters
          </span>
          <div className="border border-border rounded-md overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Parameter</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Description</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {parameterRows.map((row) => (
                  <TableRow key={row.key}>
                    <TableCell>
                      <code className="text-xs bg-muted px-1 py-0.5 rounded">
                        {row.name}
                        {row.required && (
                          <span className="text-destructive">*</span>
                        )}
                      </code>
                    </TableCell>
                    <TableCell>
                      <code className="text-xs bg-muted px-1 py-0.5 rounded text-blue-600 dark:text-blue-400">
                        {row.type}
                      </code>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {row.description}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {tool.called && tool.callData && (
        <div className="mt-4">
          <span className="text-xs text-muted-foreground block mb-2">
            Called With
          </span>
          <div className="bg-emerald-50 border border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900 rounded p-3">
            <pre className="m-0 text-xs whitespace-pre-wrap break-words">
              {JSON.stringify(tool.callData.arguments, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
