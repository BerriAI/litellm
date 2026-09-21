/**
 * Formatted view of tool definition with parameters table and call data
 */

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ParsedTool, ParameterRow } from "./types";

interface FormattedToolViewProps {
  tool: ParsedTool;
}

export function FormattedToolView({ tool }: FormattedToolViewProps) {
  // Parse parameters for table display
  const parameterRows: ParameterRow[] = Object.entries(tool.parameters?.properties || {}).map(
    ([name, schema]: [string, any]) => ({
      key: name,
      name: name,
      type: schema.type || "any",
      description: schema.description || "-",
      required: tool.parameters?.required?.includes(name) || false,
    }),
  );

  return (
    <div>
      {/* Description */}
      {tool.description && (
        <div className="mb-4">
          <span className="whitespace-pre-wrap leading-relaxed">{tool.description}</span>
        </div>
      )}

      {/* Parameters Table */}
      {parameterRows.length > 0 && (
        <div>
          <span className="mb-2 block text-xs text-muted-foreground">Parameters</span>
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
                    <code>
                      {row.name}
                      {row.required && <span className="text-destructive">*</span>}
                    </code>
                  </TableCell>
                  <TableCell>
                    <code className="text-info">{row.type}</code>
                  </TableCell>
                  <TableCell>
                    <span className="text-muted-foreground">{row.description}</span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* If tool was called, show the arguments used */}
      {tool.called && tool.callData && (
        <div className="mt-4">
          <span className="mb-2 block text-xs text-muted-foreground">Called With</span>
          <div className="rounded border border-success/30 bg-success/10 p-3">
            <pre className="m-0 whitespace-pre-wrap break-words text-xs text-foreground">
              {JSON.stringify(tool.callData.arguments, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
