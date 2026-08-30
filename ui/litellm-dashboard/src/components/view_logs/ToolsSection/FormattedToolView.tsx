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
        <div style={{ marginBottom: 16 }}>
          <span
            style={{
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
            }}
          >
            {tool.description}
          </span>
        </div>
      )}

      {/* Parameters Table */}
      {parameterRows.length > 0 && (
        <div>
          <span
            className="text-muted-foreground"
            style={{
              fontSize: 12,
              display: "block",
              marginBottom: 8,
            }}
          >
            Parameters
          </span>
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
        <div style={{ marginTop: 16 }}>
          <span
            className="text-muted-foreground"
            style={{
              fontSize: 12,
              display: "block",
              marginBottom: 8,
            }}
          >
            Called With
          </span>
          <div
            style={{
              background: "#f6ffed",
              border: "1px solid #b7eb8f",
              borderRadius: 4,
              padding: 12,
            }}
          >
            <pre
              style={{
                margin: 0,
                fontSize: 12,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {JSON.stringify(tool.callData.arguments, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
