import type { MCPServer } from "@/components/mcp_tools/types";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function ProviderMCPTools({ server }: { readonly server: MCPServer }) {
  const tools = server.allowed_tools ?? [];
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        These allowed tools are stored in the registration. The LLM provider discovers and executes them.
      </p>
      {tools.length === 0 ? (
        <p className="text-sm">No tools allowed</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Tool name</TableHead>
              <TableHead>Description</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {tools.map((name) => (
              <TableRow key={name}>
                <TableCell className="font-mono">{name}</TableCell>
                <TableCell>{server.tool_name_to_description?.[name] || "No description"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
