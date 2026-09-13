import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { importMCPServers } from "@/components/networking";
import { toast } from "@/lib/toast";
import { MCPConnectorImportResponse, parseConnectorConfig } from "./importConnectorConfig";

interface ImportMCPServersProps {
  accessToken: string;
  open: boolean;
  onClose: () => void;
  onImported: () => void;
}

const PLACEHOLDER = `{
  "mcpServers": {
    "my_server": {
      "url": "https://example.com/mcp",
      "authorization_token": "..."
    }
  }
}`;

const ImportMCPServers: React.FC<ImportMCPServersProps> = ({ accessToken, open, onClose, onImported }) => {
  const [configText, setConfigText] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [result, setResult] = useState<MCPConnectorImportResponse | null>(null);

  const handleClose = () => {
    setConfigText("");
    setParseError(null);
    setResult(null);
    onClose();
  };

  const handleImport = async () => {
    const parsed = parseConnectorConfig(configText);
    if (!parsed.ok) {
      setParseError(parsed.error);
      return;
    }
    setParseError(null);
    setIsImporting(true);
    try {
      const response = (await importMCPServers(accessToken, parsed.payload)) as MCPConnectorImportResponse;
      setResult(response);
      if (response.imported.length > 0) {
        toast.success(`Imported ${response.imported.length} MCP server${response.imported.length === 1 ? "" : "s"}`);
        onImported();
      }
    } catch (error) {
      console.error("Failed to import MCP servers:", error);
      setParseError("Import request failed. Check the proxy logs for details.");
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && handleClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import MCP Connectors</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Paste an Anthropic connector configuration: the <code>mcpServers</code> mapping from a Claude Desktop /
            Claude Code config file, or the <code>mcp_servers</code> array from the Anthropic Messages API.
          </p>
          <Textarea
            aria-label="Connector JSON"
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            placeholder={PLACEHOLDER}
            rows={10}
            className="font-mono text-xs"
          />
          {parseError && (
            <Alert variant="destructive">
              <AlertTitle>{parseError}</AlertTitle>
            </Alert>
          )}
          {result && (
            <div className="space-y-2 text-sm">
              {result.imported.length > 0 && (
                <div>
                  <span className="font-semibold">Imported:</span>{" "}
                  {result.imported.map((entry) => entry.alias || entry.name).join(", ")}
                </div>
              )}
              {result.skipped.length > 0 && (
                <div>
                  <span className="font-semibold">Skipped:</span>
                  <ul className="ml-4 list-disc">
                    {result.skipped.map((entry) => (
                      <li key={entry.name}>
                        {entry.name}: {entry.reason}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {result.errors.length > 0 && (
                <div>
                  <span className="font-semibold">Failed:</span>
                  <ul className="ml-4 list-disc">
                    {result.errors.map((entry) => (
                      <li key={entry.name}>
                        {entry.name}: {entry.error}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={handleClose} disabled={isImporting}>
              Close
            </Button>
            <Button onClick={handleImport} disabled={isImporting}>
              {isImporting ? "Importing..." : "Import"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ImportMCPServers;
