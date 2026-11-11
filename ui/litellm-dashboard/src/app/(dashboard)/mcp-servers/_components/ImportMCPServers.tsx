import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
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
    const parsed = parseConnectorConfig(configText, t);
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
        toast.success(t("mcpServers.importMcpServers.toasts.imported", { count: response.imported.length }));
        onImported();
      }
    } catch (error) {
      console.error("Failed to import MCP servers:", error);
      setParseError(t("mcpServers.importMcpServers.errors.requestFailed"));
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && handleClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("mcpServers.importMcpServers.title")}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            <Trans i18nKey="mcpServers.importMcpServers.description" components={{ code: <code /> }} />
          </p>
          <Textarea
            aria-label={t("mcpServers.importMcpServers.connectorJsonLabel")}
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
                  <span className="font-semibold">{t("mcpServers.importMcpServers.result.imported")}</span>{" "}
                  {result.imported.map((entry) => entry.alias || entry.name).join(", ")}
                </div>
              )}
              {result.skipped.length > 0 && (
                <div>
                  <span className="font-semibold">{t("mcpServers.importMcpServers.result.skipped")}</span>
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
                  <span className="font-semibold">{t("mcpServers.importMcpServers.result.failed")}</span>
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
              {t("mcpServers.importMcpServers.close")}
            </Button>
            <Button onClick={handleImport} disabled={isImporting}>
              {isImporting ? t("mcpServers.importMcpServers.importing") : t("mcpServers.importMcpServers.import")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ImportMCPServers;
