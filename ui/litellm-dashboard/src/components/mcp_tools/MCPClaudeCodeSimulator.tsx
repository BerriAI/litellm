import React, { useMemo } from "react";
import { Modal, Button, Typography } from "antd";
import { MCPServer } from "./types";
import { getMissingUserFields } from "./header_variables_prototype";

const { Text: AntdText } = Typography;

interface MCPClaudeCodeSimulatorProps {
  server: MCPServer | null;
  open: boolean;
  onClose: () => void;
  onOpenUserFields: (server: MCPServer) => void;
}

const MCPClaudeCodeSimulator: React.FC<MCPClaudeCodeSimulatorProps> = ({
  server,
  open,
  onClose,
  onOpenUserFields,
}) => {
  const missing = useMemo(() => (server ? getMissingUserFields(server) : []), [server, open]);

  if (!server) return null;

  const serverLabel = server.alias || server.server_name || "mcp_server";
  const hasErrors = missing.length > 0;
  const dashboardLink =
    typeof window !== "undefined"
      ? `${window.location.origin}${window.location.pathname}?fill_for=${encodeURIComponent(server.server_id || server.alias || "")}`
      : "";
  const cmd = `claude mcp call ${serverLabel} list_tables`;

  return (
    <Modal
      open={open}
      title={
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
          <span className="inline-block w-2 h-2 rounded-full bg-yellow-400" />
          <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
          <span className="ml-3 font-mono text-sm text-gray-600">claude-code — simulated</span>
        </div>
      }
      onCancel={onClose}
      footer={
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Close</Button>
          {hasErrors && (
            <Button
              type="primary"
              danger
              onClick={() => {
                onClose();
                onOpenUserFields(server);
              }}
            >
              Fill missing fields →
            </Button>
          )}
        </div>
      }
      width={760}
      destroyOnClose
    >
      <div className="bg-gray-900 text-gray-100 font-mono text-xs rounded-md p-4 overflow-x-auto leading-relaxed">
        <div>
          <span className="text-green-400">user@laptop</span>
          <span className="text-gray-500">:</span>
          <span className="text-blue-400">~/project</span>
          <span className="text-gray-500">$</span> {cmd}
        </div>

        {hasErrors ? (
          <>
            <div className="mt-3 text-red-400 font-bold">
              ✗ Cannot use MCP server &quot;{serverLabel}&quot;
            </div>
            <div className="mt-2 text-gray-200">
              This MCP server is configured with <span className="text-yellow-300">per-user variables</span>{" "}
              that you haven&apos;t set yet. The following user field
              {missing.length === 1 ? " is" : "s are"} missing:
            </div>
            <ul className="ml-4 mt-2 space-y-1">
              {missing.map((m) => (
                <li key={m} className="text-yellow-300">
                  • {"${"}
                  {m}
                  {"}"}
                </li>
              ))}
            </ul>
            <div className="mt-4 text-gray-200">
              Set these values in your LiteLLM dashboard:
            </div>
            <div className="mt-1">
              <a
                href={dashboardLink}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-300 underline break-all hover:text-blue-200"
              >
                {dashboardLink}
              </a>
            </div>
            <div className="mt-4 text-gray-400">
              Once saved, retry: <span className="text-white">{cmd}</span>
            </div>
          </>
        ) : (
          <>
            <div className="mt-3 text-green-400">
              ✓ Connected to MCP server &quot;{serverLabel}&quot;
            </div>
            <div className="mt-2 text-gray-300">Available tools:</div>
            <ul className="ml-4 mt-1 text-gray-200 space-y-0.5">
              <li>• list_tables</li>
              <li>• query_database</li>
              <li>• get_schema</li>
            </ul>
            <div className="mt-4 text-gray-400">[executing list_tables]</div>
            <div className="mt-1 text-gray-200">→ users, orders, products, transactions, sessions</div>
            <div className="mt-3 text-green-400">✓ Done.</div>
          </>
        )}
      </div>
      <AntdText type="secondary" className="text-xs block mt-3">
        This terminal is a simulation for the UI prototype. In production, this same error would be
        emitted by the real <code>claude</code> CLI when calling an MCP through the LiteLLM gateway.
      </AntdText>
    </Modal>
  );
};

export default MCPClaudeCodeSimulator;
