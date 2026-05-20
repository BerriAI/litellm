// PROTOTYPE: simulates what a user would see in Claude Code when they try to
// use the MCP server. Shows a friendly error when per-user fields are missing
// (with a deep link to the fill-credentials modal), and a success state once
// the credentials are saved.

import React, { useMemo } from "react";
import { Modal, Typography, Tag } from "antd";
import {
  CheckCircleFilled,
  CloseCircleFilled,
  LinkOutlined,
} from "@ant-design/icons";
import { Button } from "@tremor/react";
import { getMissingUserFields } from "./mockMcpEnvVars";

const { Text, Title, Paragraph } = Typography;

interface MockClaudeCodeModalProps {
  open: boolean;
  serverAlias: string;
  serverName?: string | null;
  userId: string;
  onClose: () => void;
  onOpenFillModal: () => void;
}

const MockClaudeCodeModal: React.FC<MockClaudeCodeModalProps> = ({
  open,
  serverAlias,
  serverName,
  userId,
  onClose,
  onOpenFillModal,
}) => {
  // Recomputed on each render — fine for a modal that only re-mounts on open.
  const missing = useMemo(
    () => (open ? getMissingUserFields(serverAlias, userId) : []),
    [open, serverAlias, userId],
  );

  const hasMissing = missing.length > 0;

  const fillUrl = useMemo(() => {
    if (typeof window === "undefined") return "";
    const url = new URL(window.location.href);
    url.searchParams.set("fill_fields", serverAlias);
    return url.toString();
  }, [serverAlias]);

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={620}
      title={
        <div className="flex items-center gap-2">
          <Title level={5} style={{ margin: 0 }}>
            Simulated Claude Code session
          </Title>
          <Tag color="purple">Prototype</Tag>
        </div>
      }
    >
      <div className="mt-2 rounded-lg bg-[#1e1e1e] text-gray-100 font-mono text-sm p-4 leading-relaxed">
        <div className="text-gray-400">
          $ claude
          <br />
          &gt; Using MCP server <span className="text-cyan-300">{serverName || serverAlias}</span>...
        </div>

        {hasMissing ? (
          <div className="mt-4">
            <div className="flex items-start gap-2">
              <CloseCircleFilled className="text-red-400 mt-1" />
              <div>
                <div className="text-red-300 font-semibold">
                  Cannot connect to MCP server &quot;{serverName || serverAlias}&quot;
                </div>
                <div className="text-gray-300 mt-2">
                  Your administrator configured this server to require per-user
                  credentials, but you haven&apos;t set the following yet:
                </div>
                <ul className="mt-2 ml-4 text-yellow-200">
                  {missing.map((name) => (
                    <li key={name}>
                      • <span className="font-bold">{name}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-3 text-gray-300">
                  Set your credentials here:
                  <br />
                  <span className="text-blue-300 underline break-all">
                    <LinkOutlined /> {fillUrl}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 flex items-start gap-2">
            <CheckCircleFilled className="text-green-400 mt-1" />
            <div>
              <div className="text-green-300 font-semibold">
                Connected to {serverName || serverAlias}
              </div>
              <div className="text-gray-300 mt-2">
                MCP server tools are now available. Try asking Claude to use
                them.
              </div>
            </div>
          </div>
        )}
      </div>

      <Paragraph type="secondary" className="text-xs mt-3 mb-0">
        This panel mocks what a user would see in their MCP client (Claude Code,
        Cursor, etc). In a real implementation, the error would arrive over the
        MCP protocol with the same deep link.
      </Paragraph>

      <div className="flex items-center justify-end gap-2 pt-4 mt-3 border-t border-gray-100">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
        {hasMissing && (
          <Button
            variant="primary"
            onClick={() => {
              onClose();
              onOpenFillModal();
            }}
          >
            Set Credentials
          </Button>
        )}
      </div>
    </Modal>
  );
};

export default MockClaudeCodeModal;
