// PROTOTYPE: cell rendered in the MCP servers table. Shows a red "N user
// fields missing" pill plus quick-action buttons when the current user hasn't
// filled in their per-user fields, otherwise shows a green "Ready" pill.

import React, { useEffect, useState } from "react";
import { Tooltip } from "antd";
import {
  ExclamationCircleFilled,
  CheckCircleFilled,
  PlayCircleOutlined,
} from "@ant-design/icons";
import {
  getEnvVarDefinitions,
  getMissingUserFields,
  subscribeEnvVarsChanged,
} from "./mockMcpEnvVars";

interface UserFieldsStatusCellProps {
  serverAlias: string;
  userId: string;
  onOpenFill: () => void;
  onOpenDemo: () => void;
}

const UserFieldsStatusCell: React.FC<UserFieldsStatusCellProps> = ({
  serverAlias,
  userId,
  onOpenFill,
  onOpenDemo,
}) => {
  const [tick, setTick] = useState(0);
  useEffect(() => subscribeEnvVarsChanged(() => setTick((t) => t + 1)), []);

  const defs = serverAlias ? getEnvVarDefinitions(serverAlias) : [];
  const perUserCount = defs.filter((d) => d.scope === "per_user").length;
  const missing = serverAlias ? getMissingUserFields(serverAlias, userId) : [];

  // Server has no per-user fields at all → no badge to show.
  if (perUserCount === 0) {
    return (
      <button
        onClick={(e) => {
          e.stopPropagation();
          onOpenDemo();
        }}
        className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-blue-600 transition-colors"
        title="Simulate using this MCP server in Claude Code"
      >
        <PlayCircleOutlined /> Try in Claude Code
      </button>
    );
  }

  if (missing.length > 0) {
    return (
      <div
        className="inline-flex items-center gap-2 px-2 py-1 rounded-md border-2 border-red-300 bg-red-50"
        // PROTOTYPE: cell-level highlight stands in for full-row highlight
        // since DataTable doesn't expose a row-class hook today.
      >
        <Tooltip
          title={
            <div>
              <div className="font-semibold mb-1">Missing user fields:</div>
              <ul className="ml-3">
                {missing.map((m) => (
                  <li key={m}>• {m}</li>
                ))}
              </ul>
            </div>
          }
        >
          <span className="inline-flex items-center gap-1 text-xs font-semibold text-red-700">
            <ExclamationCircleFilled />
            {missing.length} user field{missing.length === 1 ? "" : "s"} missing
          </span>
        </Tooltip>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onOpenFill();
          }}
          className="text-xs bg-red-600 hover:bg-red-700 text-white px-2 py-0.5 rounded font-medium transition-colors"
        >
          Set
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onOpenDemo();
          }}
          className="text-xs text-gray-600 hover:text-blue-600 transition-colors"
          title="Simulate using this MCP server in Claude Code"
        >
          <PlayCircleOutlined />
        </button>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2">
      <Tooltip title="All per-user fields are set for your account.">
        <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full">
          <CheckCircleFilled /> Ready
        </span>
      </Tooltip>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onOpenFill();
        }}
        className="text-xs text-gray-400 hover:text-blue-600 transition-colors"
      >
        Update
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onOpenDemo();
        }}
        className="text-xs text-gray-600 hover:text-blue-600 transition-colors"
        title="Simulate using this MCP server in Claude Code"
      >
        <PlayCircleOutlined />
      </button>
    </div>
  );
};

export default UserFieldsStatusCell;
