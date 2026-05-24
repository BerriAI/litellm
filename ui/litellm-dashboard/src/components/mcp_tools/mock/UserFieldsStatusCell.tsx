// Cell rendered in the MCP servers table. Counts per-user env-var
// definitions on the server row itself (no HTTP per row) and renders quick
// actions to (a) open the fill modal and (b) open the simulated terminal
// preview.
//
// Whether the caller has *missing* fields requires a per-user lookup, which
// we defer to the FillUserFieldsModal on open. The cell only knows "this
// server has N per-user fields" — the actual missing/ready state surfaces
// in-modal and in the real terminal error.

import React from "react";
import { Tooltip } from "antd";
import {
  ExclamationCircleOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

type EnvVarDef = { name: string; scope: "instance" | "per_user"; value?: string };

interface UserFieldsStatusCellProps {
  envVars: EnvVarDef[] | undefined | null;
  onOpenFill: () => void;
  onOpenDemo: () => void;
}

const UserFieldsStatusCell: React.FC<UserFieldsStatusCellProps> = ({
  envVars,
  onOpenFill,
  onOpenDemo,
}) => {
  const perUserDefs = Array.isArray(envVars)
    ? envVars.filter((d) => d && d.scope === "per_user" && d.name)
    : [];

  if (perUserDefs.length === 0) {
    return (
      <button
        type="button"
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

  return (
    <div className="inline-flex items-center gap-2 px-2 py-1 rounded-md border border-blue-200 bg-blue-50">
      <Tooltip
        title={
          <div>
            <div className="font-semibold mb-1">Per-user fields:</div>
            <ul className="ml-3">
              {perUserDefs.map((d) => (
                <li key={d.name}>• {d.name}</li>
              ))}
            </ul>
            <div className="mt-2 text-xs">
              Click <b>Set</b> to enter your values, or <b>▶</b> to preview the
              terminal error.
            </div>
          </div>
        }
      >
        <span className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700">
          <ExclamationCircleOutlined />
          {perUserDefs.length} per-user field
          {perUserDefs.length === 1 ? "" : "s"}
        </span>
      </Tooltip>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onOpenFill();
        }}
        className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-2 py-0.5 rounded font-medium transition-colors"
      >
        Set
      </button>
      <button
        type="button"
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
