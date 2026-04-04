import { UserOutlined, KeyOutlined, TeamOutlined } from "@ant-design/icons";
import React from "react";

interface MCPToolUserEntry {
  tool_name: string;
  api_key_hash?: string | null;
  api_key_alias?: string | null;
  user_id?: string | null;
  team_id?: string | null;
  call_count: number;
  last_called: string;
}

interface MCPToolUsersTableProps {
  entries: MCPToolUserEntry[];
  total: number;
}

export function MCPToolUsersTable({ entries, total }: MCPToolUsersTableProps) {
  if (entries.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg py-12 text-center text-sm text-gray-500">
        No tool usage data for this period.
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-base font-semibold text-gray-900">
          Users &amp; Keys per Tool
        </h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Which users and API keys called which tools — {total} entries
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
            <tr>
              <th className="px-4 py-2 font-medium">Tool</th>
              <th className="px-4 py-2 font-medium">User</th>
              <th className="px-4 py-2 font-medium">API Key</th>
              <th className="px-4 py-2 font-medium">Team</th>
              <th className="px-4 py-2 font-medium text-right">Calls</th>
              <th className="px-4 py-2 font-medium">Last Called</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {entries.map((entry, idx) => (
              <tr
                key={`${entry.tool_name}-${entry.api_key_hash}-${entry.user_id}-${idx}`}
                className="hover:bg-gray-50"
              >
                <td className="px-4 py-2">
                  <span className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded bg-blue-50 text-blue-700 border border-blue-200">
                    {entry.tool_name}
                  </span>
                </td>
                <td className="px-4 py-2 text-gray-700">
                  {entry.user_id ? (
                    <span className="inline-flex items-center gap-1">
                      <UserOutlined className="text-[10px] text-gray-400" />
                      {entry.user_id}
                    </span>
                  ) : (
                    <span className="text-gray-400">—</span>
                  )}
                </td>
                <td className="px-4 py-2 text-gray-700">
                  {entry.api_key_alias || entry.api_key_hash ? (
                    <span className="inline-flex items-center gap-1">
                      <KeyOutlined className="text-[10px] text-gray-400" />
                      {entry.api_key_alias ||
                        (entry.api_key_hash
                          ? `${entry.api_key_hash.slice(0, 12)}...`
                          : "")}
                    </span>
                  ) : (
                    <span className="text-gray-400">—</span>
                  )}
                </td>
                <td className="px-4 py-2 text-gray-700">
                  {entry.team_id ? (
                    <span className="inline-flex items-center gap-1">
                      <TeamOutlined className="text-[10px] text-gray-400" />
                      {entry.team_id}
                    </span>
                  ) : (
                    <span className="text-gray-400">—</span>
                  )}
                </td>
                <td className="px-4 py-2 text-right font-medium text-gray-900">
                  {entry.call_count.toLocaleString()}
                </td>
                <td className="px-4 py-2 text-xs text-gray-500">
                  {entry.last_called}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
