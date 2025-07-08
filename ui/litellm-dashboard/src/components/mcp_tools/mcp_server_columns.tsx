import { ColumnDef } from "@tanstack/react-table";
import { MCPServer, Team } from "./types";
import { Button, Tooltip } from "antd";
import { EyeIcon, PencilIcon, TrashIcon, EyeOffIcon } from "lucide-react";
import { getMaskedAndFullUrl } from "./utils";
import { useState } from "react";

export const mcpServerColumns = (
  userRole: string,
  onView: (serverId: string) => void,
  onEdit: (serverId: string) => void,
  onDelete: (serverId: string) => void
): ColumnDef<MCPServer>[] => [
  {
    id: "alias",
    header: "Alias",
    cell: ({ row }) => {
      const alias = row.original.alias || row.original.server_id;
      return (
        <button
          onClick={() => onView(row.original.server_id)}
          className="text-blue-500 hover:text-blue-700 font-medium"
        >
          {alias}
        </button>
      );
    },
  },
  {
    id: "url",
    header: "URL",
    cell: ({ row }) => {
      const [showFullUrl, setShowFullUrl] = useState(false);
      const { maskedUrl, hasToken } = getMaskedAndFullUrl(row.original.url);
      
      return (
        <div className="flex items-center gap-2 max-w-[400px]">
          <span className="font-mono text-sm break-all">
            {hasToken ? (showFullUrl ? row.original.url : maskedUrl) : row.original.url}
          </span>
          {hasToken && (
            <button
              onClick={() => setShowFullUrl(!showFullUrl)}
              className="p-1 hover:bg-gray-100 rounded flex-shrink-0"
            >
              {showFullUrl ? (
                <EyeOffIcon className="h-4 w-4 text-gray-500" />
              ) : (
                <EyeIcon className="h-4 w-4 text-gray-500" />
              )}
            </button>
          )}
        </div>
      );
    },
  },
  {
    id: "transport",
    header: "Transport",
    accessorFn: (row) => (row.transport || "unknown").toUpperCase(),
  },
  {
    id: "teams",
    header: "Teams",
    accessorFn: (row) => {
      if (!row.teams?.length) return "No teams";
      return row.teams.map((team: Team) => team.team_alias || team.team_id).join(", ");
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => (
      <div className="flex gap-2">
        <Tooltip title="View">
          <Button
            type="text"
            icon={<EyeIcon className="h-4 w-4" />}
            onClick={() => onView(row.original.server_id)}
          />
        </Tooltip>
        <Tooltip title="Edit">
          <Button
            type="text"
            icon={<PencilIcon className="h-4 w-4" />}
            onClick={() => onEdit(row.original.server_id)}
          />
        </Tooltip>
        <Tooltip title="Delete">
          <Button
            type="text"
            danger
            icon={<TrashIcon className="h-4 w-4" />}
            onClick={() => onDelete(row.original.server_id)}
          />
        </Tooltip>
      </div>
    ),
  },
];
