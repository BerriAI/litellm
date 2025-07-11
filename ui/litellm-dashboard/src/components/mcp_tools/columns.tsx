import React from "react";
import { ColumnDef } from "@tanstack/react-table";
import { MCPTool } from "./types";
import { Button } from "@tremor/react";

export const columns: ColumnDef<MCPTool>[] = [
  {
    accessorKey: "mcp_info.server_name",
    header: "Provider",
    cell: ({ row }) => {
      const serverName = row.original.mcp_info.server_name;
      const logoUrl = row.original.mcp_info.logo_url;
      
      return (
        <div className="flex items-center space-x-2">
          {logoUrl && (
            <img 
              src={logoUrl} 
              alt={`${serverName} logo`} 
              className="h-5 w-5 object-contain"
            />
          )}
          <span className="font-medium">{serverName}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "name",
    header: "Tool Name",
    cell: ({ row }) => {
      const name = row.getValue("name") as string;
      return (
        <div>
          <span className="font-mono text-sm">{name}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "description",
    header: "Description",
    cell: ({ row }) => {
      const description = row.getValue("description") as string;
      return (
        <div className="max-w-md">
          <span className="text-sm text-gray-700">{description}</span>
        </div>
      );
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => {
      const tool = row.original;
      
      return (
        <div className="flex items-center space-x-2">
          <Button 
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => {
            if (typeof row.original.onToolSelect === 'function') {
                row.original.onToolSelect(tool);
            }
            }}
          >
            Test Tool
          </Button>
        </div>
      );
    },
  },
];

 