"use client";
import React, { useState } from "react";
import { ColumnDef, Row } from "@tanstack/react-table";
import { DataTable } from "./view_logs/table";
import { Select, SelectItem } from "@tremor/react"
import { Button } from "@tremor/react"
import KeyInfoView from "./key_info_view";
import { Tooltip } from "antd";
import { Team, KeyResponse } from "./key_team_helpers/key_list";

interface AllKeysTableProps {
  keys: KeyResponse[];
  isLoading?: boolean;
  pagination: {
    currentPage: number;
    totalPages: number;
    totalCount: number;
  };
  onPageChange: (page: number) => void;
  pageSize?: number;
  teams: Team[] | null;
  selectedTeam: Team | null;
  setSelectedTeam: (team: Team | null) => void;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

// Define columns similar to our logs table

function KeyViewer({ row }: { row: Row<KeyResponse> }) {
  return (
    <div className="p-4 bg-gray-50">
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-medium">Key Details</h3>
        <div className="mt-2 space-y-1">
          <p>
            <strong>Key Alias:</strong> {row.original.key_alias || "Not Set"}
          </p>
          <p>
            <strong>Secret Key:</strong> {row.original.key_name}
          </p>
          <p>
            <strong>Created:</strong>{" "}
            {new Date(row.original.created_at).toLocaleString()}
          </p>
          <p>
            <strong>Expires:</strong>{" "}
            {row.original.expires
              ? new Date(row.original.expires).toLocaleString()
              : "Never"}
          </p>
          <p>
            <strong>Spend:</strong> {Number(row.original.spend).toFixed(4)}
          </p>
          <p>
            <strong>Budget:</strong>{" "}
            {row.original.max_budget !== null
              ? row.original.max_budget
              : "Unlimited"}
          </p>
          <p>
            <strong>Budget Reset:</strong>{" "}
            {row.original.budget_reset_at
              ? new Date(row.original.budget_reset_at).toLocaleString()
              : "Never"}
          </p>
          <p>
            <strong>Models:</strong>{" "}
            {row.original.models && row.original.models.length > 0
              ? row.original.models.join(", ")
              : "-"}
          </p>
          <p>
            <strong>Rate Limits:</strong> TPM:{" "}
            {row.original.tpm_limit !== null
              ? row.original.tpm_limit
              : "Unlimited"}
            , RPM:{" "}
            {row.original.rpm_limit !== null
              ? row.original.rpm_limit
              : "Unlimited"}
          </p>
          <p>
            <strong>Metadata:</strong>
          </p>
          <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto">
            {JSON.stringify(row.original.metadata, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}

const TeamFilter = ({ 
  teams, 
  selectedTeam, 
  setSelectedTeam 
}: { 
  teams: Team[] | null;
  selectedTeam: Team | null;
  setSelectedTeam: (team: Team | null) => void;
}) => {
    const handleTeamChange = (value: string) => {
      const team = teams?.find(t => t.team_id === value);
      setSelectedTeam(team || null);
    };
  
    return (
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Where Team is</span>
          <Select
            value={selectedTeam?.team_id || ""}
            onValueChange={handleTeamChange}
            placeholder="Team ID"
            className="w-[400px]"
          >
            <SelectItem value="team_id">Team ID</SelectItem>
            {teams?.map((team) => (
              <SelectItem key={team.team_id} value={team.team_id}>
                <span className="font-medium">{team.team_alias}</span>{" "}
                <span className="text-gray-500">({team.team_id})</span>
              </SelectItem>
            ))}
          </Select>
        </div>
      </div>
    );
  };
  

/**
 * AllKeysTable – a new table for keys that mimics the table styling used in view_logs.
 * The team selector and filtering have been removed so that all keys are shown.
 */
export function AllKeysTable({ 
  keys, 
  isLoading = false,
  pagination,
  onPageChange,
  pageSize = 50,
  teams,
  selectedTeam,
  setSelectedTeam,
  accessToken,
  userID,
  userRole
}: AllKeysTableProps) {
  const [selectedKeyId, setSelectedKeyId] = useState<string | null>(null);

  const columns: ColumnDef<KeyResponse>[] = [
    {
      id: "expander",
      header: () => null,
      cell: ({ row }) =>
        row.getCanExpand() ? (
          <button
            onClick={row.getToggleExpandedHandler()}
            style={{ cursor: "pointer" }}
          >
            {row.getIsExpanded() ? "▼" : "▶"}
          </button>
        ) : null,
    },
    {
      header: "Key ID",
      accessorKey: "token",
      cell: (info) => (
        <div className="overflow-hidden">
          <Tooltip title={info.getValue() as string}>
            <Button 
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => setSelectedKeyId(info.getValue() as string)}
            >
              {info.getValue() ? `${(info.getValue() as string).slice(0, 7)}...` : "Not Set"}
            </Button>
          </Tooltip>
        </div>
      ),
    },
    {
      header: "Organization",
      accessorKey: "organization_id",
      cell: (info) => info.getValue() ? info.renderValue() : "Not Set",
    },
    {
      header: "Team ID",
      accessorKey: "team_id",
      cell: (info) => info.getValue() ? info.renderValue() : "Not Set",
    },
    {
      header: "Key Alias",
      accessorKey: "key_alias",
      cell: (info) => info.getValue() ? info.renderValue() : "Not Set",
    },
    {
      header: "Secret Key",
      accessorKey: "key_name",
      cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
    },
    {
      header: "Created",
      accessorKey: "created_at",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "-";
      },
    },
    {
      header: "Expires",
      accessorKey: "expires",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      cell: (info) => Number(info.getValue()).toFixed(4),
    },
    {
      header: "Budget (USD)",
      accessorKey: "max_budget",
      cell: (info) =>
        info.getValue() !== null && info.getValue() !== undefined
          ? info.getValue()
          : "Unlimited",
    },
    {
      header: "Budget Reset",
      accessorKey: "budget_reset_at",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleString() : "Never";
      },
    },
    {
      header: "Models",
      accessorKey: "models",
      cell: (info) => {
        const models = info.getValue() as string[];
        return (
          <div className="flex flex-wrap gap-1">
            {models && models.length > 0 ? (
              models.map((model, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-blue-100 rounded text-xs"
                >
                  {model}
                </span>
              ))
            ) : (
              "-"
            )}
          </div>
        );
      },
    },
    {
      header: "Rate Limits",
      cell: ({ row }) => {
        const key = row.original;
        return (
          <div>
            <div>TPM: {key.tpm_limit !== null ? key.tpm_limit : "Unlimited"}</div>
            <div>RPM: {key.rpm_limit !== null ? key.rpm_limit : "Unlimited"}</div>
          </div>
        );
      },
    },
  ];
  
  return (
    <div className="w-full">
      {selectedKeyId ? (
        <KeyInfoView 
          keyId={selectedKeyId} 
          onClose={() => setSelectedKeyId(null)}
          keyData={keys.find(k => k.token === selectedKeyId)}
          accessToken={accessToken}
          userID={userID}
          userRole={userRole}
          teams={teams}
        />
      ) : (
        <div className="border-b py-4">
          <div className="flex items-center justify-between w-full">
            <TeamFilter 
              teams={teams} 
              selectedTeam={selectedTeam} 
              setSelectedTeam={setSelectedTeam} 
            />
            <div className="flex items-center gap-4">
              <span className="inline-flex text-sm text-gray-700">
                Showing {isLoading ? "..." : `${(pagination.currentPage - 1) * pageSize + 1} - ${Math.min(pagination.currentPage * pageSize, pagination.totalCount)}`} of {isLoading ? "..." : pagination.totalCount} results
              </span>
              
              <div className="inline-flex items-center gap-2">
                <span className="text-sm text-gray-700">
                  Page {isLoading ? "..." : pagination.currentPage} of {isLoading ? "..." : pagination.totalPages}
                </span>
                
                <button
                  onClick={() => onPageChange(pagination.currentPage - 1)}
                  disabled={isLoading || pagination.currentPage === 1}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                
                <button
                  onClick={() => onPageChange(pagination.currentPage + 1)} 
                  disabled={isLoading || pagination.currentPage === pagination.totalPages}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
          <DataTable
            columns={columns.filter(col => col.id !== 'expander')}
            data={keys}
            isLoading={isLoading}
            getRowCanExpand={() => false}
            renderSubComponent={() => <></>}
          />
        </div>
        
      )}
      
    </div>
  );
}