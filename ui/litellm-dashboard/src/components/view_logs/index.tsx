import moment from "moment";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { uiSpendLogsCall } from "../networking";
import { DataTable } from "./table";
import { columns, LogEntry } from "./columns";
import { Row } from "@tanstack/react-table";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

export default function SpendLogsTable({
  accessToken,
  token,
  userRole,
  userID,
}: SpendLogsTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [keyNameFilter, setKeyNameFilter] = useState("");
  const [teamNameFilter, setTeamNameFilter] = useState("");

  const logs = useQuery({
    queryKey: ["logs", "table"],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        console.log(
          "got None values for one of accessToken, token, userRole, userID",
        );
        return;
      }

      // Get logs for last 24 hours using ISO string and proper date formatting
      const endTime = moment().format("YYYY-MM-DD HH:mm:ss");
      const startTime = moment()
        .subtract(24, "hours")
        .format("YYYY-MM-DD HH:mm:ss");

      const data = await uiSpendLogsCall(
        accessToken,
        token,
        userRole,
        userID,
        startTime,
        endTime,
      );

      return data;
    },
  });

  if (!accessToken || !token || !userRole || !userID) {
    console.log(
      "got None values for one of accessToken, token, userRole, userID",
    );
    return null;
  }

  const filteredData = logs.data?.filter((log) => {
    const matchesSearch =
      !searchTerm ||
      log.request_id.includes(searchTerm) ||
      log.model.includes(searchTerm) ||
      (log.user && log.user.includes(searchTerm));
    const matchesKeyName =
      !keyNameFilter ||
      (log.metadata?.user_api_key_alias &&
        log.metadata.user_api_key_alias
          .toLowerCase()
          .includes(keyNameFilter.toLowerCase()));
    const matchesTeamName =
      !teamNameFilter ||
      (log.metadata?.user_api_key_team_alias &&
        log.metadata.user_api_key_team_alias
          .toLowerCase()
          .includes(teamNameFilter.toLowerCase()));
    return matchesSearch && matchesKeyName && matchesTeamName;
  }) || [];

  return (
    <div className="px-4 md:px-8 py-8 w-full">
      <h1 className="text-xl font-semibold mb-4">Traces</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b flex flex-wrap gap-3 items-center">
          <div className="flex items-center gap-2 flex-1">
            <select className="px-4 py-2 border rounded-lg min-w-[120px]">
              <option>where</option>
            </select>
            <select className="px-4 py-2 border rounded-lg min-w-[150px]">
              <option>Key Name</option>
              <option>Team Name</option>
            </select>
            <select className="px-4 py-2 border rounded-lg min-w-[120px]">
              <option>equals</option>
              <option>contains</option>
            </select>
            <input
              type="text"
              placeholder="Value..."
              className="px-4 py-2 border rounded-lg flex-1"
              value={keyNameFilter}
              onChange={(e) => setKeyNameFilter(e.target.value)}
            />
            <button className="px-3 py-1 text-sm border rounded-lg hover:bg-gray-50">
              ×
            </button>
          </div>
          <button className="px-4 py-2 text-sm border rounded-lg hover:bg-gray-50">
            + Add filter
          </button>
          <button className="px-4 py-2 text-sm border rounded-lg hover:bg-gray-50">
            Clear filters
          </button>
          <div className="flex-1 md:flex-none flex justify-end gap-2">
            <select className="px-4 py-2 border rounded-lg">
              <option>Last 24 hours</option>
              <option>Last 7 days</option>
              <option>Last 30 days</option>
            </select>
            <button className="px-3 py-1 text-sm border rounded-lg hover:bg-gray-50">
              Export
            </button>
          </div>
        </div>
        <DataTable
          columns={columns}
          data={filteredData}
          renderSubComponent={RequestViewer}
          getRowCanExpand={() => true}
        />
      </div>
    </div>
  );
}

function RequestViewer({ row }: { row: Row<LogEntry> }) {
  const formatData = (input: any) => {
    if (typeof input === "string") {
      try {
        return JSON.parse(input);
      } catch {
        return input;
      }
    }
    return input;
  };

  return (
    <div className="p-6 bg-gray-50 space-y-6">
      {/* Request Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.request_id), null, 2)}
        </pre>
      </div>

      {/* Response Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Response</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.response), null, 2)}
        </pre>
      </div>

      {/* Metadata Card */}
      {row.original.metadata &&
        Object.keys(row.original.metadata).length > 0 && (
          <div className="bg-white rounded-lg shadow">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-medium">Metadata</h3>
              <div>
                <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                  JSON
                </button>
              </div>
            </div>
            <pre className="p-4 overflow-auto text-sm">
              {JSON.stringify(row.original.metadata, null, 2)}
            </pre>
          </div>
        )}
    </div>
  );
}
