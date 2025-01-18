import moment from "moment";
import { useQuery } from "@tanstack/react-query";

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

  return (
    <div className="px-4 md:px-8 py-8 w-full">
      <h1 className="text-xl font-semibold mb-4">Traces</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b flex justify-between items-center">
          <div className="flex space-x-4 items-center">
            <input
              type="text"
              placeholder="Search by request ID, model, or user..."
              className="px-4 py-2 border rounded-lg w-80"
            />
            <button className="px-4 py-2 border rounded-lg flex items-center gap-2">
              <span>Filters</span>
              <span className="text-xs bg-gray-100 px-2 py-1 rounded">0</span>
            </button>
            <select className="px-4 py-2 border rounded-lg">
              <option>Last 24 hours</option>
              <option>Last 7 days</option>
              <option>Last 30 days</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Export
            </button>
          </div>
        </div>
        <DataTable
          columns={columns}
          data={logs.data}
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
