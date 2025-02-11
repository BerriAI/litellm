import { Row } from "@tanstack/react-table";
import { LogEntry } from "./columns";

export function RequestViewer({ row }: { row: Row<LogEntry> }) {
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
        {/* Combined Info Card */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h3 className="text-lg font-medium ">Request Details</h3>
          </div>
          <div className="space-y-2 p-4 ">
            <div className="flex">
              <span className="font-medium w-1/3">Request ID:</span>
              <span>{row.original.request_id}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Api Key:</span>
              <span>{row.original.api_key}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Team ID:</span>
              <span>{row.original.team_id}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Model:</span>
              <span>{row.original.model}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Api Base:</span>
              <span>{row.original.api_base}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Call Type:</span>
              <span>{row.original.call_type}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Spend:</span>
              <span>{row.original.spend}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Total Tokens:</span>
              <span>{row.original.total_tokens}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Prompt Tokens:</span>
              <span>{row.original.prompt_tokens}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Completion Tokens:</span>
              <span>{row.original.completion_tokens}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Start Time:</span>
              <span>{row.original.startTime}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">End Time:</span>
              <span>{row.original.endTime}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Cache Hit:</span>
              <span>{row.original.cache_hit}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Cache Key:</span>
              <span>{row.original.cache_key}</span>
            </div>
            {row?.original?.requester_ip_address && (
              <div className="flex">
                <span className="font-medium w-1/3">Request IP Address:</span>
                <span>{row?.original?.requester_ip_address}</span>
              </div>
            )}
          </div>
        </div>
  
        {/* Request Card */}
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Request Tags</h3>
          </div>
          <pre className="p-4  text-wrap overflow-auto text-sm">
            {JSON.stringify(formatData(row.original.request_tags), null, 2)}
          </pre>
        </div>
  
        {/* Request Card */}
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Request</h3>
            {/* <div>
              <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
                Expand
              </button>
              <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                JSON
              </button>
            </div> */}
          </div>
          <pre className="p-4  text-wrap overflow-auto text-sm">
            {JSON.stringify(formatData(row.original.messages), null, 2)}
          </pre>
        </div>
  
        {/* Response Card */}
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Response</h3>
            <div>
              {/* <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
                Expand
              </button>
              <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                JSON
              </button> */}
            </div>
          </div>
          <pre className="p-4 text-wrap overflow-auto text-sm">
            {JSON.stringify(formatData(row.original.response), null, 2)}
          </pre>
        </div>
  
        {/* Metadata Card */}
        {row.original.metadata &&
          Object.keys(row.original.metadata).length > 0 && (
            <div className="bg-white rounded-lg shadow">
              <div className="flex justify-between items-center p-4 border-b">
                <h3 className="text-lg font-medium">Metadata</h3>
                {/* <div>
                  <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                    JSON
                  </button>
                </div> */}
              </div>
              <pre className="p-4 text-wrap  overflow-auto text-sm ">
                {JSON.stringify(row.original.metadata, null, 2)}
              </pre>
            </div>
          )}
      </div>
    );
  }
  