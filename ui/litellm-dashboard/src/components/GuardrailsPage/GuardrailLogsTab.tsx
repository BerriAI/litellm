import React, { useEffect, useState } from "react";
import {
  Card,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Text,
  Badge,
  Button,
} from "@tremor/react";
import { guardrailLogsCall } from "../networking";
import type { GuardrailLogEntry } from "./types";
import { mockGuardrailLogs } from "./mockData";

// Toggle this to use mock data vs real API
const USE_MOCK_DATA = true;

interface GuardrailLogsTabProps {
  accessToken: string;
  guardrailName: string;
  startDate: string;
  endDate: string;
}

const GuardrailLogsTab: React.FC<GuardrailLogsTabProps> = ({
  accessToken,
  guardrailName,
  startDate,
  endDate,
}) => {
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<GuardrailLogEntry[]>([]);
  const [statusFilter, setStatusFilter] = useState<string | undefined>(
    undefined
  );
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLogs();
  }, [guardrailName, startDate, endDate, statusFilter]);

  const fetchLogs = async () => {
    if (USE_MOCK_DATA) {
      // Use mock data for development
      setLoading(true);
      setTimeout(() => {
        let filteredLogs = mockGuardrailLogs;
        if (statusFilter) {
          filteredLogs = mockGuardrailLogs.filter(
            (log) => log.status === statusFilter
          );
        }
        setLogs(filteredLogs);
        setLoading(false);
      }, 500);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await guardrailLogsCall(
        accessToken,
        guardrailName,
        startDate,
        endDate,
        statusFilter
      );
      setLogs(response.logs);
    } catch (error: any) {
      console.error("Error fetching guardrail logs:", error);
      setError(error.message || "Failed to fetch guardrail logs");
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (requestId: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(requestId)) {
      newExpanded.delete(requestId);
    } else {
      newExpanded.add(requestId);
    }
    setExpandedRows(newExpanded);
  };

  return (
    <Card className="mt-6">
      <div className="flex justify-between items-center mb-4">
        <Text className="text-lg font-semibold">Logs — {guardrailName}</Text>
        <div className="flex gap-2">
          <Button
            size="xs"
            variant={statusFilter === undefined ? "primary" : "secondary"}
            onClick={() => setStatusFilter(undefined)}
          >
            All
          </Button>
          <Button
            size="xs"
            variant={statusFilter === "blocked" ? "primary" : "secondary"}
            onClick={() => setStatusFilter("blocked")}
          >
            Blocked
          </Button>
          <Button
            size="xs"
            variant={statusFilter === "passed" ? "primary" : "secondary"}
            onClick={() => setStatusFilter("passed")}
          >
            Passed
          </Button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 text-red-700 rounded">
          {error}
        </div>
      )}

      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Status</TableHeaderCell>
            <TableHeaderCell>Timestamp</TableHeaderCell>
            <TableHeaderCell>Model</TableHeaderCell>
            <TableHeaderCell>Request</TableHeaderCell>
            <TableHeaderCell className="text-right">Latency</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                Loading...
              </TableCell>
            </TableRow>
          ) : logs.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                No logs found
              </TableCell>
            </TableRow>
          ) : (
            logs.map((log) => (
              <React.Fragment key={log.request_id}>
                <TableRow
                  onClick={() => toggleExpand(log.request_id)}
                  className="cursor-pointer hover:bg-gray-50"
                >
                  <TableCell>
                    <Badge color={log.status === "blocked" ? "red" : "green"}>
                      {log.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Text className="text-sm">
                      {new Date(log.timestamp).toLocaleString()}
                    </Text>
                  </TableCell>
                  <TableCell>
                    <Text>{log.model}</Text>
                  </TableCell>
                  <TableCell>
                    <Text className="truncate max-w-md">
                      {log.request_content || "N/A"}
                    </Text>
                  </TableCell>
                  <TableCell className="text-right">
                    <Text>{Math.round(log.latency_ms)} ms</Text>
                  </TableCell>
                </TableRow>
                {expandedRows.has(log.request_id) && (
                  <TableRow>
                    <TableCell colSpan={5} className="bg-gray-50">
                      <div className="p-4">
                        <Text className="font-semibold mb-2">
                          Guardrail Response:
                        </Text>
                        <pre className="text-xs bg-white p-2 rounded border overflow-auto max-h-96">
                          {JSON.stringify(log.guardrail_response, null, 2)}
                        </pre>
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </React.Fragment>
            ))
          )}
        </TableBody>
      </Table>
    </Card>
  );
};

export default GuardrailLogsTab;
