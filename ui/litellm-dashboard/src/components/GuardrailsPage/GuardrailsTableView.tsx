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
} from "@tremor/react";
import { useRouter } from "next/navigation";
import { guardrailMetricsCall } from "../networking";
import type { GuardrailSummary } from "./types";
import { mockGuardrailSummaries } from "./mockData";

// Toggle this to use mock data vs real API
const USE_MOCK_DATA = true;

interface GuardrailsTableViewProps {
  accessToken: string;
  startDate: string;
  endDate: string;
}

const GuardrailsTableView: React.FC<GuardrailsTableViewProps> = ({
  accessToken,
  startDate,
  endDate,
}) => {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [guardrailData, setGuardrailData] = useState<GuardrailSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGuardrailMetrics();
  }, [startDate, endDate]);

  const fetchGuardrailMetrics = async () => {
    if (USE_MOCK_DATA) {
      // Use mock data for development
      setLoading(true);
      setTimeout(() => {
        setGuardrailData(mockGuardrailSummaries);
        setLoading(false);
      }, 500);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await guardrailMetricsCall(
        accessToken,
        startDate,
        endDate
      );
      setGuardrailData(response.results);
    } catch (error: any) {
      console.error("Error fetching guardrail metrics:", error);
      setError(error.message || "Failed to fetch guardrail metrics");
    } finally {
      setLoading(false);
    }
  };

  const handleRowClick = (guardrailName: string) => {
    router.push(`/guardrails/metrics/${encodeURIComponent(guardrailName)}`);
  };

  return (
    <Card>
      {error && (
        <div className="mb-4 p-4 bg-red-50 text-red-700 rounded">
          {error}
        </div>
      )}
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Guardrail Name</TableHeaderCell>
            <TableHeaderCell>Provider</TableHeaderCell>
            <TableHeaderCell className="text-right">Requests</TableHeaderCell>
            <TableHeaderCell className="text-right">Fail Rate</TableHeaderCell>
            <TableHeaderCell className="text-right">
              Avg Latency
            </TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                Loading...
              </TableCell>
            </TableRow>
          ) : guardrailData.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                No guardrail data available for selected date range
              </TableCell>
            </TableRow>
          ) : (
            guardrailData.map((guardrail) => (
              <TableRow
                key={guardrail.guardrail_name}
                onClick={() => handleRowClick(guardrail.guardrail_name)}
                className="cursor-pointer hover:bg-gray-50"
              >
                <TableCell>
                  <Text className="font-medium">
                    {guardrail.guardrail_name}
                  </Text>
                </TableCell>
                <TableCell>
                  <Text className="capitalize">{guardrail.provider}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{guardrail.total_requests.toLocaleString()}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text
                    className={
                      guardrail.fail_rate > 10
                        ? "text-red-600 font-semibold"
                        : guardrail.fail_rate > 5
                        ? "text-yellow-600"
                        : ""
                    }
                  >
                    {guardrail.fail_rate.toFixed(2)}%
                  </Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{Math.round(guardrail.avg_latency_ms)} ms</Text>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </Card>
  );
};

export default GuardrailsTableView;
