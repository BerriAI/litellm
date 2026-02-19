import React, { useEffect, useState } from "react";
import {
  Card,
  Title,
  Metric,
  Text,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  AreaChart,
  Grid,
  Col,
} from "@tremor/react";
import { guardrailDetailMetricsCall } from "../networking";
import type { GuardrailDetailMetrics } from "./types";
import GuardrailLogsTab from "./GuardrailLogsTab";
import { mockGuardrailDetailMetrics } from "./mockData";

// Toggle this to use mock data vs real API
const USE_MOCK_DATA = true;

interface GuardrailDetailViewProps {
  accessToken: string;
  guardrailName: string;
  startDate: string;
  endDate: string;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({
  accessToken,
  guardrailName,
  startDate,
  endDate,
}) => {
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<GuardrailDetailMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMetrics();
  }, [guardrailName, startDate, endDate]);

  const fetchMetrics = async () => {
    if (USE_MOCK_DATA) {
      // Use mock data for development
      setLoading(true);
      setTimeout(() => {
        setMetrics(mockGuardrailDetailMetrics);
        setLoading(false);
      }, 500);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await guardrailDetailMetricsCall(
        accessToken,
        guardrailName,
        startDate,
        endDate
      );
      setMetrics(data);
    } catch (error: any) {
      console.error("Error fetching guardrail details:", error);
      setError(error.message || "Failed to fetch guardrail details");
    } finally {
      setLoading(false);
    }
  };

  if (loading || !metrics) {
    return <div className="p-8">Loading...</div>;
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="p-4 bg-red-50 text-red-700 rounded">{error}</div>
      </div>
    );
  }

  return (
    <div>
      {/* Metric Cards */}
      <Grid numItemsMd={2} numItemsLg={4} className="gap-6 mb-6">
        <Card>
          <Text>Requests Evaluated</Text>
          <Metric>{metrics.requests_evaluated.toLocaleString()}</Metric>
        </Card>
        <Card>
          <Text>Fail Rate</Text>
          <Metric className={metrics.fail_rate > 10 ? "text-red-600" : ""}>
            {metrics.fail_rate.toFixed(2)}%
          </Metric>
        </Card>
        <Card>
          <Text>Avg Latency</Text>
          <Metric>{Math.round(metrics.avg_latency_ms)} ms</Metric>
        </Card>
        <Card>
          <Text>Blocked (Period)</Text>
          <Metric>{metrics.blocked_count.toLocaleString()}</Metric>
        </Card>
      </Grid>

      {/* Tabs */}
      <TabGroup>
        <TabList>
          <Tab>Overview</Tab>
          <Tab>Logs</Tab>
        </TabList>
        <TabPanels>
          {/* Overview Tab */}
          <TabPanel>
            <Card className="mt-6">
              <Title>Fail Rate Trend</Title>
              <AreaChart
                className="mt-4 h-80"
                data={metrics.daily_metrics}
                index="date"
                categories={["fail_rate"]}
                colors={["red"]}
                valueFormatter={(value) => `${value.toFixed(2)}%`}
                yAxisWidth={60}
              />
            </Card>
            <Card className="mt-6">
              <Title>Daily Metrics</Title>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Date</TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      Requests
                    </TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      Blocked
                    </TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      Passed
                    </TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      Fail Rate
                    </TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      Avg Latency
                    </TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {metrics.daily_metrics.map((daily) => (
                    <TableRow key={daily.date}>
                      <TableCell>
                        <Text>{daily.date}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{daily.total_requests.toLocaleString()}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{daily.intervened_count.toLocaleString()}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{daily.success_count.toLocaleString()}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text
                          className={
                            daily.fail_rate > 10
                              ? "text-red-600 font-semibold"
                              : daily.fail_rate > 5
                              ? "text-yellow-600"
                              : ""
                          }
                        >
                          {daily.fail_rate.toFixed(2)}%
                        </Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{Math.round(daily.avg_latency_ms)} ms</Text>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </TabPanel>

          {/* Logs Tab */}
          <TabPanel>
            <GuardrailLogsTab
              accessToken={accessToken}
              guardrailName={guardrailName}
              startDate={startDate}
              endDate={endDate}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

// Add missing imports
import {
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
} from "@tremor/react";

export default GuardrailDetailView;
