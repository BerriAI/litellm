/**
 * New Usage Page
 * 
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 * 
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import React, { useState, useEffect } from "react";
import { 
  BarChart, Card, Title, Text, 
  Grid, Col, TabGroup, TabList, Tab, 
  TabPanel, TabPanels, DonutChart,
  Table, TableHead, TableRow, 
  TableHeaderCell, TableBody, TableCell
} from "@tremor/react";

import { userDailyActivityCall } from "./networking";
import ViewUserSpend from "./view_user_spend";

interface NewUsagePageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface SpendMetrics {
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  api_requests: number;
}

interface DailyData {
  date: string;
  metrics: SpendMetrics;
  breakdown: {
    models?: { [key: string]: number };
    providers?: { [key: string]: number };
  };
}

const NewUsagePage: React.FC<NewUsagePageProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  const [userSpendData, setUserSpendData] = useState<{
    results: DailyData[];
    metadata: any;
  }>({ results: [], metadata: {} });

  // Derived states from userSpendData
  const totalSpend = userSpendData.metadata?.total_spend || 0;
  
  // Calculate top models from the breakdown data
  const getTopModels = () => {
    const modelSpend: { [key: string]: number } = {};
    userSpendData.results.forEach(day => {
      Object.entries(day.breakdown.models || {}).forEach(([model, spend]) => {
        modelSpend[model] = (modelSpend[model] || 0) + spend;
      });
    });
    
    return Object.entries(modelSpend)
      .map(([model, spend]) => ({ key: model, spend }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  // Calculate provider spend from the breakdown data
  const getProviderSpend = () => {
    const providerSpend: { [key: string]: number } = {};
    userSpendData.results.forEach(day => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, spend]) => {
        providerSpend[provider] = (providerSpend[provider] || 0) + spend;
      });
    });
    
    return Object.entries(providerSpend)
      .map(([provider, spend]) => ({ provider, spend }));
  };

  const fetchUserSpendData = async () => {
    if (!accessToken) return;
    const startTime = new Date(Date.now() - 28 * 24 * 60 * 60 * 1000);
    const endTime = new Date();
    const data = await userDailyActivityCall(accessToken, startTime, endTime);
    setUserSpendData(data);
  };

  useEffect(() => {
    fetchUserSpendData();
  }, [accessToken]);

  return (
    <div style={{ width: "100%" }} className="p-8">
        <Text>Experimental Usage page, using new `/user/daily/activity` endpoint.</Text>
      <Grid numItems={2} className="gap-2 h-[100vh] w-full">
        {/* Total Spend Card */}
        <Col numColSpan={2}>
          <Text className="text-tremor-default text-tremor-content dark:text-dark-tremor-content mb-2 mt-2 text-lg">
            Project Spend {new Date().toLocaleString('default', { month: 'long' })} 1 - {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).getDate()}
          </Text>
          <ViewUserSpend
            userID={userID}
            userRole={userRole}
            accessToken={accessToken}
            userSpend={totalSpend}
            selectedTeam={null}
            userMaxBudget={null}
          />
        </Col>

        {/* Daily Spend Chart */}
        <Col numColSpan={2}>
          <Card>
            <Title>Daily Spend</Title>
            <BarChart
              data={userSpendData.results}
              index="date"
              categories={["metrics.spend"]}
              colors={["cyan"]}
              valueFormatter={(value) => `$${value.toFixed(2)}`}
              yAxisWidth={100}
              showLegend={false}
              customTooltip={({ payload, active }) => {
                if (!active || !payload?.[0]) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-4 shadow-lg rounded-lg border">
                    <p className="font-bold">{data.date}</p>
                    <p className="text-cyan-500">Spend: ${data.metrics.spend.toFixed(2)}</p>
                    <p className="text-gray-600">Requests: {data.metrics.api_requests}</p>
                    <p className="text-gray-600">Tokens: {data.metrics.total_tokens}</p>
                  </div>
                );
              }}
            />
          </Card>
        </Col>

        {/* Top Models */}
        <Col numColSpan={1}>
          <Card className="h-full">
            <Title>Top Models</Title>
            <BarChart
              className="mt-4 h-40"
              data={getTopModels()}
              index="key"
              categories={["spend"]}
              colors={["cyan"]}
              valueFormatter={(value) => `$${value.toFixed(2)}`}
              layout="vertical"
              yAxisWidth={200}
              showLegend={false}
            />
          </Card>
        </Col>

        {/* Spend by Provider */}
        <Col numColSpan={1}>
          <Card className="h-full">
            <Title>Spend by Provider</Title>
            <Grid numItems={2}>
              <Col numColSpan={1}>
                <DonutChart
                  className="mt-4 h-40"
                  data={getProviderSpend()}
                  index="provider"
                  category="spend"
                  valueFormatter={(value) => `$${value.toFixed(2)}`}
                  colors={["cyan"]}
                />
              </Col>
              <Col numColSpan={1}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Provider</TableHeaderCell>
                      <TableHeaderCell>Spend</TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {getProviderSpend().map((provider) => (
                      <TableRow key={provider.provider}>
                        <TableCell>{provider.provider}</TableCell>
                        <TableCell>
                          {provider.spend < 0.00001 
                            ? "less than 0.00" 
                            : provider.spend.toFixed(2)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Col>
            </Grid>
          </Card>
        </Col>

        {/* Usage Metrics */}
        <Col numColSpan={2}>
          <Card>
            <Title>Usage Metrics</Title>
            <Grid numItems={3} className="gap-4 mt-4">
              <Card>
                <Title>Total Requests</Title>
                <Text className="text-2xl font-bold mt-2">
                  {userSpendData.metadata?.total_api_requests?.toLocaleString() || 0}
                </Text>
              </Card>
              <Card>
                <Title>Total Tokens</Title>
                <Text className="text-2xl font-bold mt-2">
                  {userSpendData.metadata?.total_tokens?.toLocaleString() || 0}
                </Text>
              </Card>
              <Card>
                <Title>Average Cost per Request</Title>
                <Text className="text-2xl font-bold mt-2">
                  ${((totalSpend || 0) / (userSpendData.metadata?.total_api_requests || 1)).toFixed(4)}
                </Text>
              </Card>
            </Grid>
          </Card>
        </Col>
      </Grid>
    </div>
  );
};

export default NewUsagePage;