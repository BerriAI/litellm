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
  TableHeaderCell, TableBody, TableCell,
  Subtitle
} from "@tremor/react";
import { AreaChart } from "@tremor/react";

import { userDailyActivityCall } from "./networking";
import ViewUserSpend from "./view_user_spend";
import TopKeyView from "./top_key_view";

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

interface BreakdownMetrics {
  models: { [key: string]: SpendMetrics };
  providers: { [key: string]: SpendMetrics };
  api_keys: { [key: string]: SpendMetrics };
}

interface DailyData {
  date: string;
  metrics: SpendMetrics;
  breakdown: BreakdownMetrics;
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
    const modelSpend: { [key: string]: SpendMetrics } = {};
    userSpendData.results.forEach(day => {
      Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
        if (!modelSpend[model]) {
          modelSpend[model] = {
            spend: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            api_requests: 0
          };
        }
        modelSpend[model].spend += metrics.spend;
        modelSpend[model].prompt_tokens += metrics.prompt_tokens;
        modelSpend[model].completion_tokens += metrics.completion_tokens;
        modelSpend[model].total_tokens += metrics.total_tokens;
        modelSpend[model].api_requests += metrics.api_requests;
      });
    });
    
    return Object.entries(modelSpend)
      .map(([model, metrics]) => ({
        key: model,
        spend: metrics.spend,
        requests: metrics.api_requests,
        tokens: metrics.total_tokens
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  // Calculate provider spend from the breakdown data
  const getProviderSpend = () => {
    const providerSpend: { [key: string]: SpendMetrics } = {};
    userSpendData.results.forEach(day => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpend[provider]) {
          providerSpend[provider] = {
            spend: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            api_requests: 0
          };
        }
        providerSpend[provider].spend += metrics.spend;
        providerSpend[provider].prompt_tokens += metrics.prompt_tokens;
        providerSpend[provider].completion_tokens += metrics.completion_tokens;
        providerSpend[provider].total_tokens += metrics.total_tokens;
        providerSpend[provider].api_requests += metrics.api_requests;
      });
    });
    
    return Object.entries(providerSpend)
      .map(([provider, metrics]) => ({
        provider,
        spend: metrics.spend,
        requests: metrics.api_requests,
        tokens: metrics.total_tokens
      }));
  };

  // Calculate top API keys from the breakdown data
  const getTopKeys = () => {
    const keySpend: { [key: string]: SpendMetrics } = {};
    userSpendData.results.forEach(day => {
      Object.entries(day.breakdown.api_keys || {}).forEach(([key, metrics]) => {
        if (!keySpend[key]) {
          keySpend[key] = {
            spend: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            api_requests: 0
          };
        }
        keySpend[key].spend += metrics.spend;
        keySpend[key].prompt_tokens += metrics.prompt_tokens;
        keySpend[key].completion_tokens += metrics.completion_tokens;
        keySpend[key].total_tokens += metrics.total_tokens;
        keySpend[key].api_requests += metrics.api_requests;
      });
    });
    
    return Object.entries(keySpend)
      .map(([api_key, metrics]) => ({
        api_key,
        key_alias: api_key.substring(0, 10), // Using truncated key as alias
        spend: metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
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
      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Cost</Tab>
          <Tab>Activity</Tab>
        </TabList>
        <TabPanels>
          {/* Cost Panel */}
          <TabPanel>
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

              {/* Top API Keys */}
              <Col numColSpan={1}>
                <Card className="h-full">
                  <Title>Top API Keys</Title>
                  <TopKeyView
                    topKeys={getTopKeys()}
                    accessToken={accessToken}
                    userID={userID}
                    userRole={userRole}
                    teams={null}
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
                    customTooltip={({ payload, active }) => {
                      if (!active || !payload?.[0]) return null;
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-4 shadow-lg rounded-lg border">
                          <p className="font-bold">{data.key}</p>
                          <p className="text-cyan-500">Spend: ${data.spend.toFixed(2)}</p>
                          <p className="text-gray-600">Requests: {data.requests.toLocaleString()}</p>
                          <p className="text-gray-600">Tokens: {data.tokens.toLocaleString()}</p>
                        </div>
                      );
                    }}
                  />
                </Card>
              </Col>

              {/* Spend by Provider */}
              <Col numColSpan={2}>
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
                            <TableHeaderCell>Requests</TableHeaderCell>
                            <TableHeaderCell>Tokens</TableHeaderCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {getProviderSpend().map((provider) => (
                            <TableRow key={provider.provider}>
                              <TableCell>{provider.provider}</TableCell>
                              <TableCell>
                                ${provider.spend < 0.00001 
                                  ? "less than 0.00" 
                                  : provider.spend.toFixed(2)}
                              </TableCell>
                              <TableCell>{provider.requests.toLocaleString()}</TableCell>
                              <TableCell>{provider.tokens.toLocaleString()}</TableCell>
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
          </TabPanel>

          {/* Activity Panel */}
          <TabPanel>
            <Grid numItems={1} className="gap-2 h-[75vh] w-full">
              <Card>
                <Title>All Up</Title>
                <Grid numItems={2}>
                  <Col>
                    <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>
                      API Requests {valueFormatterNumbers(userSpendData.metadata?.total_api_requests || 0)}
                    </Subtitle>
                    <AreaChart
                      className="h-40"
                      data={[...userSpendData.results].reverse()}
                      valueFormatter={valueFormatterNumbers}
                      index="date"
                      colors={['cyan']}
                      categories={['metrics.api_requests']}
                    />
                  </Col>
                  <Col>
                    <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>
                      Tokens {valueFormatterNumbers(userSpendData.metadata?.total_tokens || 0)}
                    </Subtitle>
                    <BarChart
                      className="h-40"
                      data={[...userSpendData.results].reverse()}
                      valueFormatter={valueFormatterNumbers}
                      index="date"
                      colors={['cyan']}
                      categories={['metrics.total_tokens']}
                    />
                  </Col>
                </Grid>
              </Card>

              {/* Per Model Activity */}
              {Object.entries(getModelActivityData(userSpendData)).map(([model, data], index) => (
                <Card key={index}>
                  <Title>{model}</Title>
                  <Grid numItems={2}>
                    <Col>
                      <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>
                        API Requests {valueFormatterNumbers(data.total_requests)}
                      </Subtitle>
                      <AreaChart
                        className="h-40"
                        data={[...data.daily_data].reverse()}
                        index="date"
                        colors={['cyan']}
                        categories={['api_requests']}
                        valueFormatter={valueFormatterNumbers}
                      />
                    </Col>
                    <Col>
                      <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>
                        Tokens {valueFormatterNumbers(data.total_tokens)}
                      </Subtitle>
                      <BarChart
                        className="h-40"
                        data={data.daily_data}  
                        index="date"
                        colors={['cyan']}
                        categories={['total_tokens']}
                        valueFormatter={valueFormatterNumbers}
                      />
                    </Col>
                  </Grid>
                </Card>
              ))}
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

// Add this helper function to process model-specific activity data
const getModelActivityData = (userSpendData: {
  results: DailyData[];
  metadata: any;
}) => {
  const modelData: {
    [key: string]: {
      total_requests: number;
      total_tokens: number;
      daily_data: Array<{
        date: string;
        api_requests: number;
        total_tokens: number;
      }>;
    };
  } = {};

  userSpendData.results.forEach((day: DailyData) => {
    Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
      if (!modelData[model]) {
        modelData[model] = {
          total_requests: 0,
          total_tokens: 0,
          daily_data: []
        };
      }
      
      modelData[model].total_requests += metrics.api_requests;
      modelData[model].total_tokens += metrics.total_tokens;
      modelData[model].daily_data.push({
        date: day.date,
        api_requests: metrics.api_requests,
        total_tokens: metrics.total_tokens
      });
    });
  });

  return modelData;
};

// Add this helper function for number formatting
function valueFormatterNumbers(number: number) {
  const formatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
    notation: 'compact',
    compactDisplay: 'short',
  });
  return formatter.format(number);
}

export default NewUsagePage;