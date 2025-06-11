import React, { useState, useEffect } from "react";
import { 
  BarChart, Card, Title, Text, 
  Grid, Col, DateRangePicker, DateRangePickerValue,
  Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell,
  DonutChart,
  TabPanel, TabGroup, TabList, Tab, TabPanels
} from "@tremor/react";
import { Select } from 'antd';
import { ActivityMetrics, processActivityData } from './activity_metrics';
import { DailyData, KeyMetricWithMetadata, EntityMetricWithMetadata } from './usage/types';
import { tagDailyActivityCall, teamDailyActivityCall } from './networking';
import TopKeyView from "./top_key_view";

interface EntityMetrics {
  metrics: {
    spend: number;
    prompt_tokens: number;
    completion_tokens: number;
    cache_read_input_tokens: number;
    cache_creation_input_tokens: number;
    total_tokens: number;
    successful_requests: number;
    failed_requests: number;
    api_requests: number;
  };
  metadata: Record<string, any>;
}

interface BreakdownMetrics {
  models: Record<string, any>;
  providers: Record<string, any>;
  api_keys: Record<string, any>;
  entities: Record<string, EntityMetrics>;
}

interface ExtendedDailyData extends DailyData {
  breakdown: BreakdownMetrics;
}

interface EntitySpendData {
  results: ExtendedDailyData[];
  metadata: {
    total_spend: number;
    total_api_requests: number;
    total_successful_requests: number;
    total_failed_requests: number;
    total_tokens: number;
  };
}

export interface EntityList {
  label: string;
  value: string;
}

interface EntityUsageProps {
  accessToken: string | null;
  entityType: 'tag' | 'team';
  entityId?: string | null;
  userID: string | null;
  userRole: string | null;
  entityList: EntityList[] | null;
  premiumUser: boolean;
}

const EntityUsage: React.FC<EntityUsageProps> = ({
  accessToken,
  entityType,
  entityId,
  userID,
  userRole,
  entityList,
  premiumUser
}) => {
  const [spendData, setSpendData] = useState<EntitySpendData>({ 
    results: [], 
    metadata: {
      total_spend: 0,
      total_api_requests: 0,
      total_successful_requests: 0,
      total_failed_requests: 0,
      total_tokens: 0
    }
  });

  const modelMetrics = processActivityData(spendData, "models");
  const keyMetrics = processActivityData(spendData, "api_keys");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 28 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const fetchSpendData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;
    const startTime = dateValue.from;
    const endTime = dateValue.to;
    
    if (entityType === 'tag') {
      const data = await tagDailyActivityCall(
        accessToken, 
        startTime, 
        endTime, 
        1, 
        selectedTags.length > 0 ? selectedTags : null
      );
      setSpendData(data);
    } else if (entityType === 'team') {
      const data = await teamDailyActivityCall(
        accessToken, 
        startTime, 
        endTime, 
        1, 
        selectedTags.length > 0 ? selectedTags : null
      );
      setSpendData(data);
    } else {
      throw new Error("Invalid entity type");
    }
  };

  useEffect(() => {
    fetchSpendData();
  }, [accessToken, dateValue, entityId, selectedTags]);

  const getTopModels = () => {
    const modelSpend: { [key: string]: any } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
        if (!modelSpend[model]) {
          modelSpend[model] = {
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0
          };
        }
        try {
          modelSpend[model].spend += metrics.metrics.spend;
        } catch (e) {
          console.log(`Error adding spend for ${model}: ${e}, got metrics: ${JSON.stringify(metrics)}`);
        }
        modelSpend[model].requests += metrics.metrics.api_requests;
        modelSpend[model].successful_requests += metrics.metrics.successful_requests;
        modelSpend[model].failed_requests += metrics.metrics.failed_requests;
        modelSpend[model].tokens += metrics.metrics.total_tokens;
      });
    });
    
    return Object.entries(modelSpend)
      .map(([model, metrics]) => ({
        key: model,
        ...metrics
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  const getTopAPIKeys = () => {
    const keySpend: { [key: string]: KeyMetricWithMetadata } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.api_keys || {}).forEach(([key, metrics]) => {
        if (!keySpend[key]) {
          keySpend[key] = {
            metrics: {
              spend: 0,
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
              api_requests: 0,
              successful_requests: 0,
              failed_requests: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0
            },
            metadata: {
              key_alias: metrics.metadata.key_alias
            }
          };
        }
        keySpend[key].metrics.spend += metrics.metrics.spend;
        keySpend[key].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        keySpend[key].metrics.completion_tokens += metrics.metrics.completion_tokens;
        keySpend[key].metrics.total_tokens += metrics.metrics.total_tokens;
        keySpend[key].metrics.api_requests += metrics.metrics.api_requests;
        keySpend[key].metrics.successful_requests += metrics.metrics.successful_requests;
        keySpend[key].metrics.failed_requests += metrics.metrics.failed_requests;
        keySpend[key].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        keySpend[key].metrics.cache_creation_input_tokens += metrics.metrics.cache_creation_input_tokens || 0;
      });
    });
    
    return Object.entries(keySpend)
      .map(([api_key, metrics]) => ({
        api_key,
        key_alias: metrics.metadata.key_alias || "-", // Using truncated key as alias
        spend: metrics.metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  const getProviderSpend = () => {
    const providerSpend: { [key: string]: any } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpend[provider]) {
          providerSpend[provider] = {
            provider,
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0
          };
        }
        try {
          providerSpend[provider].spend += metrics.metrics.spend;
          providerSpend[provider].requests += metrics.metrics.api_requests;
          providerSpend[provider].successful_requests += metrics.metrics.successful_requests;
          providerSpend[provider].failed_requests += metrics.metrics.failed_requests;
          providerSpend[provider].tokens += metrics.metrics.total_tokens;
        } catch (e) {
          console.log(`Error processing provider ${provider}: ${e}`);
        }
      });
    });
    
    return Object.values(providerSpend)
      .filter(provider => provider.spend > 0)
      .sort((a, b) => b.spend - a.spend);
  };

  const getAllTags = () => {
    if (entityList) {
      return entityList;
    }
  };

  const filterDataByTags = (data: EntityMetricWithMetadata[]) => {
    if (selectedTags.length === 0) return data;
    return data.filter(item => selectedTags.includes(item.metadata.id));
  };

  const getEntityBreakdown = () => {
    const entitySpend: { [key: string]: EntityMetricWithMetadata } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.entities || {}).forEach(([entity, data]) => {
        if (!entitySpend[entity]) {
          entitySpend[entity] = {
            metrics: {
              spend: 0,
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
              api_requests: 0,
              successful_requests: 0,
              failed_requests: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0
            },
            metadata: {
              alias: data.metadata.team_alias || entity,
              id: entity
            }
          };
        }
        entitySpend[entity].metrics.spend += data.metrics.spend;
        entitySpend[entity].metrics.api_requests += data.metrics.api_requests;
        entitySpend[entity].metrics.successful_requests += data.metrics.successful_requests;
        entitySpend[entity].metrics.failed_requests += data.metrics.failed_requests;
        entitySpend[entity].metrics.total_tokens += data.metrics.total_tokens;
      });
    });
    
    const result = Object.values(entitySpend)
      .sort((a, b) => b.metrics.spend - a.metrics.spend);
    
    return filterDataByTags(result);
  };

  

  return (
    <div style={{ width: "100%" }}>
      <Grid numItems={2} className="gap-2 w-full mb-4">
          <Col>
            <Text>Select Time Range</Text>
            <DateRangePicker
              enableSelect={true}
              value={dateValue}
              onValueChange={setDateValue}
            />
          </Col>
          {entityList && entityList.length > 0 && (
            <Col>
              <Text>Filter by {entityType === 'tag' ? 'Tags' : 'Teams'}</Text>
              <Select
              mode="multiple"
              style={{ width: '100%' }}
              placeholder={`Select ${entityType === 'tag' ? 'tags' : 'teams'} to filter...`}
              value={selectedTags}
              onChange={setSelectedTags}
              options={getAllTags()}
              className="mt-2"
              allowClear
              />
            </Col>
          )}
        </Grid>
      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Cost</Tab>
          <Tab>Model Activity</Tab>
          <Tab>Key Activity</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              {/* Total Spend Card */}
              <Col numColSpan={2}>
                <Card>
                  <Title>{entityType === 'tag' ? 'Tag' : 'Team'} Spend Overview</Title>
                  <Grid numItems={5} className="gap-4 mt-4">
                    <Card>
                      <Title>Total Spend</Title>
                      <Text className="text-2xl font-bold mt-2">
                        ${spendData.metadata.total_spend.toFixed(2)}
                      </Text>
                    </Card>
                    <Card>
                      <Title>Total Requests</Title>
                      <Text className="text-2xl font-bold mt-2">
                        {spendData.metadata.total_api_requests.toLocaleString()}
                      </Text>
                    </Card>
                    <Card>
                      <Title>Successful Requests</Title>
                      <Text className="text-2xl font-bold mt-2 text-green-600">
                        {spendData.metadata.total_successful_requests.toLocaleString()}
                      </Text>
                    </Card>
                    <Card>
                      <Title>Failed Requests</Title>
                      <Text className="text-2xl font-bold mt-2 text-red-600">
                        {spendData.metadata.total_failed_requests.toLocaleString()}
                      </Text>
                    </Card>
                    <Card>
                      <Title>Total Tokens</Title>
                      <Text className="text-2xl font-bold mt-2">
                        {spendData.metadata.total_tokens.toLocaleString()}
                      </Text>
                    </Card>
                  </Grid>
                </Card>
              </Col>

              {/* Daily Spend Chart */}
              <Col numColSpan={2}>
              <Card>
                        <Title>Daily Spend</Title>
                        <BarChart
                          data={[...spendData.results].sort((a, b) => 
                            new Date(a.date).getTime() - new Date(b.date).getTime()
                          )}
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
                                <p className="text-cyan-500">Total Spend: ${data.metrics.spend.toFixed(2)}</p>
                                <p className="text-gray-600">Total Requests: {data.metrics.api_requests}</p>
                                <p className="text-gray-600">Successful: {data.metrics.successful_requests}</p>
                                <p className="text-gray-600">Failed: {data.metrics.failed_requests}</p>
                                <p className="text-gray-600">Total Tokens: {data.metrics.total_tokens}</p>
                                <div className="mt-2 border-t pt-2">
                                  <p className="font-semibold">Spend by {entityType === 'tag' ? 'Tag' : 'Team'}:</p>
                                  {Object.entries(data.breakdown.entities || {}).map(([entity, entityData]) => {
                                    const metrics = entityData as EntityMetrics;
                                    return (
                                      <p key={entity} className="text-sm text-gray-600">
                                        {metrics.metadata.team_alias || entity}: ${metrics.metrics.spend.toFixed(2)}
                                      </p>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          }}
                        />
                      </Card>
              </Col>

              {/* Entity Breakdown Section */}
              <Col numColSpan={2}>
                <Card>
                  <div className="flex flex-col space-y-4">
                    <div className="flex flex-col space-y-2">
                      <Title>Spend Per {entityType === 'tag' ? 'Tag' : 'Team'}</Title>
                      <div className="flex items-center text-sm text-gray-500">
                        <span>Get Started Tracking cost per {entityType} </span>
                        <a href="https://docs.litellm.ai/docs/proxy/enterprise#spend-tracking" className="text-blue-500 hover:text-blue-700 ml-1">
                          here
                        </a>
                      </div>
                    </div>
                      <Grid numItems={2}>
                        <Col numColSpan={1}>
                        <BarChart
                          className="mt-4 h-52"
                          data={getEntityBreakdown()}
                          index="metadata.alias"
                          categories={["metrics.spend"]}
                          colors={["cyan"]}
                          valueFormatter={(value) => `$${value.toFixed(4)}`}
                          layout="vertical"
                          showLegend={false}
                          yAxisWidth={100}
                        />
                      </Col>
                      <Col numColSpan={1}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableHeaderCell>{entityType === 'tag' ? 'Tag' : 'Team'}</TableHeaderCell>
                              <TableHeaderCell>Spend</TableHeaderCell>
                              <TableHeaderCell className="text-green-600">Successful</TableHeaderCell>
                              <TableHeaderCell className="text-red-600">Failed</TableHeaderCell>
                              <TableHeaderCell>Tokens</TableHeaderCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {getEntityBreakdown()
                              .filter(entity => entity.metrics.spend > 0)
                              .map((entity) => (
                                <TableRow key={entity.metadata.id}>
                                  <TableCell>{entity.metadata.alias}</TableCell>
                                  <TableCell>${entity.metrics.spend.toFixed(4)}</TableCell>
                                  <TableCell className="text-green-600">
                                    {entity.metrics.successful_requests.toLocaleString()}
                                  </TableCell>
                                  <TableCell className="text-red-600">
                                    {entity.metrics.failed_requests.toLocaleString()}
                                  </TableCell>
                                  <TableCell>{entity.metrics.total_tokens.toLocaleString()}</TableCell>
                                </TableRow>
                              ))}
                          </TableBody>
                        </Table>
                      </Col>
                    </Grid>
                  </div>
                </Card>
              </Col>


              {/* Top API Keys */}
              <Col numColSpan={1}>
                <Card>
                  <Title>Top API Keys</Title>
                    <TopKeyView
                      topKeys={getTopAPIKeys()}
                      accessToken={accessToken}
                      userID={userID}
                      userRole={userRole}
                      teams={null}
                      premiumUser={premiumUser}
                    />
                </Card>
              </Col>

              {/* Top Models */}
              <Col numColSpan={1}>
                <Card>
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
              <Col numColSpan={2}>
                <Card>
                  <div className="flex flex-col space-y-4">
                    <Title>Provider Usage</Title>
                    <Grid numItems={2}>
                      <Col numColSpan={1}>
                        <DonutChart
                          className="mt-4 h-40"
                          data={getProviderSpend()}
                          index="provider"
                          category="spend"
                          valueFormatter={(value) => `$${value.toFixed(2)}`}
                          colors={["cyan", "blue", "indigo", "violet", "purple"]}
                        />
                      </Col>
                      <Col numColSpan={1}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableHeaderCell>Provider</TableHeaderCell>
                              <TableHeaderCell>Spend</TableHeaderCell>
                              <TableHeaderCell className="text-green-600">Successful</TableHeaderCell>
                              <TableHeaderCell className="text-red-600">Failed</TableHeaderCell>
                              <TableHeaderCell>Tokens</TableHeaderCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {getProviderSpend().map((provider) => (
                              <TableRow key={provider.provider}>
                                <TableCell>{provider.provider}</TableCell>
                                <TableCell>${provider.spend.toFixed(2)}</TableCell>
                                <TableCell className="text-green-600">
                                  {provider.successful_requests.toLocaleString()}
                                </TableCell>
                                <TableCell className="text-red-600">
                                  {provider.failed_requests.toLocaleString()}
                                </TableCell>
                                <TableCell>{provider.tokens.toLocaleString()}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Col>
                    </Grid>
                  </div>
                </Card>
              </Col>
            </Grid>
          </TabPanel>
          <TabPanel>
          <ActivityMetrics modelMetrics={modelMetrics} />
          </TabPanel>
          <TabPanel>
          <ActivityMetrics modelMetrics={keyMetrics} />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default EntityUsage; 