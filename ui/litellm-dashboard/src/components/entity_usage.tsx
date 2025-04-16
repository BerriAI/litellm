import React, { useState, useEffect } from "react";
import { 
  BarChart, Card, Title, Text, 
  Grid, Col, DateRangePicker, DateRangePickerValue,
  Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell,
  DonutChart
} from "@tremor/react";
import { ActivityMetrics, processActivityData } from './activity_metrics';
import { SpendMetrics, DailyData } from './usage/types';
import { tagDailyActivityCall } from './networking';

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

interface EntityUsageProps {
  accessToken: string | null;
  entityType: 'tag' | 'team';
  entityId?: string | null;
}

const EntityUsage: React.FC<EntityUsageProps> = ({
  accessToken,
  entityType,
  entityId
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

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 28 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const fetchSpendData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;
    const startTime = dateValue.from;
    const endTime = dateValue.to;
    
    if (entityType === 'tag') {
      const data = await tagDailyActivityCall(accessToken, startTime, endTime, 1, entityId);
      setSpendData(data);
    } else {
      throw new Error("Invalid entity type");
    }
  };

  useEffect(() => {
    fetchSpendData();
  }, [accessToken, dateValue, entityId]);

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

  const getProviderSpend = () => {
    const providerSpend: { [key: string]: any } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpend[provider]) {
          providerSpend[provider] = {
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0
          };
        }
        providerSpend[provider].spend += metrics.metrics.spend;
        providerSpend[provider].requests += metrics.metrics.api_requests;
        providerSpend[provider].successful_requests += metrics.metrics.successful_requests;
        providerSpend[provider].failed_requests += metrics.metrics.failed_requests;
        providerSpend[provider].tokens += metrics.metrics.total_tokens;
      });
    });
    
    return Object.entries(providerSpend)
      .map(([provider, metrics]) => ({
        provider,
        ...metrics
      }));
  };

  const getEntityBreakdown = () => {
    const entitySpend: { [key: string]: any } = {};
    spendData.results.forEach(day => {
      Object.entries(day.breakdown.entities || {}).forEach(([entity, data]) => {
        if (!entitySpend[entity]) {
          entitySpend[entity] = {
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0
          };
        }
        entitySpend[entity].spend += data.metrics.spend;
        entitySpend[entity].requests += data.metrics.api_requests;
        entitySpend[entity].successful_requests += data.metrics.successful_requests;
        entitySpend[entity].failed_requests += data.metrics.failed_requests;
        entitySpend[entity].tokens += data.metrics.total_tokens;
      });
    });
    
    return Object.entries(entitySpend)
      .map(([entity, metrics]) => ({
        entity,
        ...metrics
      }))
      .sort((a, b) => b.spend - a.spend);
  };

  return (
    <div style={{ width: "100%" }} className="p-8">
      <Grid numItems={2} className="gap-2 w-full mb-4">
        <Col>
          <Text>Select Time Range</Text>
          <DateRangePicker
            enableSelect={true}
            value={dateValue}
            onValueChange={setDateValue}
          />
        </Col>
      </Grid>

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
            />
          </Card>
        </Col>

        {/* Entity Breakdown Section */}
        <Col numColSpan={1}>
          <Card>
            <div className="flex flex-col space-y-2">
              <Title>Spend Per {entityType === 'tag' ? 'Tag' : 'Team'}</Title>
              <div className="flex items-center text-sm text-gray-500">
                <span>Get Started Tracking cost per {entityType} </span>
                <a href="https://docs.litellm.ai/docs/proxy/tags" className="text-blue-500 hover:text-blue-700 ml-1">
                  here
                </a>
              </div>
            </div>
            <BarChart
              className="mt-4 h-52"
              data={getEntityBreakdown()}
              index="entity"
              categories={["spend"]}
              colors={["cyan"]}
              valueFormatter={(value) => value.toFixed(4)}
              layout="horizontal"
              showLegend={true}
              yAxisWidth={100}
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
        <Col numColSpan={1}>
          <Card>
            <Title>Spend by Provider</Title>
            <DonutChart
              className="mt-4 h-40"
              data={getProviderSpend()}
              index="provider"
              category="spend"
              valueFormatter={(value) => `$${value.toFixed(2)}`}
              colors={["cyan"]}
            />
          </Card>
        </Col>

        {/* Provider Details Table */}
        <Col numColSpan={2}>
          <Card>
            <Title>Provider Details</Title>
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
                {getProviderSpend()
                  .filter(provider => provider.spend > 0)
                  .map((provider) => (
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
          </Card>
        </Col>
      </Grid>
    </div>
  );
};

export default EntityUsage; 