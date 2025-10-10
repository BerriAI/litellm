import {
  AreaChart,
  BarChart,
  Button,
  Card,
  Col,
  DateRangePickerValue,
  Grid,
  Select,
  SelectItem,
  Subtitle,
  Tab,
  TabGroup,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  Title,
} from "@tremor/react";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { Popover } from "antd";
import { FilterIcon } from "@heroicons/react/outline";
import TimeToFirstToken from "@/components/model_metrics/time_to_first_token";
import React, { ReactNode } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface GlobalExceptionActivityData {
  sum_num_rate_limit_exceptions: number;
  daily_data: { date: string; num_rate_limit_exceptions: number }[];
}

interface ModelAnalyticsTabProps {
  dateValue: DateRangePickerValue;
  setDateValue: (dateValue: DateRangePickerValue) => void;
  selectedModelGroup: string | null;
  availableModelGroups: string[];
  FilterByContent: ReactNode;
  setShowAdvancedFilters: (showAdvancedFilters: boolean) => void;
  updateModelMetrics: (
    modelGroup: string | null,
    startTime: Date | undefined,
    endTime: Date | undefined,
  ) => Promise<void>;
  modelMetrics: any[];
  modelMetricsCategories: any[];
  streamingModelMetrics: any[];
  streamingModelMetricsCategories: any[];
  customTooltip: any;
  slowResponsesData: any[];
  modelExceptions: any[];
  globalExceptionData: GlobalExceptionActivityData;
  allExceptions: any[];
  globalExceptionPerDeployment: any[];
}

const ModelAnalyticsTab = ({
  dateValue,
  setDateValue,
  selectedModelGroup,
  availableModelGroups,
  FilterByContent,
  setShowAdvancedFilters,
  updateModelMetrics,
  modelMetrics,
  modelMetricsCategories,
  streamingModelMetrics,
  streamingModelMetricsCategories,
  customTooltip,
  slowResponsesData,
  modelExceptions,
  globalExceptionData,
  allExceptions,
  globalExceptionPerDeployment,
}: ModelAnalyticsTabProps) => {
  const { premiumUser } = useAuthorized();

  return (
    <TabPanel>
      <Grid numItems={4} className="mt-2 mb-2">
        <Col>
          <UsageDatePicker
            value={dateValue}
            className="mr-2"
            onValueChange={(value) => {
              setDateValue(value);
              updateModelMetrics(selectedModelGroup, value.from, value.to);
            }}
          />
        </Col>
        <Col className="ml-2">
          <Text>Select Model Group</Text>
          <Select
            defaultValue={selectedModelGroup ? selectedModelGroup : availableModelGroups[0]}
            value={selectedModelGroup ? selectedModelGroup : availableModelGroups[0]}
          >
            {availableModelGroups.map((group, idx) => (
              <SelectItem
                key={idx}
                value={group}
                onClick={() => updateModelMetrics(group, dateValue.from, dateValue.to)}
              >
                {group}
              </SelectItem>
            ))}
          </Select>
        </Col>
        <Col>
          <Popover
            trigger="click"
            content={FilterByContent}
            overlayStyle={{
              width: "20vw",
            }}
          >
            <Button
              icon={FilterIcon}
              size="md"
              variant="secondary"
              className="mt-4 ml-2"
              style={{
                border: "none",
              }}
              onClick={() => setShowAdvancedFilters(true)}
            ></Button>
          </Popover>
        </Col>
      </Grid>

      <Grid numItems={2}>
        <Col>
          <Card className="mr-2 max-h-[400px] min-h-[400px]">
            <TabGroup>
              <TabList variant="line" defaultValue="1">
                <Tab value="1">Avg. Latency per Token</Tab>
                <Tab value="2">Time to first token</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <p className="text-gray-500 italic"> (seconds/token)</p>
                  <Text className="text-gray-500 italic mt-1 mb-1">
                    average Latency for successfull requests divided by the total tokens
                  </Text>
                  {modelMetrics && modelMetricsCategories && (
                    <AreaChart
                      title="Model Latency"
                      className="h-72"
                      data={modelMetrics}
                      showLegend={false}
                      index="date"
                      categories={modelMetricsCategories}
                      connectNulls={true}
                      customTooltip={customTooltip}
                    />
                  )}
                </TabPanel>
                <TabPanel>
                  <TimeToFirstToken
                    modelMetrics={streamingModelMetrics}
                    modelMetricsCategories={streamingModelMetricsCategories}
                    customTooltip={customTooltip}
                    premiumUser={premiumUser}
                  />
                </TabPanel>
              </TabPanels>
            </TabGroup>
          </Card>
        </Col>
        <Col>
          <Card className="ml-2 max-h-[400px] min-h-[400px]  overflow-y-auto">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Deployment</TableHeaderCell>
                  <TableHeaderCell>Success Responses</TableHeaderCell>
                  <TableHeaderCell>
                    Slow Responses <p>Success Responses taking 600+s</p>
                  </TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {slowResponsesData.map((metric, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{metric.api_base}</TableCell>
                    <TableCell>{metric.total_count}</TableCell>
                    <TableCell>{metric.slow_count}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </Col>
      </Grid>
      <Grid numItems={1} className="gap-2 w-full mt-2">
        <Card>
          <Title>All Exceptions for {selectedModelGroup}</Title>

          <BarChart
            className="h-60"
            data={modelExceptions}
            index="model"
            categories={allExceptions}
            stack={true}
            yAxisWidth={30}
          />
        </Card>
      </Grid>

      <Grid numItems={1} className="gap-2 w-full mt-2">
        <Card>
          <Title>All Up Rate Limit Errors (429) for {selectedModelGroup}</Title>
          <Grid numItems={1}>
            <Col>
              <Subtitle
                style={{
                  fontSize: "15px",
                  fontWeight: "normal",
                  color: "#535452",
                }}
              >
                Num Rate Limit Errors {globalExceptionData.sum_num_rate_limit_exceptions}
              </Subtitle>
              <BarChart
                className="h-40"
                data={globalExceptionData.daily_data}
                index="date"
                colors={["rose"]}
                categories={["num_rate_limit_exceptions"]}
                onValueChange={(v) => console.log(v)}
              />
            </Col>
            <Col></Col>
          </Grid>
        </Card>

        {premiumUser ? (
          <>
            {globalExceptionPerDeployment.map((globalActivity, index) => (
              <Card key={index}>
                <Title>{globalActivity.api_base ? globalActivity.api_base : "Unknown API Base"}</Title>
                <Grid numItems={1}>
                  <Col>
                    <Subtitle
                      style={{
                        fontSize: "15px",
                        fontWeight: "normal",
                        color: "#535452",
                      }}
                    >
                      Num Rate Limit Errors (429) {globalActivity.sum_num_rate_limit_exceptions}
                    </Subtitle>
                    <BarChart
                      className="h-40"
                      data={globalActivity.daily_data}
                      index="date"
                      colors={["rose"]}
                      categories={["num_rate_limit_exceptions"]}
                      onValueChange={(v) => console.log(v)}
                    />
                  </Col>
                </Grid>
              </Card>
            ))}
          </>
        ) : (
          <>
            {globalExceptionPerDeployment &&
              globalExceptionPerDeployment.length > 0 &&
              globalExceptionPerDeployment.slice(0, 1).map((globalActivity, index) => (
                <Card key={index}>
                  <Title>✨ Rate Limit Errors by Deployment</Title>
                  <p className="mb-2 text-gray-500 italic text-[12px]">Upgrade to see exceptions for all deployments</p>
                  <Button variant="primary" className="mb-2">
                    <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                      Get Free Trial
                    </a>
                  </Button>
                  <Card>
                    <Title>{globalActivity.api_base}</Title>
                    <Grid numItems={1}>
                      <Col>
                        <Subtitle
                          style={{
                            fontSize: "15px",
                            fontWeight: "normal",
                            color: "#535452",
                          }}
                        >
                          Num Rate Limit Errors {globalActivity.sum_num_rate_limit_exceptions}
                        </Subtitle>
                        <BarChart
                          className="h-40"
                          data={globalActivity.daily_data}
                          index="date"
                          colors={["rose"]}
                          categories={["num_rate_limit_exceptions"]}
                          onValueChange={(v) => console.log(v)}
                        />
                      </Col>
                    </Grid>
                  </Card>
                </Card>
              ))}
          </>
        )}
      </Grid>
    </TabPanel>
  );
};

export default ModelAnalyticsTab;
