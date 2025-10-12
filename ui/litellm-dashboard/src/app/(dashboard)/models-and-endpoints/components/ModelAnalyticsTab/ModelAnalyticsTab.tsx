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
import React, { useEffect } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Team } from "@/components/key_team_helpers/key_list";
import {
  adminGlobalActivityExceptions,
  adminGlobalActivityExceptionsPerDeployment,
  modelExceptionsCall,
  modelMetricsCall,
  modelMetricsSlowResponsesCall,
  streamingModelMetricsCall,
} from "@/components/networking";
import FilterByContent from "@/app/(dashboard)/models-and-endpoints/components/ModelAnalyticsTab/FilterByContent";

interface GlobalExceptionActivityData {
  sum_num_rate_limit_exceptions: number;
  daily_data: { date: string; num_rate_limit_exceptions: number }[];
}

interface ModelAnalyticsTabProps {
  dateValue: DateRangePickerValue;
  setDateValue: (dateValue: DateRangePickerValue) => void;
  selectedModelGroup: string | null;
  availableModelGroups: string[];
  setShowAdvancedFilters: (showAdvancedFilters: boolean) => void;
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
  setSelectedAPIKey: (key: string | null) => void;
  keys: any[] | null;
  setSelectedCustomer: (selectedCustomer: string | null) => void;
  teams: Team[] | null;
  allEndUsers: any[];
  selectedAPIKey: any;
  selectedCustomer: string | null;
  selectedTeam: string | null;
  setSelectedModelGroup: (selectedModelGroup: string | null) => void;
  setModelMetrics: (metrics: any) => void;
  setModelMetricsCategories: (categories: any) => void;
  setStreamingModelMetrics: (metrics: any) => void;
  setStreamingModelMetricsCategories: (categories: any) => void;
  setSlowResponsesData: (data: any) => void;
  setModelExceptions: (exceptions: any) => void;
  setAllExceptions: (exceptions: any) => void;
  setGlobalExceptionData: (data: any) => void;
  setGlobalExceptionPerDeployment: (data: any) => void;
}

const ModelAnalyticsTab = ({
  dateValue,
  setDateValue,
  selectedModelGroup,
  availableModelGroups,
  setShowAdvancedFilters,
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
  setSelectedAPIKey,
  keys,
  setSelectedCustomer,
  teams,
  allEndUsers,
  selectedAPIKey,
  selectedCustomer,
  selectedTeam,
  setSelectedModelGroup,
  setModelMetrics,
  setModelMetricsCategories,
  setStreamingModelMetrics,
  setStreamingModelMetricsCategories,
  setSlowResponsesData,
  setModelExceptions,
  setAllExceptions,
  setGlobalExceptionData,
  setGlobalExceptionPerDeployment,
}: ModelAnalyticsTabProps) => {
  const { accessToken, userId, userRole, premiumUser } = useAuthorized();

  useEffect(() => {
    updateModelMetrics(selectedModelGroup, dateValue.from, dateValue.to);
  }, [selectedAPIKey, selectedCustomer, selectedTeam]);

  const updateModelMetrics = async (
    modelGroup: string | null,
    startTime: Date | undefined,
    endTime: Date | undefined,
  ) => {
    console.log("Updating model metrics for group:", modelGroup);
    if (!accessToken || !userId || !userRole || !startTime || !endTime) {
      return;
    }
    console.log("inside updateModelMetrics - startTime:", startTime, "endTime:", endTime);
    setSelectedModelGroup(modelGroup);

    let selected_token = selectedAPIKey?.token;
    if (selected_token === undefined) {
      selected_token = null;
    }

    let selected_customer = selectedCustomer;
    if (selected_customer === undefined) {
      selected_customer = null;
    }

    try {
      const modelMetricsResponse = await modelMetricsCall(
        accessToken,
        userId,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );
      console.log("Model metrics response:", modelMetricsResponse);

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setModelMetrics(modelMetricsResponse.data);
      setModelMetricsCategories(modelMetricsResponse.all_api_bases);

      const streamingModelMetricsResponse = await streamingModelMetricsCall(
        accessToken,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
      );

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setStreamingModelMetrics(streamingModelMetricsResponse.data);
      setStreamingModelMetricsCategories(streamingModelMetricsResponse.all_api_bases);

      const modelExceptionsResponse = await modelExceptionsCall(
        accessToken,
        userId,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );
      console.log("Model exceptions response:", modelExceptionsResponse);
      setModelExceptions(modelExceptionsResponse.data);
      setAllExceptions(modelExceptionsResponse.exception_types);

      const slowResponses = await modelMetricsSlowResponsesCall(
        accessToken,
        userId,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );

      console.log("slowResponses:", slowResponses);

      setSlowResponsesData(slowResponses);

      if (modelGroup) {
        const dailyExceptions = await adminGlobalActivityExceptions(
          accessToken,
          startTime?.toISOString().split("T")[0],
          endTime?.toISOString().split("T")[0],
          modelGroup,
        );

        setGlobalExceptionData(dailyExceptions);

        const dailyExceptionsPerDeplyment = await adminGlobalActivityExceptionsPerDeployment(
          accessToken,
          startTime?.toISOString().split("T")[0],
          endTime?.toISOString().split("T")[0],
          modelGroup,
        );

        setGlobalExceptionPerDeployment(dailyExceptionsPerDeplyment);
      }
    } catch (error) {
      console.error("Failed to fetch model metrics", error);
    }
  };

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
            content={
              <FilterByContent
                allEndUsers={allEndUsers}
                keys={keys}
                setSelectedAPIKey={setSelectedAPIKey}
                setSelectedCustomer={setSelectedCustomer}
                teams={teams}
              />
            }
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
