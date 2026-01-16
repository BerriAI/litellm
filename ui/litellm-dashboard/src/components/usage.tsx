import {
  BarChart,
  BarList,
  Card,
  Title,
  Table,
  TableHead,
  TableHeaderCell,
  TableRow,
  TableCell,
  TableBody,
  Subtitle,
} from "@tremor/react";

import React, { useState, useEffect } from "react";

import ViewUserSpend from "./view_user_spend";
import { ProxySettings } from "./user_dashboard";
import UsageDatePicker from "./shared/usage_date_picker";
import {
  Grid,
  Col,
  Text,
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
  Select,
  SelectItem,
  DateRangePickerValue,
  DonutChart,
  AreaChart,
  Button,
  MultiSelect,
  MultiSelectItem,
} from "@tremor/react";

import {
  adminSpendLogsCall,
  adminTopKeysCall,
  adminTopModelsCall,
  adminTopEndUsersCall,
  teamSpendLogsCall,
  tagsSpendLogsCall,
  allTagNamesCall,
  adminspendByProvider,
  adminGlobalActivity,
  adminGlobalActivityPerModel,
  getProxyUISettings,
} from "./networking";
import TopKeyView from "./UsagePage/components/EntityUsage/TopKeyView";
import { formatNumberWithCommas } from "@/utils/dataUtils";
console.log("process.env.NODE_ENV", process.env.NODE_ENV);

interface UsagePageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  keys: any[] | null;
  premiumUser: boolean;
}

interface GlobalActivityData {
  sum_api_requests: number;
  sum_total_tokens: number;
  daily_data: { date: string; api_requests: number; total_tokens: number }[];
}

type CustomTooltipTypeBar = {
  payload: any;
  active: boolean | undefined;
  label: any;
};

const customTooltip = (props: CustomTooltipTypeBar) => {
  const { payload, active } = props;
  if (!active || !payload) return null;

  const value = payload[0].payload;
  const date = value["startTime"];
  const model_values = value["models"];
  const entries: [string, number][] = Object.entries(model_values).map(([key, value]) => [key, value as number]);

  entries.sort((a, b) => b[1] - a[1]);
  const topEntries = entries.slice(0, 5);

  return (
    <div className="w-56 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
      {date}
      {topEntries.map(([key, value]) => (
        <div key={key} className="flex flex-1 space-x-10">
          <div className="p-2">
            <p className="text-tremor-content text-xs">
              {key}
              {":"}
              <span className="text-xs text-tremor-content-emphasis">
                {" "}
                {value ? `$${formatNumberWithCommas(value, 2)}` : ""}
              </span>
            </p>
          </div>
        </div>
      ))}
    </div>
  );
};

function getTopKeys(data: Array<{ [key: string]: unknown }>): any[] {
  const spendKeys: { key: string; spend: unknown }[] = [];

  data.forEach((dict) => {
    Object.entries(dict).forEach(([key, value]) => {
      if (key !== "spend" && key !== "startTime" && key !== "models" && key !== "users") {
        spendKeys.push({ key, spend: value });
      }
    });
  });

  spendKeys.sort((a, b) => Number(b.spend) - Number(a.spend));

  const topKeys = spendKeys.slice(0, 5).map((k) => k.key);
  console.log(`topKeys: ${Object.keys(topKeys[0])}`);
  return topKeys;
}
type DataDict = { [key: string]: unknown };
type UserData = { user_id: string; spend: number };

const isAdminOrAdminViewer = (role: string | null): boolean => {
  if (role === null) return false;
  return role === "Admin" || role === "Admin Viewer";
};

const UsagePage: React.FC<UsagePageProps> = ({ accessToken, token, userRole, userID, keys, premiumUser }) => {
  const currentDate = new Date();
  const [keySpendData, setKeySpendData] = useState<any[]>([]);
  const [topKeys, setTopKeys] = useState<any[]>([]);
  const [topModels, setTopModels] = useState<any[]>([]);
  const [topUsers, setTopUsers] = useState<any[]>([]);
  const [teamSpendData, setTeamSpendData] = useState<any[]>([]);
  const [topTagsData, setTopTagsData] = useState<any[]>([]);
  const [allTagNames, setAllTagNames] = useState<string[]>([]);
  const [uniqueTeamIds, setUniqueTeamIds] = useState<any[]>([]);
  const [totalSpendPerTeam, setTotalSpendPerTeam] = useState<any[]>([]);
  const [spendByProvider, setSpendByProvider] = useState<any[]>([]);
  const [globalActivity, setGlobalActivity] = useState<GlobalActivityData>({} as GlobalActivityData);
  const [globalActivityPerModel, setGlobalActivityPerModel] = useState<any[]>([]);
  const [selectedKeyID, setSelectedKeyID] = useState<string | null>("");
  const [selectedTags, setSelectedTags] = useState<string[]>(["all-tags"]);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });
  const [proxySettings, setProxySettings] = useState<ProxySettings | null>(null);
  const [totalMonthlySpend, setTotalMonthlySpend] = useState<number>(0);

  const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
  const lastDay = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);

  let startTime = formatDate(firstDay);
  let endTime = formatDate(lastDay);

  console.log("keys in usage", keys);
  console.log("premium user in usage", premiumUser);

  function valueFormatterNumbers(number: number) {
    const formatter = new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 0,
      notation: "compact",
      compactDisplay: "short",
    });

    return formatter.format(number);
  }

  const fetchProxySettings = async () => {
    if (accessToken) {
      try {
        const proxy_settings: ProxySettings = await getProxyUISettings(accessToken);
        console.log("usage tab: proxy_settings", proxy_settings);
        return proxy_settings;
      } catch (error) {
        console.error("Error fetching proxy settings:", error);
      }
    }
  };

  useEffect(() => {
    updateTagSpendData(dateValue.from, dateValue.to);
  }, [dateValue, selectedTags]);

  const updateEndUserData = async (
    startTime: Date | undefined,
    endTime: Date | undefined,
    uiSelectedKey: string | null,
  ) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    console.log("uiSelectedKey", uiSelectedKey);

    let newTopUserData = await adminTopEndUsersCall(
      accessToken,
      uiSelectedKey,
      startTime.toISOString(),
      endTime.toISOString(),
    );
    console.log("End user data updated successfully", newTopUserData);
    setTopUsers(newTopUserData);
  };

  const updateTagSpendData = async (startTime: Date | undefined, endTime: Date | undefined) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    // we refetch because the state variable can be None when the user refreshes the page
    const proxy_settings: ProxySettings | undefined = await fetchProxySettings();

    if (proxy_settings?.DISABLE_EXPENSIVE_DB_QUERIES) {
      return; // Don't run expensive DB queries - return out when SpendLogs has more than 1M rows
    }

    let top_tags = await tagsSpendLogsCall(
      accessToken,
      startTime.toISOString(),
      endTime.toISOString(),
      selectedTags.length === 0 ? undefined : selectedTags,
    );
    setTopTagsData(top_tags.spend_per_tag);
    console.log("Tag spend data updated successfully");
  };

  function formatDate(date: Date) {
    const year = date.getFullYear();
    let month = date.getMonth() + 1; // JS month index starts from 0
    let day = date.getDate();

    // Pad with 0 if month or day is less than 10
    const monthStr = month < 10 ? "0" + month : month;
    const dayStr = day < 10 ? "0" + day : day;

    return `${year}-${monthStr}-${dayStr}`;
  }

  console.log(`Start date is ${startTime}`);
  console.log(`End date is ${endTime}`);

  const valueFormatter = (number: number) => `$ ${formatNumberWithCommas(number, 2)}`;

  const fetchAndSetData = async (
    fetchFunction: () => Promise<any>,
    setStateFunction: React.Dispatch<React.SetStateAction<any>>,
    errorMessage: string,
  ) => {
    try {
      const data = await fetchFunction();
      setStateFunction(data);
    } catch (error) {
      console.error(errorMessage, error);
      // Optionally, update UI to reflect error state for this specific data
    }
  };

  // Update the fillMissingDates function to handle different date formats
  const fillMissingDates = (data: any[], startDate: Date, endDate: Date, categories: string[]) => {
    const filledData = [];
    const currentDate = new Date(startDate);

    // Helper function to standardize date format
    const standardizeDate = (dateStr: string) => {
      if (dateStr.includes("-")) {
        // Already in YYYY-MM-DD format
        return dateStr;
      } else {
        // Convert "Jan 06" format
        const [month, day] = dateStr.split(" ");
        const year = new Date().getFullYear();
        const monthIndex = new Date(`${month} 01 2024`).getMonth();
        const fullDate = new Date(year, monthIndex, parseInt(day));
        return fullDate.toISOString().split("T")[0];
      }
    };

    // Create a map of existing dates for quick lookup
    const existingDates = new Map(
      data.map((item) => {
        const standardizedDate = standardizeDate(item.date);
        return [
          standardizedDate,
          {
            ...item,
            date: standardizedDate, // Store standardized date format
          },
        ];
      }),
    );

    // Iterate through each date in the range
    while (currentDate <= endDate) {
      const dateStr = currentDate.toISOString().split("T")[0];

      if (existingDates.has(dateStr)) {
        // Use existing data if we have it
        filledData.push(existingDates.get(dateStr));
      } else {
        // Create an entry with zero values
        const emptyEntry: any = {
          date: dateStr,
          api_requests: 0,
          total_tokens: 0,
        };

        // Add zero values for each model/team if needed
        categories.forEach((category) => {
          if (!emptyEntry[category]) {
            emptyEntry[category] = 0;
          }
        });

        filledData.push(emptyEntry);
      }

      // Move to next day
      currentDate.setDate(currentDate.getDate() + 1);
    }

    return filledData;
  };

  // Update the fetchOverallSpend function
  const fetchOverallSpend = async () => {
    if (!accessToken) {
      return;
    }
    try {
      const data = await adminSpendLogsCall(accessToken);

      // Get the first and last day of the current month
      const now = new Date();
      const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
      const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);

      // Fill in missing dates
      const filledData = fillMissingDates(data, firstDay, lastDay, []);

      // Calculate total spend for the month and round to 2 decimal places
      const monthlyTotal = Number(filledData.reduce((sum, day) => sum + (day.spend || 0), 0).toFixed(2));
      setTotalMonthlySpend(monthlyTotal);

      setKeySpendData(filledData);
    } catch (error) {
      console.error("Error fetching overall spend:", error);
    }
  };

  const fetchProviderSpend = () =>
    fetchAndSetData(
      () =>
        accessToken && token
          ? adminspendByProvider(accessToken, token, startTime, endTime)
          : Promise.reject("No access token or token"),
      setSpendByProvider,
      "Error fetching provider spend",
    );

  const fetchTopKeys = async () => {
    if (!accessToken) return;
    await fetchAndSetData(
      async () => {
        const top_keys = await adminTopKeysCall(accessToken);
        return top_keys.map((k: any) => ({
          key: k["api_key"].substring(0, 10),
          api_key: k["api_key"],
          key_alias: k["key_alias"],
          spend: Number(k["total_spend"].toFixed(2)),
        }));
      },
      setTopKeys,
      "Error fetching top keys",
    );
  };

  const fetchTopModels = async () => {
    if (!accessToken) return;
    await fetchAndSetData(
      async () => {
        const top_models = await adminTopModelsCall(accessToken);
        return top_models.map((k: any) => ({
          key: k["model"],
          spend: formatNumberWithCommas(k["total_spend"], 2),
        }));
      },
      setTopModels,
      "Error fetching top models",
    );
  };

  // Update the fetchTeamSpend function
  const fetchTeamSpend = async () => {
    if (!accessToken) return;
    await fetchAndSetData(
      async () => {
        const teamSpend = await teamSpendLogsCall(accessToken);

        // Get the first and last day of the current month
        const now = new Date();
        const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
        const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);

        // Fill in missing dates with zero values for all teams
        const filledData = fillMissingDates(teamSpend.daily_spend, firstDay, lastDay, teamSpend.teams);

        setTeamSpendData(filledData);
        setUniqueTeamIds(teamSpend.teams);
        return teamSpend.total_spend_per_team.map((tspt: any) => ({
          name: tspt["team_id"] || "",
          value: formatNumberWithCommas(tspt["total_spend"] || 0, 2),
        }));
      },
      setTotalSpendPerTeam,
      "Error fetching team spend",
    );
  };

  const fetchTagNames = () => {
    if (!accessToken) return;
    fetchAndSetData(
      async () => {
        const all_tag_names = await allTagNamesCall(accessToken);
        return all_tag_names.tag_names;
      },
      setAllTagNames,
      "Error fetching tag names",
    );
  };

  const fetchTopTags = () => {
    if (!accessToken) return;
    fetchAndSetData(
      () => tagsSpendLogsCall(accessToken, dateValue.from?.toISOString(), dateValue.to?.toISOString(), undefined),
      (data) => setTopTagsData(data.spend_per_tag),
      "Error fetching top tags",
    );
  };

  const fetchTopEndUsers = () => {
    if (!accessToken) return;
    fetchAndSetData(
      () => adminTopEndUsersCall(accessToken, null, undefined, undefined),
      setTopUsers,
      "Error fetching top end users",
    );
  };

  // Update the fetchGlobalActivity function
  const fetchGlobalActivity = async () => {
    if (!accessToken) return;
    try {
      const data = await adminGlobalActivity(accessToken, startTime, endTime);

      // Get the date range from the current month
      const now = new Date();
      const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
      const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);

      // Fill in missing dates for daily_data
      const filledDailyData = fillMissingDates(data.daily_data || [], firstDay, lastDay, [
        "api_requests",
        "total_tokens",
      ]);

      setGlobalActivity({
        ...data,
        daily_data: filledDailyData,
      });
    } catch (error) {
      console.error("Error fetching global activity:", error);
    }
  };

  // Update the fetchGlobalActivityPerModel function
  const fetchGlobalActivityPerModel = async () => {
    if (!accessToken) return;
    try {
      const data = await adminGlobalActivityPerModel(accessToken, startTime, endTime);

      // Get the date range from the current month
      const now = new Date();
      const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
      const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);

      // Fill in missing dates for each model's daily data
      const filledModelData = data.map((modelData: any) => ({
        ...modelData,
        daily_data: fillMissingDates(modelData.daily_data || [], firstDay, lastDay, ["api_requests", "total_tokens"]),
      }));

      setGlobalActivityPerModel(filledModelData);
    } catch (error) {
      console.error("Error fetching global activity per model:", error);
    }
  };

  useEffect(() => {
    const initlizeUsageData = async () => {
      if (accessToken && token && userRole && userID) {
        const proxy_settings: ProxySettings | undefined = await fetchProxySettings();
        if (proxy_settings) {
          setProxySettings(proxy_settings); // saved in state so it can be used when rendering UI
          if (proxy_settings?.DISABLE_EXPENSIVE_DB_QUERIES) {
            return; // Don't run expensive UI queries - return out of initlizeUsageData at this point
          }
        }

        console.log("fetching data - valiue of proxySettings", proxySettings);

        fetchOverallSpend();
        fetchProviderSpend();
        fetchTopKeys();
        fetchTopModels();
        fetchGlobalActivity();
        fetchGlobalActivityPerModel();

        if (isAdminOrAdminViewer(userRole)) {
          fetchTeamSpend();
          fetchTagNames();
          fetchTopTags();
          fetchTopEndUsers();
        }
      }
    };

    initlizeUsageData();
  }, [accessToken, token, userRole, userID, startTime, endTime]);

  if (proxySettings?.DISABLE_EXPENSIVE_DB_QUERIES) {
    return (
      <div style={{ width: "100%" }} className="p-8">
        <Card>
          <Title>Database Query Limit Reached</Title>
          <Text className="mt-4">
            SpendLogs in DB has {proxySettings.NUM_SPEND_LOGS_ROWS} rows.
            <br></br>
            Please follow our guide to view usage when SpendLogs has more than 1M rows.
          </Text>
          <Button className="mt-4">
            <a href="https://docs.litellm.ai/docs/proxy/spending_monitoring" target="_blank">
              View Usage Guide
            </a>
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }} className="p-8">
      <TabGroup>
        <TabList className="mt-2">
          <Tab>All Up</Tab>

          {isAdminOrAdminViewer(userRole) ? (
            <>
              <Tab>Team Based Usage</Tab>
              <Tab>Customer Usage</Tab>
              <Tab>Tag Based Usage</Tab>
            </>
          ) : (
            <>
              <div></div>
            </>
          )}
        </TabList>
        <TabPanels>
          <TabPanel>
            <TabGroup>
              <TabList variant="solid" className="mt-1">
                <Tab>Cost</Tab>
                <Tab>Activity</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <Grid numItems={2} className="gap-2 h-[100vh] w-full">
                    <Col numColSpan={2}>
                      <Text className="text-tremor-default text-tremor-content dark:text-dark-tremor-content mb-2 mt-2 text-lg">
                        Project Spend {new Date().toLocaleString("default", { month: "long" })} 1 -{" "}
                        {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).getDate()}
                      </Text>
                      <ViewUserSpend userSpend={totalMonthlySpend} selectedTeam={null} userMaxBudget={null} />
                    </Col>
                    <Col numColSpan={2}>
                      <Card>
                        <Title>Monthly Spend</Title>
                        <BarChart
                          data={keySpendData}
                          index="date"
                          categories={["spend"]}
                          colors={["cyan"]}
                          valueFormatter={valueFormatter}
                          yAxisWidth={100}
                          tickGap={5}
                          // customTooltip={customTooltip}
                        />
                      </Card>
                    </Col>
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <Title>Top Virtual Keys</Title>
                        <TopKeyView topKeys={topKeys} teams={null} topKeysLimit={5} setTopKeysLimit={() => {}} />
                      </Card>
                    </Col>
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <Title>Top Models</Title>
                        <BarChart
                          className="mt-4 h-40"
                          data={topModels}
                          index="key"
                          categories={["spend"]}
                          colors={["cyan"]}
                          yAxisWidth={200}
                          layout="vertical"
                          showXAxis={false}
                          showLegend={false}
                          valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
                        />
                      </Card>
                    </Col>
                    <Col numColSpan={1}></Col>
                    <Col numColSpan={2}>
                      <Card className="mb-2">
                        <Title>Spend by Provider</Title>
                        <>
                          <Grid numItems={2}>
                            <Col numColSpan={1}>
                              <DonutChart
                                className="mt-4 h-40"
                                variant="pie"
                                data={spendByProvider}
                                index="provider"
                                category="spend"
                                colors={["cyan"]}
                                valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
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
                                  {spendByProvider.map((provider) => (
                                    <TableRow key={provider.provider}>
                                      <TableCell>{provider.provider}</TableCell>
                                      <TableCell>
                                        {parseFloat(provider.spend.toFixed(2)) < 0.00001
                                          ? "less than 0.00"
                                          : formatNumberWithCommas(provider.spend, 2)}
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </Col>
                          </Grid>
                        </>
                      </Card>
                    </Col>
                  </Grid>
                </TabPanel>
                <TabPanel>
                  <Grid numItems={1} className="gap-2 h-[75vh] w-full">
                    <Card>
                      <Title>All Up</Title>
                      <Grid numItems={2}>
                        <Col>
                          <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452" }}>
                            API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}
                          </Subtitle>
                          <AreaChart
                            className="h-40"
                            data={globalActivity.daily_data}
                            valueFormatter={valueFormatterNumbers}
                            index="date"
                            colors={["cyan"]}
                            categories={["api_requests"]}
                            onValueChange={(v) => console.log(v)}
                          />
                        </Col>
                        <Col>
                          <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452" }}>
                            Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}
                          </Subtitle>
                          <BarChart
                            className="h-40"
                            data={globalActivity.daily_data}
                            valueFormatter={valueFormatterNumbers}
                            index="date"
                            colors={["cyan"]}
                            categories={["total_tokens"]}
                            onValueChange={(v) => console.log(v)}
                          />
                        </Col>
                      </Grid>
                    </Card>

                    <>
                      {globalActivityPerModel.map((globalActivity, index) => (
                        <Card key={index}>
                          <Title>{globalActivity.model}</Title>
                          <Grid numItems={2}>
                            <Col>
                              <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452" }}>
                                API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}
                              </Subtitle>
                              <AreaChart
                                className="h-40"
                                data={globalActivity.daily_data}
                                index="date"
                                colors={["cyan"]}
                                categories={["api_requests"]}
                                valueFormatter={valueFormatterNumbers}
                                onValueChange={(v) => console.log(v)}
                              />
                            </Col>
                            <Col>
                              <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452" }}>
                                Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}
                              </Subtitle>
                              <BarChart
                                className="h-40"
                                data={globalActivity.daily_data}
                                index="date"
                                colors={["cyan"]}
                                categories={["total_tokens"]}
                                valueFormatter={valueFormatterNumbers}
                                onValueChange={(v) => console.log(v)}
                              />
                            </Col>
                          </Grid>
                        </Card>
                      ))}
                    </>
                  </Grid>
                </TabPanel>
              </TabPanels>
            </TabGroup>
          </TabPanel>
          <TabPanel>
            <Grid numItems={2} className="gap-2 h-[75vh] w-full">
              <Col numColSpan={2}>
                <Card className="mb-2">
                  <Title>Total Spend Per Team</Title>
                  <BarList data={totalSpendPerTeam} />
                </Card>
                <Card>
                  <Title>Daily Spend Per Team</Title>
                  <BarChart
                    className="h-72"
                    data={teamSpendData}
                    showLegend={true}
                    index="date"
                    categories={uniqueTeamIds}
                    yAxisWidth={80}
                    stack={true}
                  />
                </Card>
              </Col>
              <Col numColSpan={2}></Col>
            </Grid>
          </TabPanel>
          <TabPanel>
            <p className="mb-2 text-gray-500 italic text-[12px]">
              Customers of your LLM API calls. Tracked when a `user` param is passed in your LLM calls{" "}
              <a className="text-blue-500" href="https://docs.litellm.ai/docs/proxy/users" target="_blank">
                docs here
              </a>
            </p>
            <Grid numItems={2}>
              <Col>
                <UsageDatePicker
                  value={dateValue}
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateEndUserData(value.from, value.to, null);
                  }}
                />
              </Col>
              <Col>
                <Text>Select Key</Text>
                <Select defaultValue="all-keys">
                  <SelectItem
                    key="all-keys"
                    value="all-keys"
                    onClick={() => {
                      updateEndUserData(dateValue.from, dateValue.to, null);
                    }}
                  >
                    All Keys
                  </SelectItem>
                  {keys?.map((key: any, index: number) => {
                    if (key && key["key_alias"] !== null && key["key_alias"].length > 0) {
                      return (
                        <SelectItem
                          key={index}
                          value={String(index)}
                          onClick={() => {
                            updateEndUserData(dateValue.from, dateValue.to, key["token"]);
                          }}
                        >
                          {key["key_alias"]}
                        </SelectItem>
                      );
                    }
                    return null; // Add this line to handle the case when the condition is not met
                  })}
                </Select>
              </Col>
            </Grid>

            <Card className="mt-4">
              <Table className="max-h-[70vh] min-h-[500px]">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Customer</TableHeaderCell>
                    <TableHeaderCell>Spend</TableHeaderCell>
                    <TableHeaderCell>Total Events</TableHeaderCell>
                  </TableRow>
                </TableHead>

                <TableBody>
                  {topUsers?.map((user: any, index: number) => (
                    <TableRow key={index}>
                      <TableCell>{user.end_user}</TableCell>
                      <TableCell>{formatNumberWithCommas(user.total_spend, 2)}</TableCell>
                      <TableCell>{user.total_count}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </TabPanel>
          <TabPanel>
            <Grid numItems={2}>
              <Col numColSpan={1}>
                <UsageDatePicker
                  className="mb-4"
                  value={dateValue}
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateTagSpendData(value.from, value.to);
                  }}
                />
              </Col>

              <Col>
                {premiumUser ? (
                  <div>
                    <MultiSelect value={selectedTags} onValueChange={(value) => setSelectedTags(value as string[])}>
                      <MultiSelectItem
                        key={"all-tags"}
                        value={"all-tags"}
                        onClick={() => setSelectedTags(["all-tags"])}
                      >
                        All Tags
                      </MultiSelectItem>
                      {allTagNames &&
                        allTagNames
                          .filter((tag) => tag !== "all-tags")
                          .map((tag: any, index: number) => {
                            return (
                              <MultiSelectItem key={tag} value={String(tag)}>
                                {tag}
                              </MultiSelectItem>
                            );
                          })}
                    </MultiSelect>
                  </div>
                ) : (
                  <div>
                    <MultiSelect value={selectedTags} onValueChange={(value) => setSelectedTags(value as string[])}>
                      <MultiSelectItem
                        key={"all-tags"}
                        value={"all-tags"}
                        onClick={() => setSelectedTags(["all-tags"])}
                      >
                        All Tags
                      </MultiSelectItem>
                      {allTagNames &&
                        allTagNames
                          .filter((tag) => tag !== "all-tags")
                          .map((tag: any, index: number) => {
                            return (
                              <SelectItem
                                key={tag}
                                value={String(tag)}
                                // @ts-ignore
                                disabled={true}
                              >
                                ✨ {tag} (Enterprise only Feature)
                              </SelectItem>
                            );
                          })}
                    </MultiSelect>
                  </div>
                )}
              </Col>
            </Grid>
            <Grid numItems={2} className="gap-2 h-[75vh] w-full mb-4">
              <Col numColSpan={2}>
                <Card>
                  <Title>Spend Per Tag</Title>
                  <Text>
                    Get Started by Tracking cost per tag{" "}
                    <a
                      className="text-blue-500"
                      href="https://docs.litellm.ai/docs/proxy/cost_tracking"
                      target="_blank"
                    >
                      here
                    </a>
                  </Text>
                  <BarChart
                    className="h-72"
                    data={topTagsData}
                    index="name"
                    categories={["spend"]}
                    colors={["cyan"]}
                  ></BarChart>
                </Card>
              </Col>
              <Col numColSpan={2}></Col>
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default UsagePage;
