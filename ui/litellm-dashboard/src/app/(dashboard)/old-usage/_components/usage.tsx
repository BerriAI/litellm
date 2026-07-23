import React, { useState, useEffect } from "react";

import ViewUserSpend from "@/components/view_user_spend";
import { ProxySettings } from "@/components/user_dashboard";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
} from "@/components/ui/combobox";
import { Meter, MeterIndicator, MeterTrack } from "@/components/ui/meter";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AreaChart, BarChart, DonutChart } from "@/components/shared/charts";

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
} from "@/components/networking";
import TopKeyView from "@/components/UsagePage/components/EntityUsage/TopKeyView";
import { MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";

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

type UsageDateRange = { from?: Date; to?: Date };

type TeamSpendTotal = { name: string; value: number };

type TagOption = { value: string; label: string; disabled: boolean };

const ALL_TAGS = "all-tags";

const isAdminOrAdminViewer = (role: string | null): boolean => {
  if (role === null) return false;
  return role === "Admin" || role === "Admin Viewer";
};

const TeamSpendBarList: React.FC<{ data: TeamSpendTotal[] }> = ({ data }) => {
  const max = Math.max(0, ...data.map((team) => team.value));

  return (
    <div className="flex flex-col gap-3">
      {data.map((team) => (
        <div key={team.name} className="flex items-center gap-4">
          <p className="w-1/3 truncate text-sm text-foreground">{team.name}</p>
          <Meter value={team.value} max={max === 0 ? 1 : max} className="flex-1">
            <MeterTrack>
              <MeterIndicator />
            </MeterTrack>
          </Meter>
          <p className="w-24 shrink-0 text-right text-sm tabular-nums text-foreground">
            {formatNumberWithCommas(team.value, 2)}
          </p>
        </div>
      ))}
    </div>
  );
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
  const [totalSpendPerTeam, setTotalSpendPerTeam] = useState<TeamSpendTotal[]>([]);
  const [spendByProvider, setSpendByProvider] = useState<any[]>([]);
  const [globalActivity, setGlobalActivity] = useState<GlobalActivityData>({} as GlobalActivityData);
  const [globalActivityPerModel, setGlobalActivityPerModel] = useState<any[]>([]);
  const [selectedKeyToken, setSelectedKeyToken] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([ALL_TAGS]);
  const [dateValue, setDateValue] = useState<UsageDateRange>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });
  const [proxySettings, setProxySettings] = useState<ProxySettings | null>(null);
  const [totalMonthlySpend, setTotalMonthlySpend] = useState<number>(0);

  const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
  const lastDay = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);

  let startTime = formatDate(firstDay);
  let endTime = formatDate(lastDay);

  const selectableKeys: { token: string; alias: string }[] = (keys ?? [])
    .filter((key: any) => key && typeof key["key_alias"] === "string" && key["key_alias"].length > 0)
    .map((key: any) => ({ token: String(key["token"]), alias: String(key["key_alias"]) }));

  const tagOptions: TagOption[] = [
    { value: ALL_TAGS, label: "All Tags", disabled: false },
    ...allTagNames
      .filter((tag) => tag !== ALL_TAGS)
      .map((tag) => ({
        value: tag,
        label: premiumUser ? tag : `✨ ${tag} (Enterprise only Feature)`,
        disabled: !premiumUser,
      })),
  ];

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

    let newTopUserData = await adminTopEndUsersCall(
      accessToken,
      uiSelectedKey,
      startTime.toISOString(),
      endTime.toISOString(),
    );
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
          value: Number(tspt["total_spend"] || 0),
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
      <div className="w-full p-8">
        <Card>
          <CardHeader>
            <CardTitle>Database Query Limit Reached</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-start gap-4">
            <p className="text-sm text-muted-foreground">
              SpendLogs in DB has {proxySettings.NUM_SPEND_LOGS_ROWS} rows.
              <br></br>
              Please follow our guide to view usage when SpendLogs has more than 1M rows.
            </p>
            <Button
              render={
                <a href="https://docs.litellm.ai/docs/proxy/cost_tracking" target="_blank" rel="noreferrer">
                  View Usage Guide
                </a>
              }
            />
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full p-8">
      <Tabs defaultValue="all-up">
        <TabsList variant="line" className="mt-2">
          <TabsTrigger value="all-up">All Up</TabsTrigger>

          {isAdminOrAdminViewer(userRole) && (
            <>
              <TabsTrigger value="team-based-usage">Team Based Usage</TabsTrigger>
              <TabsTrigger value="customer-usage">Customer Usage</TabsTrigger>
              <TabsTrigger value="tag-based-usage">Tag Based Usage</TabsTrigger>
            </>
          )}
        </TabsList>

        <TabsContent value="all-up">
          <Tabs defaultValue="cost">
            <TabsList className="mt-1">
              <TabsTrigger value="cost">Cost</TabsTrigger>
              <TabsTrigger value="activity">Activity</TabsTrigger>
            </TabsList>

            <TabsContent value="cost">
              <div className="grid h-screen w-full grid-cols-2 gap-2">
                <div className="col-span-2">
                  <p className="mt-2 mb-2 text-lg text-muted-foreground">
                    Project Spend {new Date().toLocaleString("default", { month: "long" })} 1 -{" "}
                    {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).getDate()}
                  </p>
                  <ViewUserSpend userSpend={totalMonthlySpend} selectedTeam={null} userMaxBudget={null} />
                </div>
                <div className="col-span-2">
                  <Card>
                    <CardHeader>
                      <CardTitle>Monthly Spend</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <BarChart
                        data={keySpendData}
                        index="date"
                        categories={["spend"]}
                        colors={["cyan"]}
                        valueFormatter={valueFormatter}
                        yAxisWidth={100}
                        tickGap={5}
                      />
                    </CardContent>
                  </Card>
                </div>
                <div className="col-span-1">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Top Virtual Keys</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <TopKeyView topKeys={topKeys} teams={null} topKeysLimit={5} setTopKeysLimit={() => {}} />
                    </CardContent>
                  </Card>
                </div>
                <div className="col-span-1">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Top Models</CardTitle>
                    </CardHeader>
                    <CardContent>
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
                    </CardContent>
                  </Card>
                </div>
                <div className="col-span-1"></div>
                <div className="col-span-2">
                  <Card className="mb-2">
                    <CardHeader>
                      <CardTitle>Spend by Provider</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2">
                        <div className="col-span-1">
                          <DonutChart
                            className="mt-4 h-40"
                            variant="pie"
                            data={spendByProvider}
                            index="provider"
                            category="spend"
                            colors={["cyan"]}
                            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
                          />
                        </div>
                        <div className="col-span-1">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Provider</TableHead>
                                <TableHead>Spend</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {spendByProvider.map((provider) => (
                                <TableRow key={provider.provider}>
                                  <TableCell>{provider.provider}</TableCell>
                                  <TableCell>
                                    <MoneyCell value={provider.spend} decimals={2} />
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="activity">
              <div className="grid h-[75vh] w-full grid-cols-1 gap-2">
                <Card>
                  <CardHeader>
                    <CardTitle>All Up</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2">
                      <div>
                        <p className="text-[15px] font-normal text-muted-foreground">
                          API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}
                        </p>
                        <AreaChart
                          className="h-40"
                          data={globalActivity.daily_data}
                          valueFormatter={valueFormatterNumbers}
                          index="date"
                          colors={["cyan"]}
                          categories={["api_requests"]}
                        />
                      </div>
                      <div>
                        <p className="text-[15px] font-normal text-muted-foreground">
                          Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}
                        </p>
                        <BarChart
                          className="h-40"
                          data={globalActivity.daily_data}
                          valueFormatter={valueFormatterNumbers}
                          index="date"
                          colors={["cyan"]}
                          categories={["total_tokens"]}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {globalActivityPerModel.map((globalActivity, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <CardTitle>{globalActivity.model}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2">
                        <div>
                          <p className="text-[15px] font-normal text-muted-foreground">
                            API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}
                          </p>
                          <AreaChart
                            className="h-40"
                            data={globalActivity.daily_data}
                            index="date"
                            colors={["cyan"]}
                            categories={["api_requests"]}
                            valueFormatter={valueFormatterNumbers}
                          />
                        </div>
                        <div>
                          <p className="text-[15px] font-normal text-muted-foreground">
                            Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}
                          </p>
                          <BarChart
                            className="h-40"
                            data={globalActivity.daily_data}
                            index="date"
                            colors={["cyan"]}
                            categories={["total_tokens"]}
                            valueFormatter={valueFormatterNumbers}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="team-based-usage">
          <div className="grid h-[75vh] w-full grid-cols-2 gap-2">
            <div className="col-span-2">
              <Card className="mb-2">
                <CardHeader>
                  <CardTitle>Total Spend Per Team</CardTitle>
                </CardHeader>
                <CardContent>
                  <TeamSpendBarList data={totalSpendPerTeam} />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Daily Spend Per Team</CardTitle>
                </CardHeader>
                <CardContent>
                  <BarChart
                    className="h-72"
                    data={teamSpendData}
                    showLegend={true}
                    index="date"
                    categories={uniqueTeamIds}
                    yAxisWidth={80}
                    stack={true}
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="customer-usage">
          <p className="mb-2 text-[12px] text-muted-foreground italic">
            Customers of your LLM API calls. Tracked when a `user` param is passed in your LLM calls{" "}
            <a
              className="text-primary"
              href="https://docs.litellm.ai/docs/proxy/users"
              target="_blank"
              rel="noreferrer"
            >
              docs here
            </a>
          </p>
          <div className="grid grid-cols-2">
            <div>
              <UsageDatePicker
                value={dateValue}
                onValueChange={(value) => {
                  setDateValue(value);
                  updateEndUserData(value.from, value.to, null);
                }}
              />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Select Key</p>
              <Select
                value={selectedKeyToken}
                onValueChange={(value: string | null) => {
                  setSelectedKeyToken(value);
                  updateEndUserData(dateValue.from, dateValue.to, value);
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="All Keys">
                    {(token: string | null) => selectableKeys.find((key) => key.token === token)?.alias ?? "All Keys"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={null}>All Keys</SelectItem>
                  {selectableKeys.map((key) => (
                    <SelectItem key={key.token} value={key.token}>
                      {key.alias}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Card className="mt-4">
            <CardContent>
              <div className="max-h-[70vh] min-h-[500px] overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Customer</TableHead>
                      <TableHead>Spend</TableHead>
                      <TableHead>Total Events</TableHead>
                    </TableRow>
                  </TableHeader>

                  <TableBody>
                    {topUsers?.map((user: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>{user.end_user}</TableCell>
                        <TableCell>
                          <MoneyCell value={user.total_spend} decimals={2} />
                        </TableCell>
                        <TableCell>{user.total_count}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tag-based-usage">
          <div className="grid grid-cols-2">
            <div className="col-span-1">
              <UsageDatePicker
                className="mb-4"
                value={dateValue}
                onValueChange={(value) => {
                  setDateValue(value);
                  updateTagSpendData(value.from, value.to);
                }}
              />
            </div>

            <div>
              <Combobox
                multiple
                items={tagOptions}
                value={tagOptions.filter((option) => selectedTags.includes(option.value))}
                onValueChange={(options: TagOption[]) => setSelectedTags(options.map((option) => option.value))}
                isItemEqualToValue={(a: TagOption, b: TagOption) => a.value === b.value}
                itemToStringLabel={(option: TagOption) => option.label}
              >
                <ComboboxChips>
                  <ComboboxValue>
                    {(options: TagOption[]) =>
                      options.map((option) => (
                        <ComboboxChip key={option.value} aria-label={option.label}>
                          {option.label}
                        </ComboboxChip>
                      ))
                    }
                  </ComboboxValue>
                  <ComboboxChipsInput placeholder="Select tags" className="border-0 bg-transparent" />
                </ComboboxChips>
                <ComboboxContent>
                  <ComboboxEmpty>No tags found</ComboboxEmpty>
                  <ComboboxList>
                    {(option: TagOption) => (
                      <ComboboxItem key={option.value} value={option} disabled={option.disabled}>
                        {option.label}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
          </div>
          <div className="mb-4 grid h-[75vh] w-full grid-cols-2 gap-2">
            <div className="col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Spend Per Tag</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-2">
                  <p className="text-sm text-muted-foreground">
                    Get Started by Tracking cost per tag{" "}
                    <a
                      className="text-primary"
                      href="https://docs.litellm.ai/docs/proxy/cost_tracking"
                      target="_blank"
                      rel="noreferrer"
                    >
                      here
                    </a>
                  </p>
                  <BarChart className="h-72" data={topTagsData} index="name" categories={["spend"]} colors={["cyan"]} />
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default UsagePage;
