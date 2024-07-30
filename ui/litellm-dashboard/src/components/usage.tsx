import { BarChart, BarList, Card, Title, Table, TableHead, TableHeaderCell, TableRow, TableCell, TableBody, Metric, Subtitle } from "@tremor/react";

import React, { useState, useEffect } from "react";

import ViewUserSpend from "./view_user_spend";
import { 
  Grid, Col, Text, 
  LineChart, TabPanel, TabPanels, 
  TabGroup, TabList, Tab, Select, SelectItem, 
  DateRangePicker, DateRangePickerValue, 
  DonutChart,
  AreaChart,
  Callout,
  Button,
  MultiSelect,
  MultiSelectItem,
} from "@tremor/react";

import {
  Select as Select2
} from "antd";

import {
  userSpendLogsCall,
  keyInfoCall,
  adminSpendLogsCall,
  adminTopKeysCall,
  adminTopModelsCall,
  adminTopEndUsersCall,
  teamSpendLogsCall,
  tagsSpendLogsCall,
  allTagNamesCall,
  modelMetricsCall,
  modelAvailableCall,
  adminspendByProvider,
  adminGlobalActivity,
  adminGlobalActivityPerModel,
} from "./networking";
import { start } from "repl";

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
  // Convert the object into an array of key-value pairs
  const entries: [string, number][] = Object.entries(model_values).map(
    ([key, value]) => [key, value as number]
  ); // Type assertion to specify the value as number

  // Sort the array based on the float value in descending order
  entries.sort((a, b) => b[1] - a[1]);

  // Get the top 5 key-value pairs
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
                {value ? (value < 0.01 ? "<$0.01" : value.toFixed(2)) : ""}
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
      if (
        key !== "spend" &&
        key !== "startTime" &&
        key !== "models" &&
        key !== "users"
      ) {
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


const UsagePage: React.FC<UsagePageProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  keys,
  premiumUser,
}) => {
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

  const firstDay = new Date(
    currentDate.getFullYear(),
    currentDate.getMonth(),
    1
  );
  const lastDay = new Date(
    currentDate.getFullYear(),
    currentDate.getMonth() + 1,
    0
  );

  let startTime = formatDate(firstDay);
  let endTime = formatDate(lastDay);

  console.log("keys in usage", keys);
  console.log("premium user in usage", premiumUser);

  function valueFormatterNumbers(number: number) {
    const formatter = new Intl.NumberFormat('en-US', {
      maximumFractionDigits: 0,
      notation: 'compact',
      compactDisplay: 'short',
    });
  
    return formatter.format(number);
  }

  useEffect(() => {
    updateTagSpendData(dateValue.from, dateValue.to);
  }, [dateValue, selectedTags]);
  

  const updateEndUserData = async (startTime:  Date | undefined, endTime:  Date | undefined, uiSelectedKey: string | null) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    // the endTime put it to the last hour of the selected date
    endTime.setHours(23, 59, 59, 999);

    // startTime put it to the first hour of the selected date
    startTime.setHours(0, 0, 0, 0);

    console.log("uiSelectedKey", uiSelectedKey);

    let newTopUserData = await adminTopEndUsersCall(
      accessToken,
      uiSelectedKey,
      startTime.toISOString(),
      endTime.toISOString()
    )
    console.log("End user data updated successfully", newTopUserData);
    setTopUsers(newTopUserData);
  
  }

  const updateTagSpendData = async (startTime:  Date | undefined, endTime:  Date | undefined) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    // the endTime put it to the last hour of the selected date
    endTime.setHours(23, 59, 59, 999);

    // startTime put it to the first hour of the selected date
    startTime.setHours(0, 0, 0, 0);

    let top_tags = await tagsSpendLogsCall(
      accessToken, 
      startTime.toISOString(), 
      endTime.toISOString(),
      selectedTags.length === 0 ? undefined : selectedTags
    );
    setTopTagsData(top_tags.spend_per_tag);
    console.log("Tag spend data updated successfully");

  }

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

  const valueFormatter = (number: number) =>
    `$ ${new Intl.NumberFormat("us").format(number).toString()}`;

  useEffect(() => {
    if (accessToken && token && userRole && userID) {
      const fetchData = async () => {
        try {
          /**
           * If user is Admin - query the global views endpoints
           * If user is App Owner - use the normal spend logs call
           */
          console.log(`user role: ${userRole}`);
          if (userRole == "Admin" || userRole == "Admin Viewer") {
            const overall_spend = await adminSpendLogsCall(accessToken);
            setKeySpendData(overall_spend);

            const provider_spend = await adminspendByProvider(accessToken, token, startTime, endTime);
            console.log("provider_spend", provider_spend);
            setSpendByProvider(provider_spend);


            const top_keys = await adminTopKeysCall(accessToken);
            const filtered_keys = top_keys.map((k: any) => ({
              key: (k["key_alias"] || k["key_name"] || k["api_key"]).substring(
                0,
                10
              ),
              spend: k["total_spend"],
            }));
            setTopKeys(filtered_keys);
            const top_models = await adminTopModelsCall(accessToken);
            const filtered_models = top_models.map((k: any) => ({
              key: k["model"],
              spend: k["total_spend"],
            }));
            setTopModels(filtered_models);

            const teamSpend = await teamSpendLogsCall(accessToken);
            console.log("teamSpend", teamSpend);
            setTeamSpendData(teamSpend.daily_spend);
            setUniqueTeamIds(teamSpend.teams)

            let total_spend_per_team = teamSpend.total_spend_per_team;
            // in total_spend_per_team, replace null team_id with "" and replace null total_spend with 0

            total_spend_per_team = total_spend_per_team.map((tspt: any) => {
              tspt["name"] = tspt["team_id"] || "";
              tspt["value"] = tspt["total_spend"] || 0;
              // round the value to 2 decimal places

              tspt["value"] = tspt["value"].toFixed(2);
              

              return tspt;
            })

            setTotalSpendPerTeam(total_spend_per_team);

            // all_tag_names -> used for dropdown
            const all_tag_names = await allTagNamesCall(accessToken);
            setAllTagNames(all_tag_names.tag_names);

            //get top tags
            const top_tags = await tagsSpendLogsCall(accessToken, dateValue.from?.toISOString(), dateValue.to?.toISOString(), undefined);
            setTopTagsData(top_tags.spend_per_tag);


            // get spend per end-user
            let spend_user_call = await adminTopEndUsersCall(accessToken, null, undefined, undefined);
            setTopUsers(spend_user_call);

            console.log("spend/user result", spend_user_call);

            let global_activity_response = await adminGlobalActivity(accessToken, startTime, endTime);
            setGlobalActivity(global_activity_response)

            let global_activity_per_model = await adminGlobalActivityPerModel(accessToken, startTime, endTime);
            console.log("global activity per model", global_activity_per_model);
            setGlobalActivityPerModel(global_activity_per_model)


          } else if (userRole == "App Owner") {
            await userSpendLogsCall(
              accessToken,
              token,
              userRole,
              userID,
              startTime,
              endTime
            ).then(async (response) => {
              console.log("result from spend logs call", response);
              if ("daily_spend" in response) {
                // this is from clickhouse analytics
                //
                let daily_spend = response["daily_spend"];
                console.log("daily spend", daily_spend);
                setKeySpendData(daily_spend);
                let topApiKeys = response.top_api_keys;
                setTopKeys(topApiKeys);
              } else {
                const topKeysResponse = await keyInfoCall(
                  accessToken,
                  getTopKeys(response)
                );
                const filtered_keys = topKeysResponse["info"].map((k: any) => ({
                  key: (
                    k["key_name"] ||
                    k["key_alias"]
                  ).substring(0, 10),
                  spend: k["spend"],
                }));
                setTopKeys(filtered_keys);
                setKeySpendData(response);
              }
            });
          }
        } catch (error) {
          console.error("There was an error fetching the data", error);
          // Optionally, update your UI to reflect the error state here as well
        }
      };
      fetchData();
    }
  }, [accessToken, token, userRole, userID, startTime, endTime]);


  return (
    <div style={{ width: "100%" }} className="p-8">
      
      <TabGroup>
        <TabList className="mt-2">
          <Tab>All Up</Tab>
          <Tab>Team Based Usage</Tab>
          <Tab>Customer Usage</Tab>
           <Tab>Tag Based Usage</Tab>
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
            <ViewUserSpend
            userID={userID}
            userRole={userRole}
            accessToken={accessToken}
            userSpend={null}
            selectedTeam={null}
          />
              <Col numColSpan={2}>
                <Card>
                  <Title>Monthly Spend</Title>
                  <BarChart
                    data={keySpendData}
                    index="date"
                    categories={["spend"]}
                    colors={["blue"]}
                    valueFormatter={valueFormatter}
                    yAxisWidth={100}
                    tickGap={5}
                    // customTooltip={customTooltip}
                  />
                </Card>
              </Col>
              <Col numColSpan={1}>
                <Card>
                  <Title>Top API Keys</Title>
                  <BarChart
                    className="mt-4 h-40"
                    data={topKeys}
                    index="key"
                    categories={["spend"]}
                    colors={["blue"]}
                    yAxisWidth={80}
                    tickGap={5}
                    layout="vertical"
                    showXAxis={false}
                    showLegend={false}
                  />
                </Card>
              </Col>
              <Col numColSpan={1}>
              <Card>
                  <Title>Top Models</Title>
                  <BarChart
                    className="mt-4 h-40"
                    data={topModels}
                    index="key"
                    categories={["spend"]}
                    colors={["blue"]}
                    yAxisWidth={200}
                    layout="vertical"
                    showXAxis={false}
                    showLegend={false}
                  />
                </Card>
               
              </Col>
              <Col numColSpan={1}>
                
              </Col>
              <Col numColSpan={2}>
              <Card className="mb-2">
                <Title>✨ Spend by Provider</Title>
                {
                  premiumUser ? (
                    <>
                    <Grid numItems={2}>
                  <Col numColSpan={1}>
                    <DonutChart
                      className="mt-4 h-40"
                      variant="pie"
                      data={spendByProvider}
                      index="provider"
                      category="spend"
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
                                : provider.spend.toFixed(2)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Col>
                </Grid>
                    </>
                  ) : (
                    <div>
                    <p className="mb-2 text-gray-500 italic text-[12px]">Upgrade to use this feature</p>
                    <Button variant="primary" className="mb-2">
                          <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                            Get Free Trial
                          </a>
                        </Button>
                    </div>
                  )
                }
                
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
                <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>API Requests { valueFormatterNumbers(globalActivity.sum_api_requests)}</Subtitle>
                <AreaChart
                    className="h-40"
                    data={globalActivity.daily_data}
                    valueFormatter={valueFormatterNumbers}
                    index="date"
                    colors={['cyan']}
                    categories={['api_requests']}
                    onValueChange={(v) => console.log(v)}
                  />

                </Col>
                <Col>
                <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>Tokens { valueFormatterNumbers(globalActivity.sum_total_tokens)}</Subtitle>
                <BarChart
                    className="h-40"
                    data={globalActivity.daily_data}
                    valueFormatter={valueFormatterNumbers}
                    index="date"
                    colors={['cyan']}
                    categories={['total_tokens']}
                    onValueChange={(v) => console.log(v)}
                  />
                </Col>
                </Grid>
                

                </Card>

                {
                  premiumUser ? ( 
                    <>
                    {globalActivityPerModel.map((globalActivity, index) => (
                <Card key={index}>
                  <Title>{globalActivity.model}</Title>
                  <Grid numItems={2}>
                    <Col>
                      <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}</Subtitle>
                      <AreaChart
                        className="h-40"
                        data={globalActivity.daily_data}
                        index="date"
                        colors={['cyan']}
                        categories={['api_requests']}
                        valueFormatter={valueFormatterNumbers}
                        onValueChange={(v) => console.log(v)}
                      />
                    </Col>
                    <Col>
                      <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}</Subtitle>
                      <BarChart
                        className="h-40"
                        data={globalActivity.daily_data}
                        index="date"
                        colors={['cyan']}
                        categories={['total_tokens']}
                        valueFormatter={valueFormatterNumbers}
                        onValueChange={(v) => console.log(v)}
                      />
                    </Col>
                  </Grid>
                </Card>
              ))}
                    </>
                  ) : 
                  <>
                  {globalActivityPerModel && globalActivityPerModel.length > 0 &&
                    globalActivityPerModel.slice(0, 1).map((globalActivity, index) => (
                      <Card key={index}>
                        <Title>✨ Activity by Model</Title>
                        <p className="mb-2 text-gray-500 italic text-[12px]">Upgrade to see analytics for all models</p>
                        <Button variant="primary" className="mb-2">
                          <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                            Get Free Trial
                          </a>
                        </Button>
                        <Card>
                        <Title>{globalActivity.model}</Title>
                        <Grid numItems={2}>
                          <Col>
                            <Subtitle
                              style={{
                                fontSize: "15px",
                                fontWeight: "normal",
                                color: "#535452",
                              }}
                            >
                              API Requests {valueFormatterNumbers(globalActivity.sum_api_requests)}
                            </Subtitle>
                            <AreaChart
                              className="h-40"
                              data={globalActivity.daily_data}
                              index="date"
                              colors={['cyan']}
                              categories={['api_requests']}
                              valueFormatter={valueFormatterNumbers}
                              onValueChange={(v) => console.log(v)}
                            />
                          </Col>
                          <Col>
                            <Subtitle
                              style={{
                                fontSize: "15px",
                                fontWeight: "normal",
                                color: "#535452",
                              }}
                            >
                              Tokens {valueFormatterNumbers(globalActivity.sum_total_tokens)}
                            </Subtitle>
                            <BarChart
                              className="h-40"
                              data={globalActivity.daily_data}
                              index="date"
                              colors={['cyan']}
                              valueFormatter={valueFormatterNumbers}
                              categories={['total_tokens']}
                              onValueChange={(v) => console.log(v)}
                            />
                          </Col>
                          
                        </Grid>
                        </Card>
                      </Card>
                    ))}
                </>
                }              
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
                <BarList
                  data={totalSpendPerTeam}
                  
                />
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
              <Col numColSpan={2}>
              </Col>
            </Grid>
            </TabPanel>
            <TabPanel>
            <p className="mb-2 text-gray-500 italic text-[12px]">Customers of your LLM API calls. Tracked when a `user` param is passed in your LLM calls <a className="text-blue-500" href="https://docs.litellm.ai/docs/proxy/users" target="_blank">docs here</a></p>
              <Grid numItems={2}>
                <Col>
                <Text>Select Time Range</Text>
       
              <DateRangePicker 
                  enableSelect={true} 
                  value={dateValue} 
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateEndUserData(value.from, value.to, null); // Call updateModelMetrics with the new date range
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
                      if (
                        key &&
                        key["key_alias"] !== null &&
                        key["key_alias"].length > 0
                      ) {
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
                        <TableCell>{user.total_spend?.toFixed(4)}</TableCell>
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
            <DateRangePicker 
                  className="mb-4"
                  enableSelect={true} 
                  value={dateValue} 
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateTagSpendData(value.from, value.to); // Call updateModelMetrics with the new date range
                  }}
              />

              </Col>

              <Col>
                  {
                    premiumUser ? (
                      <div>
                        <MultiSelect
                            value={selectedTags}
                            onValueChange={(value) => setSelectedTags(value as string[])}
                          >
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
                                <MultiSelectItem
                                  key={tag}
                                  value={String(tag)}
                                >
                                  {tag}
                                </MultiSelectItem>
                              );
                            })}
                      </MultiSelect>

                      </div>

                    ) : (
                      <div>

<MultiSelect
                            value={selectedTags}
                            onValueChange={(value) => setSelectedTags(value as string[])}
                          >
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
                    )
                  }
  
              </Col>

              </Grid>
            <Grid numItems={2} className="gap-2 h-[75vh] w-full mb-4">
            

              <Col numColSpan={2}>

              <Card>
              <Title>Spend Per Tag</Title>
              <Text>Get Started Tracking cost per tag <a className="text-blue-500" href="https://docs.litellm.ai/docs/proxy/cost_tracking" target="_blank">here</a></Text>
             <BarChart
              className="h-72"
              data={topTagsData}
              index="name"
              categories={["spend"]}
              colors={["blue"]}
             >

             </BarChart>
              </Card>
              </Col>
              <Col numColSpan={2}>
              </Col>
            </Grid>
            </TabPanel>
            
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default UsagePage;
