import { BarChart, BarList, Card, Title, Table, TableHead, TableHeaderCell, TableRow, TableCell, TableBody, Metric } from "@tremor/react";

import React, { useState, useEffect } from "react";

import ViewUserSpend from "./view_user_spend";
import { Grid, Col, Text, LineChart, TabPanel, TabPanels, TabGroup, TabList, Tab, Select, SelectItem } from "@tremor/react";
import {
  userSpendLogsCall,
  keyInfoCall,
  adminSpendLogsCall,
  adminTopKeysCall,
  adminTopModelsCall,
  teamSpendLogsCall,
  tagsSpendLogsCall,
  modelMetricsCall,
  modelAvailableCall,
  modelInfoCall,
} from "./networking";
import { start } from "repl";

interface UsagePageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
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
function getTopUsers(data: Array<DataDict>): UserData[] {
  const userSpend: { [key: string]: number } = {};

  data.forEach((dict) => {
    const payload: DataDict = dict["users"] as DataDict;
    Object.entries(payload).forEach(([user_id, value]) => {
      if (
        user_id === "" ||
        user_id === undefined ||
        user_id === null ||
        user_id == "None"
      ) {
        return;
      }

      if (!userSpend[user_id]) {
        userSpend[user_id] = 0;
      }
      userSpend[user_id] += value as number;
    });
  });

  const spendUsers: UserData[] = Object.entries(userSpend).map(
    ([user_id, spend]) => ({
      user_id,
      spend,
    })
  );

  spendUsers.sort((a, b) => b.spend - a.spend);

  const topKeys = spendUsers.slice(0, 5);
  console.log(`topKeys: ${Object.values(topKeys[0])}`);
  return topKeys;
}

const UsagePage: React.FC<UsagePageProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const currentDate = new Date();
  const [keySpendData, setKeySpendData] = useState<any[]>([]);
  const [topKeys, setTopKeys] = useState<any[]>([]);
  const [topModels, setTopModels] = useState<any[]>([]);
  const [topUsers, setTopUsers] = useState<any[]>([]);
  const [teamSpendData, setTeamSpendData] = useState<any[]>([]);
  const [topTagsData, setTopTagsData] = useState<any[]>([]);
  const [uniqueTeamIds, setUniqueTeamIds] = useState<any[]>([]);
  const [totalSpendPerTeam, setTotalSpendPerTeam] = useState<any[]>([]);
  const [modelMetrics, setModelMetrics] = useState<any[]>([]);
  const [modelLatencyMetrics, setModelLatencyMetrics] = useState<any[]>([]);
  const [modelGroups, setModelGroups] = useState<any[]>([]);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);

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
            const top_keys = await adminTopKeysCall(accessToken);
            const filtered_keys = top_keys.map((k: any) => ({
              key: (k["key_name"] || k["key_alias"] || k["api_key"]).substring(
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
              return tspt;
            })

            setTotalSpendPerTeam(total_spend_per_team);

            //get top tags
            const top_tags = await tagsSpendLogsCall(accessToken);
            setTopTagsData(top_tags.top_10_tags);

            // get model groups 
            const _model_groups = await modelInfoCall(accessToken, userID, userRole);
            let model_groups = _model_groups.data;
            console.log("model groups in model dashboard", model_groups);

            let available_model_groups = [];
            // loop through each model in model_group, access litellm_params and only inlclude the model if model["litellm_params"]["model"] startswith "azure/"
            for (let i = 0; i < model_groups.length; i++) {
              let model = model_groups[i];
              console.log("model check", model);
              let model_group = model["litellm_params"]["model"];
              console.log("model group", model_group);
              if (model_group.startsWith("azure/")) {
                available_model_groups.push(model["model_name"]);
              }
            }
            setModelGroups(available_model_groups);


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
                setTopUsers(getTopUsers(response));
                setKeySpendData(response);
              }
            });
          }

          const modelMetricsResponse = await modelMetricsCall(
            accessToken,
            userID,
            userRole,
            null
          );
  
          console.log("Model metrics response:", modelMetricsResponse);
          // Sort by latency (avg_latency_seconds)
          const sortedByLatency = [...modelMetricsResponse].sort((a, b) => b.avg_latency_seconds - a.avg_latency_seconds);
          console.log("Sorted by latency:", sortedByLatency);

          setModelMetrics(modelMetricsResponse);
          setModelLatencyMetrics(sortedByLatency);

        } catch (error) {
          console.error("There was an error fetching the data", error);
          // Optionally, update your UI to reflect the error state here as well
        }
      };
      fetchData();
    }
  }, [accessToken, token, userRole, userID, startTime, endTime]);


  const updateModelMetrics = async (modelGroup: string | null) => {
    console.log("Updating model metrics for group:", modelGroup);
    if (!accessToken || !userID || !userRole) {
      return
    }
    setSelectedModelGroup(modelGroup);  // If you want to store the selected model group in state

  
    try {
      const modelMetricsResponse = await modelMetricsCall(accessToken, userID, userRole, modelGroup);
      console.log("Model metrics response:", modelMetricsResponse);
  
      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      const sortedByLatency = [...modelMetricsResponse].sort((a, b) => b.avg_latency_seconds - a.avg_latency_seconds);
      console.log("Sorted by latency:", sortedByLatency);
  
      setModelMetrics(modelMetricsResponse);
      setModelLatencyMetrics(sortedByLatency);
    } catch (error) {
      console.error("Failed to fetch model metrics", error);
    }
  }
  

  return (
    <div style={{ width: "100%" }} className="p-8">
      <ViewUserSpend
            userID={userID}
            userRole={userRole}
            accessToken={accessToken}
            userSpend={null}
            selectedTeam={null}
          />
      <TabGroup>
        <TabList className="mt-2">
          <Tab>All Up</Tab>
          <Tab>Team Based Usage</Tab>
           <Tab>Tag Based Usage</Tab>
           <Tab>Model Based Usage</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={2} className="gap-2 h-[75vh] w-full">
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
                  <Title>Top Users</Title>
                  <BarChart
                    className="mt-4 h-40"
                    data={topUsers}
                    index="user_id"
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
            </Grid>
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
                  colors={["blue", "green", "yellow", "red", "purple"]}
                  
                  stack={true}
                />
              </Card>
              </Col>
              <Col numColSpan={2}>
              </Col>
            </Grid>
            </TabPanel>
            <TabPanel>
            <Grid numItems={2} className="gap-2 h-[75vh] w-full mb-4">
            <Col numColSpan={2}>

              <Card>
              <Title>Spend Per Tag - Last 30 Days</Title>
              <Text>Get Started Tracking cost per tag <a href="https://docs.litellm.ai/docs/proxy/enterprise#tracking-spend-for-custom-tags" target="_blank">here</a></Text>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Tag</TableHeaderCell>
                    <TableHeaderCell>Spend</TableHeaderCell>
                    <TableHeaderCell>Requests</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topTagsData.map((tag) => (
                    <TableRow key={tag.name}>
                      <TableCell>{tag.name}</TableCell>
                      <TableCell>{tag.value}</TableCell>
                      <TableCell>{tag.log_count}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
                {/* <BarChart
                  className="h-72"
                  data={teamSpendData}
                  showLegend={true}
                  index="date"
                  categories={uniqueTeamIds}
                  yAxisWidth={80}
                  
                  stack={true}
                /> */}
              </Card>
              </Col>
              <Col numColSpan={2}>
              </Col>
            </Grid>
            </TabPanel>
            
            <TabPanel>
              <Title>Filter By Model Group</Title>
              <p style={{fontSize: '0.85rem', color: '#808080'}}>View how requests were load balanced within a model group</p>
              <p style={{fontSize: '0.85rem', color: '#808080', fontStyle: 'italic'}}>(Beta feature) only supported for Azure Model Groups</p>


            <Select
              className="mb-4 mt-2"
              defaultValue="all"
            >
              <SelectItem 
                  value={"all"}
                  onClick={() => updateModelMetrics(null)}
                >
                  All Model Groups
                </SelectItem>
              {modelGroups.map((group, idx) => (
                <SelectItem 
                  key={idx} 
                  value={group}
                  onClick={() => updateModelMetrics(group)}
                >
                  {group}
                </SelectItem>
              ))}
            </Select>
            <Card>
          <Title>Number Requests per Model</Title>
              <BarChart
                data={modelMetrics}
                className="h-[50vh]"
                index="model"
                categories={["num_requests"]}
                colors={["blue"]}
                yAxisWidth={400}
                layout="vertical"
                tickGap={5}
              />
        </Card>
        <Card className="mt-4">
          <Title>Latency Per Model</Title>
              <BarChart
                data={modelLatencyMetrics}
                className="h-[50vh]"
                index="model"
                categories={["avg_latency_seconds"]}
                colors={["red"]}
                yAxisWidth={400}
                layout="vertical"
                tickGap={5}
              />
        </Card>

            </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default UsagePage;
