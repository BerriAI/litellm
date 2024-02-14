import { BarChart, Card, Title } from "@tremor/react";

import React, { useState, useEffect } from "react";
import { Grid, Col, Text, LineChart } from "@tremor/react";
import { userSpendLogsCall, keyInfoCall } from "./networking";
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
  const [topUsers, setTopUsers] = useState<any[]>([]);

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
          await userSpendLogsCall(
            accessToken,
            token,
            userRole,
            userID,
            startTime,
            endTime
          ).then(async (response) => {
            const topKeysResponse = await keyInfoCall(
              accessToken,
              getTopKeys(response)
            );
            const filtered_keys = topKeysResponse["info"].map((k: any) => ({
              key: (k["key_name"] || k["key_alias"] || k["token"]).substring(
                0,
                7
              ),
              spend: k["spend"],
            }));
            setTopKeys(filtered_keys);
            setTopUsers(getTopUsers(response));
            setKeySpendData(response);
          });
        } catch (error) {
          console.error("There was an error fetching the data", error);
          // Optionally, update your UI to reflect the error state here as well
        }
      };
      fetchData();
    }
  }, [accessToken, token, userRole, userID, startTime, endTime]);

  return (
    <div style={{ width: "100%" }}>
      <Grid numItems={2} className="gap-2 p-10 h-[75vh] w-full">
        <Col numColSpan={2}>
          <Card>
            <Title>Monthly Spend</Title>
            <BarChart
              data={keySpendData}
              index="startTime"
              categories={["spend"]}
              colors={["blue"]}
              valueFormatter={valueFormatter}
              yAxisWidth={100}
              tickGap={5}
              customTooltip={customTooltip}
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
      </Grid>
    </div>
  );
};

export default UsagePage;
