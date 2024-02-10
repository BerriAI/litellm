import { BarChart, Card, Title } from "@tremor/react";

import React, { useState, useEffect } from "react";
import { Grid, Col, Text, LineChart } from "@tremor/react";
import { userSpendLogsCall } from "./networking";
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

  // Convert the object into an array of key-value pairs
  const entries: [string, number][] = Object.entries(value)
    .filter(([key]) => key !== "spend" && key !== "startTime")
    .map(([key, value]) => [key, value as number]); // Type assertion to specify the value as number

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
            <p className="text-tremor-content">
              Token: {key.substring(0, 4)}{" "}
              <span className="font-medium text-tremor-content-emphasis">
                Spend: {value}
              </span>
            </p>
          </div>
        </div>
      ))}
    </div>
  );

  //   return (
  //     <div className="w-56 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
  //       {payload.map((category: any, idx: number) => {
  //         <div key={idx} className="flex flex-1 space-x-2.5">
  //           <div
  //             className={`flex w-1 flex-col bg-${category.color}-500 rounded`}
  //           />
  //           <div className="space-y-1">
  //             <p className="text-tremor-content">{category.dataKey}</p>
  //             <p className="font-medium text-tremor-content-emphasis">
  //               {category.value} bpm
  //             </p>
  //           </div>
  //         </div>;
  //       })}
  //     </div>
  //   );
};

const UsagePage: React.FC<UsagePageProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const currentDate = new Date();
  const [keySpendData, setKeySpendData] = useState<any[]>([]);
  const [keyCategories, setKeyCategories] = useState<string[]>([]);

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
          const response = await userSpendLogsCall(
            accessToken,
            token,
            userRole,
            userID,
            startTime,
            endTime
          );

          const uniqueKeys: Set<string> = new Set();

          response.forEach((item: any) => {
            Object.keys(item).forEach((key) => {
              if (key !== "spend" && key !== "startTime") {
                uniqueKeys.add(key);
              }
            });
          });
          let uniqueKeysList = Array.from(uniqueKeys);
          setKeyCategories(uniqueKeysList);
          setKeySpendData(response);
          // localStorage.setItem("keySpendData", JSON.stringify(response));
          // localStorage.setItem(
          //   "keyCategories",
          //   JSON.stringify(uniqueKeysList)
          // );
        } catch (error) {
          console.error("There was an error fetching the data", error);
          // Optionally, update your UI to reflect the error state here as well
        }
      };
      fetchData();
    }
  }, [accessToken, token, userRole, userID]);
  return (
    <div style={{ width: "100%" }}>
      <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
        <Col numColSpan={1}>
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
      </Grid>
    </div>
  );
};

export default UsagePage;
