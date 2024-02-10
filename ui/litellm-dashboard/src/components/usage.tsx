import { BarChart, Card, Title } from "@tremor/react";

import React, { useState, useEffect } from "react";
import { Grid, Col, Text } from "@tremor/react";
import { userSpendLogsCall } from "./networking";
import { AreaChart, Flex, Switch, Subtitle } from "@tremor/react";

interface UsagePageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

type DataType = {
  api_key: string;
  startTime: string;
  _sum: {
    spend: number;
  };
};

const UsagePage: React.FC<UsagePageProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const currentDate = new Date();
  const [keySpendData, setKeySpendData] = useState<any[]>([]);

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
      const cachedKeySpendData = localStorage.getItem("keySpendData");
      if (cachedKeySpendData) {
        setKeySpendData(JSON.parse(cachedKeySpendData));
      } else {
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
            setKeySpendData(response);
            localStorage.setItem("keySpendData", JSON.stringify(response));
          } catch (error) {
            console.error("There was an error fetching the data", error);
            // Optionally, update your UI to reflect the error state here as well
          }
        };
        fetchData();
      }
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
            />
          </Card>
        </Col>
      </Grid>
    </div>
  );
};

export default UsagePage;
