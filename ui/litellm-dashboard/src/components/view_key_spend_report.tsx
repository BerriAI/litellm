"use client";

import React, { useState, useEffect } from "react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  message,
} from "antd";
import {
  Button,
  Text,
  Card,
  Table,
  BarChart,
  Title,
  Subtitle,
  BarList,
  Metric,
} from "@tremor/react";
import { keySpendLogsCall } from "./networking";

interface ViewKeySpendReportProps {
  token: string;
  accessToken: string;
  keySpend: number;
  keyBudget: number;
  keyName: string;
}

type ResponseValueType = {
  startTime: string; // Assuming startTime is a string, adjust it if it's of a different type
  spend: number; // Assuming spend is a number, adjust it if it's of a different type
  user: string; // Assuming user is a string, adjust it if it's of a different type
};

const ViewKeySpendReport: React.FC<ViewKeySpendReportProps> = ({
  token,
  accessToken,
  keySpend,
  keyBudget,
  keyName,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [data, setData] = useState<{ day: string; spend: number }[] | null>(
    null
  );
  const [userData, setUserData] = useState<
    { name: string; value: number }[] | null
  >(null);

  const showModal = () => {
    console.log("Show Modal triggered");
    setIsModalVisible(true);
    fetchData();
  };

  const handleOk = () => {
    setIsModalVisible(false);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
  };

  // call keySpendLogsCall and set the data
  const fetchData = async () => {
    try {
      if (accessToken == null || token == null) {
        return;
      }
      console.log(`accessToken: ${accessToken}; token: ${token}`);
      const response = await keySpendLogsCall(
        (accessToken = accessToken),
        (token = token)
      );
      console.log("Response:", response);
      // loop through response
      // get spend, startTime for each element, place in new array

      const pricePerDay: Record<string, number> = (
        Object.values(response) as ResponseValueType[]
      ).reduce((acc: Record<string, number>, value) => {
        const startTime = new Date(value.startTime);
        const day = new Intl.DateTimeFormat("en-US", {
          day: "2-digit",
          month: "short",
        }).format(startTime);

        acc[day] = (acc[day] || 0) + value.spend;

        return acc;
      }, {});

      // sort pricePerDay by day
      // Convert object to array of key-value pairs
      const pricePerDayArray = Object.entries(pricePerDay);

      // Sort the array based on the date (key)
      pricePerDayArray.sort(([aKey], [bKey]) => {
        const dateA = new Date(aKey);
        const dateB = new Date(bKey);
        return dateA.getTime() - dateB.getTime();
      });

      // Convert the sorted array back to an object
      const sortedPricePerDay = Object.fromEntries(pricePerDayArray);

      console.log(sortedPricePerDay);

      const pricePerUser: Record<string, number> = (
        Object.values(response) as ResponseValueType[]
      ).reduce((acc: Record<string, number>, value) => {
        const user = value.user;
        acc[user] = (acc[user] || 0) + value.spend;

        return acc;
      }, {});

      console.log(pricePerDay);
      console.log(pricePerUser);

      const arrayBarChart = [];
      // [
      // {
      //     "day": "02 Feb",
      //     "spend": pricePerDay["02 Feb"],
      // }
      // ]
      for (const [key, value] of Object.entries(sortedPricePerDay)) {
        arrayBarChart.push({ day: key, spend: value });
      }

      // get 5 most expensive users
      const sortedUsers = Object.entries(pricePerUser).sort(
        (a, b) => b[1] - a[1]
      );
      const top5Users = sortedUsers.slice(0, 5);
      const userChart = top5Users.map(([key, value]) => ({
        name: key,
        value: value,
      }));

      setData(arrayBarChart);
      setUserData(userChart);
      console.log("arrayBarChart:", arrayBarChart);
    } catch (error) {
      console.error("There was an error fetching the data", error);
      // Optionally, update your UI to reflect the error state here as well
    }
  };

  // useEffect(() => {
  //   // Fetch data only when the token changes
  //   fetchData();
  // }, [token]); // Dependency array containing the 'token' variable

  if (!token) {
    return null;
  }

  return (
    <div>
      <Button className="mx-auto" onClick={showModal}>
        View Spend Report
      </Button>
      <Modal
        visible={isModalVisible}
        width={1000}
        onOk={handleOk}
        onCancel={handleCancel}
        footer={null}
      >
        <Title style={{ textAlign: "left" }}>Key Name: {keyName}</Title>

        <Metric>Monthly Spend ${keySpend}</Metric>

        <Card className="mt-6 mb-6">
          {data && (
            <BarChart
              className="mt-6"
              data={data}
              colors={["green"]}
              index="day"
              categories={["spend"]}
              yAxisWidth={48}
            />
          )}
        </Card>
        <Title className="mt-6">Top 5 Users Spend (USD)</Title>
        <Card className="mb-6">
          {userData && (
            <BarList className="mt-6" data={userData} color="teal" />
          )}
        </Card>
      </Modal>
    </div>
  );
};

export default ViewKeySpendReport;
