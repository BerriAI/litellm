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
import { keySpendLogsCall, PredictedSpendLogsCall } from "./networking";

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
  const [predictedSpendString, setPredictedSpendString] = useState("");
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
      setData(response);

      // predict spend based on response
      const predictedSpend = await PredictedSpendLogsCall(accessToken, response);
      console.log("Response2:", predictedSpend);

      // append predictedSpend to data
      const combinedData = [...response, ...predictedSpend.response];
      setData(combinedData);
      setPredictedSpendString(predictedSpend.predicted_spend)

      console.log("Combined Data:", combinedData);
      // setPredictedSpend(predictedSpend);
      
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
  };


  if (!token) {
    return null;
  }

  return (
    <div>
      <Button size = "xs" onClick={showModal} variant="secondary">
        View Spend Report
      </Button>
      <Modal
        visible={isModalVisible}
        width={1400}
        onOk={handleOk}
        onCancel={handleCancel}
        footer={null}
      >
        <Title style={{ textAlign: "left" }}>Key Name: {keyName}</Title>

        <Metric>Monthly Spend ${keySpend}</Metric>
        <Title>{predictedSpendString}</Title>

        <Card className="mt-6 mb-6">
          {data && (
            <BarChart
              className="mt-6"
              data={data}
              colors={["blue", "amber"]}
              index="date"
              categories={["spend", "predicted_spend"]}
              yAxisWidth={80}
            />
          )}
        </Card>
  
      </Modal>
    </div>
  );
};

export default ViewKeySpendReport;
