"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { message } from "antd";
import { Card, Metric, Text } from "@tremor/react";
import { keyCreateCall } from "./networking";
// Define the props type
interface CreateKeyProps {
  userID: string;
  accessToken: string;
  proxyBaseUrl: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

import { Modal, Button as Button2 } from "antd";

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  accessToken,
  proxyBaseUrl,
  data,
  setData,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);
  const handleOk = () => {
    // Handle the OK action
    console.log("OK Clicked");
    setIsModalVisible(false);
  };

  const handleCancel = () => {
    // Handle the cancel action or closing the modal
    console.log("Modal closed");
    setIsModalVisible(false);
    setApiKey(null);
  };

  const handleCreate = async () => {
    if (data == null) {
      return;
    }
    try {
      message.info("Making API Call");
      setIsModalVisible(true);
      const response = await keyCreateCall(proxyBaseUrl, accessToken, userID);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      setData([...data, response]);
      setApiKey(response["key"]);
      message.success("API Key Created");
    } catch (error) {
      console.error("Error deleting the key:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }
  };

  return (
    <div>
      <Button className="mx-auto" onClick={handleCreate}>
        + Create New Key
      </Button>
      <Modal
        title="Save your key"
        open={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Grid numItems={1} className="gap-2 w-full">
          <Col numColSpan={1}>
            <p>
              Please save this secret key somewhere safe and accessible. For
              security reasons, <b>you will not be able to view it again</b>{" "}
              through your LiteLLM account. If you lose this secret key, you will
              need to generate a new one.
            </p>
          </Col>
          <Col numColSpan={1}>
            {apiKey != null ? (
              <Text>API Key: {apiKey}</Text>
            ) : (
              <Text>Key being created, this might take 30s</Text>
            )}
          </Col>
        </Grid>
      </Modal>
    </div>
  );
};

export default CreateKey;
