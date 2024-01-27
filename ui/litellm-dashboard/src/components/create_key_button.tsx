"use client";

import React, { use } from "react";
import { Button, TextInput } from "@tremor/react";

import { Card, Metric, Text } from "@tremor/react";
import { createKeyCall } from "./networking";
// Define the props type
interface CreateKeyProps {
  userID: string;
  accessToken: string;
  proxyBaseUrl: string;
}

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  accessToken,
  proxyBaseUrl,
}) => {
  const handleClick = () => {
    console.log("Hello World");
  };

  return (
    <Button
      className="mx-auto"
      onClick={() =>
        createKeyCall(
          (proxyBaseUrl = proxyBaseUrl),
          (accessToken = accessToken),
          (userID = userID)
        )
      }
    >
      + Create New Key
    </Button>
  );
};

export default CreateKey;
