"use client";

import React, { use, useState } from "react";
import { Button, TextInput } from "@tremor/react";

import { Card, Metric, Text } from "@tremor/react";
import { createKeyCall } from "./networking";
// Define the props type
interface EnterProxyUrlProps {
    onUrlChange: (url: string) => void;
  }
  
  const EnterProxyUrl: React.FC<EnterProxyUrlProps> = ({ onUrlChange }) => {
    const [proxyUrl, setProxyUrl] = useState<string>('');
  
    const handleUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      setProxyUrl(event.target.value);
    };
  
    const handleSaveClick = () => {
      // You can perform any additional validation or actions here
      // For now, let's just pass the entered URL to the parent component
      onUrlChange(proxyUrl);
    };
  
    return (
      <div>
        <Card decoration="top" decorationColor="blue" style={{ width: '100%' }}>
            <Text>Admin Configuration</Text>
            <label htmlFor="proxyUrl">Enter Proxy URL:</label>
            <TextInput
            type="text"
            id="proxyUrl"
            value={proxyUrl}
            onChange={handleUrlChange}
            placeholder="https://your-proxy-endpoint.com"
            />
            <Button onClick={handleSaveClick}>Save</Button>
        </Card>
      </div>
    );
  };
  
export default EnterProxyUrl;