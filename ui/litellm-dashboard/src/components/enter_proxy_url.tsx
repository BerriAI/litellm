"use client";

import React, { useState } from "react";
import { Button, TextInput } from "@tremor/react";
import { Card, Text } from "@tremor/react";

const EnterProxyUrl: React.FC = () => {
  const [proxyUrl, setProxyUrl] = useState<string>("");
  const [isUrlSaved, setIsUrlSaved] = useState<boolean>(false);

  const handleUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setProxyUrl(event.target.value);
    // Reset the saved status when the URL changes
    setIsUrlSaved(false);
  };

  const handleSaveClick = () => {
    // You can perform any additional validation or actions here
    // For now, let's just display the message
    setIsUrlSaved(true);
  };

  return (
    <div>
      <Card decoration="top" decorationColor="blue" style={{ width: '1000px' }}>
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
        {/* Display message if the URL is saved */}
        {isUrlSaved && (
        <div>

            <p>
            Save this URL: {window.location.href + "?proxyBaseUrl=" + proxyUrl}
            </p>
            <p>
            Go here 👉 
            <a href={window.location.href + "?proxyBaseUrl=" + proxyUrl} target="_blank" rel="noopener noreferrer">
            {window.location.href + "?proxyBaseUrl=" + proxyUrl}
            </a>
            </p>

        </div>
        )}

      </Card>
    </div>
  );
};

export default EnterProxyUrl;
