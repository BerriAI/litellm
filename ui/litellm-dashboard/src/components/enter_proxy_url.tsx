"use client";

import React, { useState, ChangeEvent } from "react";
import { Button, Col, Grid, TextInput } from "@tremor/react";
import { Card, Text } from "@tremor/react";

const EnterProxyUrl: React.FC = () => {
  const [proxyUrl, setProxyUrl] = useState<string>("");
  const [isUrlSaved, setIsUrlSaved] = useState<boolean>(false);

  const handleUrlChange = (event: ChangeEvent<HTMLInputElement>) => {
    setProxyUrl(event.target.value);
    // Reset the saved status when the URL changes
    setIsUrlSaved(false);
  };

  const handleSaveClick = () => {
    // You can perform any additional validation or actions here
    // For now, let's just display the message
    setIsUrlSaved(true);
  };

  // Construct the URL for clicking
  const clickableUrl = `${window.location.href}?proxyBaseUrl=${proxyUrl}`;

  return (
    <div>
      <Card decoration="top" decorationColor="blue" style={{ width: "1000px" }}>
        <Text>Admin Configuration</Text>
        <label htmlFor="proxyUrl">Enter Proxy URL:</label>
        <TextInput
          type="text"
          id="proxyUrl"
          value={proxyUrl}
          onChange={handleUrlChange}
          placeholder="https://your-proxy-endpoint.com"
        />
        <Button onClick={handleSaveClick} className="gap-2">
          Save
        </Button>
        {/* Display message if the URL is saved */}
        {isUrlSaved && (
          <div>
            <Grid numItems={1} className="gap-2">
              <Col>
                <p>Proxy Admin UI (Save this URL): {clickableUrl}</p>
              </Col>
              <Col>
                <p>
                  Get Started with Proxy Admin UI 👉
                  <a
                    href={clickableUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "blue", textDecoration: "underline" }}
                  >
                    {clickableUrl}
                  </a>
                </p>
              </Col>
            </Grid>
          </div>
        )}
      </Card>
    </div>
  );
};

export default EnterProxyUrl;
