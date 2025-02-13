import React, { useState } from 'react';
import { message, Typography } from 'antd';
import { healthCheckCall } from './networking';
import { Text, Card, Button } from '@tremor/react';

interface HealthCheckProps {
  accessToken: string | null;
}

const HealthCheck: React.FC<HealthCheckProps> = ({ accessToken }) => {
  const [healthCheckResponse, setHealthCheckResponse] = useState("");

  const runHealthCheck = async () => {
    try {
      message.info("Running health check...");
      setHealthCheckResponse("");
      if (!accessToken) {
        message.error("Access token is missing");
        setHealthCheckResponse("Access token is missing");
        return;
      }
      const response = await healthCheckCall(accessToken);
      setHealthCheckResponse(response);
    } catch (error) {
      console.error("Error running health check:", error);
      setHealthCheckResponse("Error running health check");
    }
  };

  return (
    <Card>
        <Text>
        `/health` will run a very small request through your models
        configured on litellm
        </Text>

        <Button onClick={runHealthCheck}>Run `/health`</Button>
        {healthCheckResponse && (
        <pre>{JSON.stringify(healthCheckResponse, null, 2)}</pre>
        )}
    </Card>
  );
};

export default HealthCheck;