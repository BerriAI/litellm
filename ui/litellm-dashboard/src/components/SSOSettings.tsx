import React, { useState, useEffect } from "react";
import { Card, Title, Text, Divider } from "@tremor/react";
import { Typography, Spin, message } from "antd";
import { getSSOSettingsCall } from "./networking";

interface SSOSettingsProps {
  accessToken: string | null;
}

const SSOSettings: React.FC<SSOSettingsProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [settings, setSettings] = useState<any>(null);
  const { Paragraph } = Typography;

  useEffect(() => {
    const fetchSSOSettings = async () => {
      if (!accessToken) {
        setLoading(false);
        return;
      }

      try {
        const data = await getSSOSettingsCall(accessToken);
        setSettings(data);
      } catch (error) {
        console.error("Error fetching SSO settings:", error);
        message.error("Failed to fetch SSO settings");
      } finally {
        setLoading(false);
      }
    };

    fetchSSOSettings();
  }, [accessToken]);

  const renderValue = (value: any): JSX.Element => {
    if (value === null) return <span className="text-gray-400">Not set</span>;
    
    if (typeof value === "object") {
      return (
        <pre className="bg-gray-50 p-2 rounded overflow-auto max-h-60">
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    }
    
    return <span>{String(value)}</span>;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    );
  }

  if (!settings) {
    return (
      <Card>
        <Title>Personal Key Creation</Title>
        <Text>No settings available or you don't have permission to view them.</Text>
      </Card>
    );
  }

  // Dynamically render settings based on the schema
  const renderSettings = () => {
    const { values, schema } = settings;
    
    if (!schema || !schema.properties) {
      return <Text>No schema information available</Text>;
    }

    return Object.entries(schema.properties).map(([key, property]: [string, any]) => {
      const value = values[key];
      
      return (
        <div key={key} className="mb-6">
          <Text className="font-medium text-lg">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</Text>
          <Paragraph className="text-sm text-gray-500 mt-1">
            {property.description || "No description available"}
          </Paragraph>
          <div className="mt-2 p-3 bg-gray-50 rounded-md">
            {renderValue(value)}
          </div>
        </div>
      );
    });
  };

  return (
    <Card>
      <Title>SSO Settings</Title>
      {settings.schema?.description && (
        <Paragraph>{settings.schema.description}</Paragraph>
      )}
      <Divider />
      
      <div className="mt-4">
        {renderSettings()}
      </div>
    </Card>
  );
};

export default SSOSettings; 