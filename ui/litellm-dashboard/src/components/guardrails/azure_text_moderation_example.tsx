import React, { useState } from "react";
import { Button, Space, Card } from "antd";
import AzureTextModerationConfiguration from "./azure_text_moderation_configuration";
import NotificationsManager from "../molecules/notifications_manager";

/**
 * Example component showing how to use the AzureTextModerationConfiguration
 * This demonstrates the state management and event handling required
 */
const AzureTextModerationExample: React.FC = () => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>(["Hate", "Violence"]);
  const [globalSeverityThreshold, setGlobalSeverityThreshold] = useState<number>(2);
  const [categorySpecificThresholds, setCategorySpecificThresholds] = useState<{ [key: string]: number }>({
    Hate: 4,
  });

  const handleCategorySelect = (category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category],
    );
  };

  const handleGlobalSeverityChange = (threshold: number) => {
    setGlobalSeverityThreshold(threshold);
  };

  const handleCategorySeverityChange = (category: string, threshold: number) => {
    setCategorySpecificThresholds((prev) => ({
      ...prev,
      [category]: threshold,
    }));
  };

  const handleSave = () => {
    // Example of how to construct the configuration object
    const config = {
      categories: selectedCategories,
      severity_threshold: globalSeverityThreshold,
      severity_threshold_by_category: categorySpecificThresholds,
    };

    console.log("Azure Text Moderation Configuration:", config);
    NotificationsManager.success("Configuration saved successfully!");
  };

  const handleReset = () => {
    setSelectedCategories(["Hate", "Violence"]);
    setGlobalSeverityThreshold(2);
    setCategorySpecificThresholds({ Hate: 4 });
    NotificationsManager.info("Configuration reset to defaults");
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 20 }}>
      <Card title="Azure Text Moderation Configuration Example" className="mb-6">
        <AzureTextModerationConfiguration
          selectedCategories={selectedCategories}
          globalSeverityThreshold={globalSeverityThreshold}
          categorySpecificThresholds={categorySpecificThresholds}
          onCategorySelect={handleCategorySelect}
          onGlobalSeverityChange={handleGlobalSeverityChange}
          onCategorySeverityChange={handleCategorySeverityChange}
        />

        <div className="mt-6 pt-4 border-t">
          <Space>
            <Button type="primary" onClick={handleSave}>
              Save Configuration
            </Button>
            <Button onClick={handleReset}>Reset to Defaults</Button>
          </Space>
        </div>
      </Card>

      <Card title="Current Configuration" size="small">
        <pre style={{ fontSize: 12, backgroundColor: "#f5f5f5", padding: 12, borderRadius: 4 }}>
          {JSON.stringify(
            {
              categories: selectedCategories,
              severity_threshold: globalSeverityThreshold,
              severity_threshold_by_category: categorySpecificThresholds,
            },
            null,
            2,
          )}
        </pre>
      </Card>
    </div>
  );
};

export default AzureTextModerationExample;
