import React, { useState } from "react";
import { Typography, Badge, Card, Checkbox, Select, Slider, Tooltip, Divider, Row, Col } from "antd";
import {
  AzureTextModerationConfigurationProps,
  AZURE_TEXT_MODERATION_CATEGORIES,
  SEVERITY_LEVELS,
} from "./azure_text_moderation_types";
import { InfoCircleOutlined, SafetyOutlined } from "@ant-design/icons";

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

/**
 * A reusable component for configuring Azure Text Moderation guardrail settings
 * Allows configuration of categories, global severity threshold, and per-category thresholds
 */
const AzureTextModerationConfiguration: React.FC<AzureTextModerationConfigurationProps> = ({
  selectedCategories,
  globalSeverityThreshold,
  categorySpecificThresholds,
  onCategorySelect,
  onGlobalSeverityChange,
  onCategorySeverityChange,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const getSeverityLabel = (value: number) => {
    const level = SEVERITY_LEVELS.find((l) => l.value === value);
    return level ? level.label : `Level ${value}`;
  };

  const getSeverityColor = (value: number) => {
    if (value === 0) return "#52c41a"; // green
    if (value === 2) return "#faad14"; // orange
    if (value === 4) return "#fa8c16"; // dark orange
    return "#f5222d"; // red
  };

  const handleSelectAll = () => {
    AZURE_TEXT_MODERATION_CATEGORIES.forEach((category) => {
      if (!selectedCategories.includes(category.name)) {
        onCategorySelect(category.name);
      }
    });
  };

  const handleUnselectAll = () => {
    selectedCategories.forEach((category) => {
      onCategorySelect(category);
    });
  };

  return (
    <div className="azure-text-moderation-configuration">
      <div className="flex justify-between items-center mb-5">
        <div className="flex items-center">
          <SafetyOutlined className="text-blue-600 mr-2 text-lg" />
          <Title level={4} className="mb-0 font-semibold text-gray-800">
            Azure Text Moderation
          </Title>
        </div>
        <Badge
          count={selectedCategories.length}
          showZero
          style={{ backgroundColor: selectedCategories.length > 0 ? "#1890ff" : "#d9d9d9" }}
          overflowCount={999}
        >
          <Text className="text-gray-500">{selectedCategories.length} categories selected</Text>
        </Badge>
      </div>

      <Card className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={5} className="mb-0">
            Content Categories
          </Title>
          <div className="space-x-2">
            <a onClick={handleSelectAll} className="text-blue-600 hover:text-blue-800 cursor-pointer">
              Select All
            </a>
            <span className="text-gray-300">|</span>
            <a
              onClick={handleUnselectAll}
              className="text-red-600 hover:text-red-800 cursor-pointer"
              style={{ display: selectedCategories.length > 0 ? "inline" : "none" }}
            >
              Unselect All
            </a>
          </div>
        </div>

        <Paragraph className="text-gray-600 mb-4">Select which content categories to monitor and filter</Paragraph>

        <Row gutter={[16, 16]}>
          {AZURE_TEXT_MODERATION_CATEGORIES.map((category) => (
            <Col xs={24} sm={12} key={category.name}>
              <Card
                size="small"
                className={`cursor-pointer transition-all duration-200 ${
                  selectedCategories.includes(category.name)
                    ? "border-blue-500 bg-blue-50 shadow-sm"
                    : "border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => onCategorySelect(category.name)}
              >
                <div className="flex items-start">
                  <Checkbox
                    checked={selectedCategories.includes(category.name)}
                    onChange={() => onCategorySelect(category.name)}
                    className="mr-3 mt-1"
                  />
                  <div className="flex-1">
                    <Text strong className="text-sm">
                      {category.name}
                    </Text>
                    <br />
                    <Text className="text-xs text-gray-600">{category.description}</Text>
                  </div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      <Card className="mb-6">
        <div className="flex items-center mb-4">
          <Title level={5} className="mb-0 mr-2">
            Global Severity Threshold
          </Title>
          <Tooltip title="Content with severity levels at or above this threshold will be flagged">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>

        <Paragraph className="text-gray-600 mb-4">
          Set the minimum severity level that will trigger the guardrail for all selected categories
        </Paragraph>

        <div className="mb-4">
          <Row>
            <Col span={16}>
              <Slider
                min={0}
                max={6}
                step={2}
                value={globalSeverityThreshold}
                onChange={onGlobalSeverityChange}
                marks={{
                  0: { label: "Safe", style: { color: "#52c41a" } },
                  2: { label: "Low", style: { color: "#faad14" } },
                  4: { label: "Medium", style: { color: "#fa8c16" } },
                  6: { label: "High", style: { color: "#f5222d" } },
                }}
                tooltip={{
                  formatter: (value) => getSeverityLabel(value || 0),
                }}
              />
            </Col>
            <Col span={8} className="pl-4">
              <div className="text-center">
                <div
                  className="inline-block px-3 py-1 rounded-full text-white font-medium"
                  style={{ backgroundColor: getSeverityColor(globalSeverityThreshold) }}
                >
                  {getSeverityLabel(globalSeverityThreshold)}
                </div>
              </div>
            </Col>
          </Row>
        </div>
      </Card>

      <Card>
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <Title level={5} className="mb-0 mr-2">
              Per-Category Thresholds
            </Title>
            <Tooltip title="Override the global threshold for specific categories">
              <InfoCircleOutlined className="text-gray-400" />
            </Tooltip>
          </div>
          <a
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-blue-600 hover:text-blue-800 cursor-pointer"
          >
            {showAdvanced ? "Hide Advanced" : "Show Advanced"}
          </a>
        </div>

        {showAdvanced && (
          <>
            <Paragraph className="text-gray-600 mb-4">
              Set custom severity thresholds for individual categories. Leave empty to use the global threshold.
            </Paragraph>

            {selectedCategories.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                Please select at least one category to configure per-category thresholds
              </div>
            ) : (
              <div className="space-y-4">
                {selectedCategories.map((category) => (
                  <div key={category}>
                    <div className="flex justify-between items-center mb-2">
                      <Text strong>{category}</Text>
                      <Select
                        placeholder="Use global threshold"
                        style={{ width: 200 }}
                        value={categorySpecificThresholds[category]}
                        onChange={(value) => onCategorySeverityChange(category, value)}
                        allowClear
                      >
                        {SEVERITY_LEVELS.map((level) => (
                          <Option key={level.value} value={level.value}>
                            <div className="flex items-center justify-between">
                              <span>{level.label}</span>
                              <div
                                className="w-3 h-3 rounded-full ml-2"
                                style={{ backgroundColor: getSeverityColor(level.value) }}
                              />
                            </div>
                          </Option>
                        ))}
                      </Select>
                    </div>
                    <Text className="text-xs text-gray-500">
                      {AZURE_TEXT_MODERATION_CATEGORIES.find((c) => c.name === category)?.description}
                    </Text>
                    {category !== selectedCategories[selectedCategories.length - 1] && <Divider className="my-4" />}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  );
};

export default AzureTextModerationConfiguration;
