import React, { useState } from "react";
import { Card, Title, Text, TextInput } from "@tremor/react";
import { List, Empty, Spin, Checkbox } from "antd";
import { ExperimentOutlined, SearchOutlined } from "@ant-design/icons";
import GuardrailTestPanel from "./GuardrailTestPanel";
import { applyGuardrail } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface GuardrailItem {
  guardrail_id?: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
  };
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}

interface GuardrailTestPlaygroundProps {
  guardrailsList: GuardrailItem[];
  isLoading: boolean;
  accessToken: string | null;
  onClose: () => void;
}

interface TestResult {
  guardrailName: string;
  response_text: string;
  latency: number;
}

interface TestError {
  guardrailName: string;
  error: Error;
  latency: number;
}

const GuardrailTestPlayground: React.FC<GuardrailTestPlaygroundProps> = ({
  guardrailsList,
  isLoading,
  accessToken,
  onClose,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testErrors, setTestErrors] = useState<TestError[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  const filteredGuardrails = guardrailsList.filter((guardrail) =>
    guardrail.guardrail_name?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleGuardrailSelection = (guardrailName: string) => {
    const newSelection = new Set(selectedGuardrails);
    if (newSelection.has(guardrailName)) {
      newSelection.delete(guardrailName);
    } else {
      newSelection.add(guardrailName);
    }
    setSelectedGuardrails(newSelection);
  };

  const handleTestGuardrails = async (text: string) => {
    if (selectedGuardrails.size === 0 || !accessToken) {
      return;
    }

    setIsTesting(true);
    setTestResults([]);
    setTestErrors([]);

    const results: TestResult[] = [];
    const errors: TestError[] = [];

    await Promise.all(
      Array.from(selectedGuardrails).map(async (guardrailName) => {
        const startTime = Date.now();
        try {
          const result = await applyGuardrail(accessToken, guardrailName, text, null, null);
          const latency = Date.now() - startTime;
          results.push({
            guardrailName,
            response_text: result.response_text,
            latency,
          });
        } catch (error) {
          const latency = Date.now() - startTime;
          console.error(`Error testing guardrail ${guardrailName}:`, error);
          errors.push({
            guardrailName,
            error: error as Error,
            latency,
          });
        }
      })
    );

    setTestResults(results);
    setTestErrors(errors);
    setIsTesting(false);

    if (results.length > 0) {
      NotificationsManager.success(
        `${results.length} guardrail${results.length > 1 ? "s" : ""} applied successfully`
      );
    }
    if (errors.length > 0) {
      NotificationsManager.fromBackend(
        `${errors.length} guardrail${errors.length > 1 ? "s" : ""} failed`
      );
    }
  };

  return (
    <div className="w-full h-[calc(100vh-200px)]">
      <Card className="h-full">
        <div className="flex h-full">
          {/* Left Sidebar - Guardrails List */}
          <div className="w-1/4 border-r border-gray-200 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-gray-200">
              <div className="mb-3">
                <Title className="text-lg font-semibold mb-3">Guardrails</Title>
                <TextInput
                  icon={SearchOutlined}
                  placeholder="Search guardrails..."
                  value={searchQuery}
                  onValueChange={setSearchQuery}
                />
              </div>
            </div>

            <div className="flex-1 overflow-auto">
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <Spin />
                </div>
              ) : filteredGuardrails.length === 0 ? (
                <div className="p-4">
                  <Empty
                    description={
                      searchQuery ? "No guardrails match your search" : "No guardrails available"
                    }
                  />
                </div>
              ) : (
                <List
                  dataSource={filteredGuardrails}
                  renderItem={(guardrail) => (
                    <List.Item
                      onClick={() => {
                        if (guardrail.guardrail_name) {
                          toggleGuardrailSelection(guardrail.guardrail_name);
                        }
                      }}
                      className={`cursor-pointer hover:bg-gray-50 transition-colors px-4 ${
                        selectedGuardrails.has(guardrail.guardrail_name || "")
                          ? "bg-blue-50 border-l-4 border-l-blue-500"
                          : "border-l-4 border-l-transparent"
                      }`}
                    >
                      <List.Item.Meta
                        avatar={
                          <Checkbox
                            checked={selectedGuardrails.has(guardrail.guardrail_name || "")}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (guardrail.guardrail_name) {
                                toggleGuardrailSelection(guardrail.guardrail_name);
                              }
                            }}
                          />
                        }
                        title={
                          <div className="flex items-center space-x-2">
                            <ExperimentOutlined className="text-gray-400" />
                            <span className="font-medium text-gray-900">
                              {guardrail.guardrail_name}
                            </span>
                          </div>
                        }
                        description={
                          <div className="text-xs space-y-1 mt-1">
                            <div>
                              <span className="font-medium">Type: </span>
                              <span className="text-gray-600">
                                {guardrail.litellm_params.guardrail}
                              </span>
                            </div>
                            <div>
                              <span className="font-medium">Mode: </span>
                              <span className="text-gray-600">
                                {guardrail.litellm_params.mode}
                              </span>
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              )}
            </div>

            <div className="p-3 border-t border-gray-200 bg-gray-50">
              <Text className="text-xs text-gray-600">
                {selectedGuardrails.size} of {filteredGuardrails.length} selected
              </Text>
            </div>
          </div>

          {/* Right Panel - Test Area */}
          <div className="w-3/4 flex flex-col bg-white">
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
              <Title className="text-xl font-semibold mb-0">Guardrail Testing Playground</Title>
            </div>

            <div className="flex-1 overflow-auto p-4">
              {selectedGuardrails.size === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-gray-400">
                  <ExperimentOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
                  <Text className="text-lg font-medium text-gray-600 mb-2">
                    Select Guardrails to Test
                  </Text>
                  <Text className="text-center text-gray-500 max-w-md">
                    Choose one or more guardrails from the left sidebar to start testing and
                    comparing results.
                  </Text>
                </div>
              ) : (
                <div className="h-full">
                  <GuardrailTestPanel
                    guardrailNames={Array.from(selectedGuardrails)}
                    onSubmit={handleTestGuardrails}
                    results={testResults.length > 0 ? testResults : null}
                    errors={testErrors.length > 0 ? testErrors : null}
                    isLoading={isTesting}
                    onClose={() => setSelectedGuardrails(new Set())}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default GuardrailTestPlayground;

