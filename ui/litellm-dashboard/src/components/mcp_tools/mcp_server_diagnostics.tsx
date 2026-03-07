import React, { useState, useEffect } from "react";
import { Card, Title, Text } from "@tremor/react";
import { Alert, Button, Spin, Tag, List } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { fetchMCPServerDiagnostics } from "../networking";

interface MCPServerDiagnosticsProps {
  serverId: string;
  accessToken: string | null;
}

interface DiagnosticCheck {
  name: string;
  status: "pass" | "fail";
  message: string;
  details?: Record<string, any>;
}

interface DiagnosticsResult {
  server_id: string;
  server_name: string;
  overall_status: "healthy" | "unhealthy";
  checks: DiagnosticCheck[];
  suggestions: string[];
}

const CHECK_LABELS: Record<string, string> = {
  configuration: "Configuration",
  auth_type: "Authentication",
  connectivity: "Connectivity",
  network: "Network Access",
};

const MCPServerDiagnostics: React.FC<MCPServerDiagnosticsProps> = ({
  serverId,
  accessToken,
}) => {
  const [result, setResult] = useState<DiagnosticsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runDiagnostics = async () => {
    if (!accessToken) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMCPServerDiagnostics(accessToken, serverId);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to run diagnostics");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runDiagnostics();
  }, [serverId, accessToken]);

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>Connection Diagnostics</Title>
        <Button
          icon={<ReloadOutlined />}
          onClick={runDiagnostics}
          loading={loading}
        >
          Re-run
        </Button>
      </div>

      {loading && !result && (
        <div className="flex justify-center py-8">
          <Spin tip="Running diagnostics..." />
        </div>
      )}

      {error && (
        <Alert
          type="error"
          message="Diagnostics Failed"
          description={error}
          showIcon
          className="mb-4"
        />
      )}

      {result && (
        <>
          <div className="mb-4">
            <Tag
              color={result.overall_status === "healthy" ? "green" : "red"}
              className="text-sm px-3 py-1"
            >
              {result.overall_status === "healthy"
                ? "All Checks Passing"
                : "Issues Detected"}
            </Tag>
          </div>

          <List
            dataSource={result.checks}
            renderItem={(check: DiagnosticCheck) => (
              <List.Item>
                <List.Item.Meta
                  avatar={
                    check.status === "pass" ? (
                      <CheckCircleOutlined
                        style={{ fontSize: 20, color: "#52c41a" }}
                      />
                    ) : (
                      <CloseCircleOutlined
                        style={{ fontSize: 20, color: "#ff4d4f" }}
                      />
                    )
                  }
                  title={
                    <span className="font-medium">
                      {CHECK_LABELS[check.name] || check.name}
                    </span>
                  }
                  description={
                    <div>
                      <Text>{check.message}</Text>
                      {check.details && (
                        <div className="mt-1 flex flex-wrap gap-2">
                          {Object.entries(check.details).map(
                            ([key, value]) =>
                              value != null && (
                                <Tag key={key} className="text-xs">
                                  {key}: {String(value)}
                                </Tag>
                              )
                          )}
                        </div>
                      )}
                    </div>
                  }
                />
              </List.Item>
            )}
          />

          {result.suggestions.length > 0 && (
            <Alert
              type="warning"
              message="Suggestions"
              description={
                <ul className="list-disc pl-4 mt-1">
                  {result.suggestions.map((s, i) => (
                    <li key={i} className="text-sm">
                      {s}
                    </li>
                  ))}
                </ul>
              }
              showIcon
              className="mt-4"
            />
          )}
        </>
      )}
    </Card>
  );
};

export default MCPServerDiagnostics;
