import React from "react";
import { Typography } from "antd";
import { CheckCircleTwoTone, CloseCircleTwoTone, LoadingOutlined } from "@ant-design/icons";
import { testModelGroupConnection, ModelGroupConnectionResult } from "../networking";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";

const { Text } = Typography;

interface AutoRouterConnectionTestProps {
  accessToken: string;
  targets: AutoRouterTestTarget[];
  onTestComplete?: () => void;
}

type TargetResult = { status: "pending" } | ModelGroupConnectionResult;

const cleanErrorMessage = (error: string): string => {
  const mainError = error.split("stack trace:")[0].trim();
  return mainError.replace(/^litellm\.(.*?)Error: /, "");
};

const AutoRouterConnectionTest: React.FC<AutoRouterConnectionTestProps> = ({
  accessToken,
  targets,
  onTestComplete,
}) => {
  const [results, setResults] = React.useState<TargetResult[]>(() => targets.map(() => ({ status: "pending" })));

  React.useEffect(() => {
    let cancelled = false;
    const run = async () => {
      await Promise.all(
        targets.map(async (target, index) => {
          const result = await testModelGroupConnection(accessToken, target.modelGroup, target.mode);
          if (cancelled) return;
          const cleaned: TargetResult =
            result.status === "error" ? { status: "error", error: cleanErrorMessage(result.error) } : result;
          setResults((prev) => prev.map((r, i) => (i === index ? cleaned : r)));
        }),
      );
      if (!cancelled && onTestComplete) onTestComplete();
    };
    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- probes run once per mount; the parent remounts via `key` to start a fresh test, and re-running on prop identity changes would refire paid requests
  }, []);

  if (targets.length === 0) {
    return <Text type="secondary">No complexity tiers are configured yet, so there is nothing to test.</Text>;
  }

  return (
    <div className="space-y-3">
      <Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
        Each configured tier routes to a saved model group. Test Connection sends a minimal request through the proxy to
        each one, exactly as the auto router would.
      </Text>
      {targets.map((target, index) => {
        const result = results[index] ?? { status: "pending" };
        return (
          <div
            key={`${target.modelGroup}-${target.mode}`}
            data-testid="auto-router-test-row"
            style={{
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              padding: "12px 16px",
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
            }}
          >
            <div style={{ fontSize: 18, lineHeight: "24px" }}>
              {result.status === "pending" && <LoadingOutlined data-testid="test-status-pending" />}
              {result.status === "success" && (
                <CheckCircleTwoTone twoToneColor="#52c41a" data-testid="test-status-success" />
              )}
              {result.status === "error" && (
                <CloseCircleTwoTone twoToneColor="#ff4d4f" data-testid="test-status-error" />
              )}
            </div>
            <div style={{ flex: 1 }}>
              <Text strong>{target.labels.join(", ")}</Text>{" "}
              <Text type="secondary">
                {"->"} {target.modelGroup}
                {target.mode === "embedding" ? " (embedding)" : ""}
              </Text>
              {result.status === "error" && (
                <Text
                  type="danger"
                  data-testid="test-error-message"
                  style={{ display: "block", marginTop: 4, fontSize: 13 }}
                >
                  {result.error}
                </Text>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default AutoRouterConnectionTest;
