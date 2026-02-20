import {
  ArrowLeftOutlined,
  BellOutlined,
  CheckOutlined,
  CloseOutlined,
  SafetyOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { Card, Col, Grid, Title } from "@tremor/react";
import { Button, Input, Tabs } from "antd";
import React, { useState } from "react";
import { getGuardrailDetailOrDefault } from "./mockData";
import { LogViewer } from "./LogViewer";
import { MetricCard } from "./MetricCard";

interface GuardrailDetailProps {
  guardrailId: string;
  onBack: () => void;
}

const statusColors: Record<
  string,
  { bg: string; text: string; dot: string }
> = {
  healthy: { bg: "bg-green-50", text: "text-green-700", dot: "bg-green-500" },
  warning: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  critical: { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" },
};

export function GuardrailDetail({ guardrailId, onBack }: GuardrailDetailProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [showNotifyPanel, setShowNotifyPanel] = useState(false);
  const [notifySaved, setNotifySaved] = useState(false);
  const [notifyConfig, setNotifyConfig] = useState({
    failRateThreshold: "",
    apiErrorThreshold: "",
    webhookUrl: "",
  });
  const data = getGuardrailDetailOrDefault(guardrailId);
  const statusStyle = statusColors[data.status] ?? statusColors.healthy;

  const handleSaveNotify = () => {
    setNotifySaved(true);
    setTimeout(() => {
      setNotifySaved(false);
      setShowNotifyPanel(false);
    }, 1500);
  };

  return (
    <div>
      <div className="mb-6">
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={onBack}
          className="pl-0 mb-4"
        >
          Back to Overview
        </Button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <SafetyOutlined className="text-xl text-gray-400" />
              <h1 className="text-xl font-semibold text-gray-900">{data.name}</h1>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-full ${statusStyle.bg} ${statusStyle.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${statusStyle.dot}`} />
                {data.status.charAt(0).toUpperCase() + data.status.slice(1)}
              </span>
            </div>
            <p className="text-sm text-gray-500 ml-8">{data.description}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
              {data.provider}
            </span>
            <div className="relative">
              <Button
                type={showNotifyPanel ? "primary" : "default"}
                icon={<BellOutlined />}
                onClick={() => setShowNotifyPanel(!showNotifyPanel)}
                className={showNotifyPanel ? "bg-indigo-100 text-indigo-700 border-indigo-200" : ""}
              >
                Notify
              </Button>
              {showNotifyPanel && (
                <div className="absolute right-0 top-full mt-2 w-96 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                  <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
                    <div>
                      <h4 className="text-sm font-semibold text-gray-900">Configure Alerts</h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Get notified via webhook (Slack, Teams, etc.)
                      </p>
                    </div>
                    <Button
                      type="text"
                      icon={<CloseOutlined />}
                      onClick={() => setShowNotifyPanel(false)}
                      className="text-gray-400 hover:text-gray-600"
                    />
                  </div>
                  <div className="p-5 space-y-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1.5">
                        Fail Rate Threshold
                      </label>
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        placeholder="e.g. 15"
                        value={notifyConfig.failRateThreshold}
                        onChange={(e) =>
                          setNotifyConfig((prev) => ({
                            ...prev,
                            failRateThreshold: e.target.value,
                          }))
                        }
                        addonAfter="%"
                      />
                      <p className="text-xs text-gray-400 mt-1">Alert when fail rate exceeds this value</p>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1.5">
                        API Error Threshold
                      </label>
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        placeholder="e.g. 5"
                        value={notifyConfig.apiErrorThreshold}
                        onChange={(e) =>
                          setNotifyConfig((prev) => ({
                            ...prev,
                            apiErrorThreshold: e.target.value,
                          }))
                        }
                        addonAfter="%"
                      />
                      <p className="text-xs text-gray-400 mt-1">
                        Alert when guardrail API errors exceed this value
                      </p>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1.5">
                        Webhook URL
                      </label>
                      <Input
                        type="url"
                        placeholder="https://hooks.slack.com/services/..."
                        value={notifyConfig.webhookUrl}
                        onChange={(e) =>
                          setNotifyConfig((prev) => ({
                            ...prev,
                            webhookUrl: e.target.value,
                          }))
                        }
                      />
                      <p className="text-xs text-gray-400 mt-1">
                        Works with Slack, Microsoft Teams, Discord, or any webhook endpoint
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-gray-100 bg-gray-50 rounded-b-lg">
                    <Button onClick={() => setShowNotifyPanel(false)}>Cancel</Button>
                    <Button
                      type="primary"
                      onClick={handleSaveNotify}
                      disabled={notifySaved}
                      icon={notifySaved ? <CheckOutlined /> : undefined}
                    >
                      {notifySaved ? "Saved" : "Save Alert"}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          { key: "overview", label: "Overview" },
          { key: "logs", label: "Logs" },
        ]}
      />

      {activeTab === "overview" && (
        <div className="space-y-6 mt-4">
          <Grid numItems={2} numItemsMd={6} className="gap-4">
            <Col>
              <MetricCard label="Requests Evaluated" value={data.requestsEvaluated.toLocaleString()} />
            </Col>
            <Col>
              <MetricCard
                label="Fail Rate"
                value={`${data.failRate}%`}
                valueColor={
                  data.failRate > 15 ? "text-red-600" : data.failRate > 5 ? "text-amber-600" : "text-green-600"
                }
                icon={data.failRate > 15 ? <WarningOutlined className="text-red-400" /> : undefined}
              />
            </Col>
            <Col>
              <MetricCard
                label="False Positives"
                value={`${data.falsePositiveRate}%`}
                valueColor={
                  data.falsePositiveRate > 20
                    ? "text-red-600"
                    : data.falsePositiveRate > 10
                      ? "text-amber-600"
                      : "text-green-600"
                }
                subtitle={`${data.falsePositiveCount} of last 100 logs`}
                icon={
                  data.falsePositiveRate > 20 ? (
                    <WarningOutlined className="text-red-400" />
                  ) : undefined
                }
              />
            </Col>
            <Col>
              <MetricCard
                label="False Negatives"
                value={`${data.falseNegativeRate}%`}
                valueColor={
                  data.falseNegativeRate > 5
                    ? "text-red-600"
                    : data.falseNegativeRate > 2
                      ? "text-amber-600"
                      : "text-green-600"
                }
                subtitle={`${data.falseNegativeCount} of last 100 logs`}
                icon={
                  data.falseNegativeRate > 5 ? (
                    <WarningOutlined className="text-red-400" />
                  ) : undefined
                }
              />
            </Col>
            <Col>
              <MetricCard
                label="Avg. latency added"
                value={`${data.avgLatency}ms`}
                valueColor={
                  data.avgLatency > 150
                    ? "text-red-600"
                    : data.avgLatency > 50
                      ? "text-amber-600"
                      : "text-green-600"
                }
                subtitle={`p95: ${data.p95Latency}ms`}
              />
            </Col>
            <Col>
              <MetricCard
                label="Blocked Today"
                value="47"
                valueColor="text-red-600"
                subtitle="↑ 12% from yesterday"
              />
            </Col>
          </Grid>

          <Card className="bg-white border border-gray-200 rounded-lg p-6">
            <Title className="text-base font-semibold text-gray-900 mb-1">
              Root Cause Analysis
            </Title>
            <p className="text-xs text-gray-500 mb-4">Common patterns in failing requests</p>
            <div className="space-y-3">
              <div className="flex items-start gap-3 p-3 bg-red-50 rounded-lg border border-red-100">
                <WarningOutlined className="text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-red-800">
                    High sensitivity to medical terminology
                  </p>
                  <p className="text-xs text-red-600 mt-0.5">
                    34% of blocked requests contain common medical terms (e.g., &quot;symptoms&quot;,
                    &quot;treatment&quot;, &quot;medication&quot;) that are benign in context.
                    Consider adding an allowlist or relaxing sensitivity for these categories.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-amber-50 rounded-lg border border-amber-100">
                <WarningOutlined className="text-amber-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-amber-800">
                    False positives on educational content
                  </p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    22% of blocked requests are educational queries about safety topics. The guardrail
                    is flagging the topic itself rather than harmful intent.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <WarningOutlined className="text-gray-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-800">
                    Sensitivity may be too aggressive
                  </p>
                  <p className="text-xs text-gray-600 mt-0.5">
                    Many blocked requests may be false positives. Consider relaxing sensitivity or
                    adding allowlisted patterns to reduce blocks by ~40% while maintaining safety.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <LogViewer guardrailName={data.name} filterAction="blocked" />
        </div>
      )}

      {activeTab === "logs" && (
        <div className="mt-4">
          <LogViewer guardrailName={data.name} />
        </div>
      )}
    </div>
  );
}
