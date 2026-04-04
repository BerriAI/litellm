import type { DateRangePickerValue } from "@tremor/react";
import { Tabs } from "antd";
import React, { useCallback, useMemo, useState } from "react";
import { formatDate } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { MCPOverview } from "./MCPOverview";
import { MCPServerDetail } from "./MCPServerDetail";
import { MCPAlertRules } from "./MCPAlertRules";

type View =
  | { type: "overview" }
  | { type: "detail"; serverName: string };

interface MCPMonitorViewProps {
  accessToken?: string | null;
}

const defaultEnd = new Date();
const defaultStart = new Date();
defaultStart.setDate(defaultStart.getDate() - 7);

export default function MCPMonitorView({ accessToken = null }: MCPMonitorViewProps) {
  const [view, setView] = useState<View>({ type: "overview" });
  const [activeTab, setActiveTab] = useState("servers");

  const initialFrom = useMemo(() => new Date(defaultStart), []);
  const initialTo = useMemo(() => new Date(defaultEnd), []);

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: initialFrom,
    to: initialTo,
  });

  const startDate = dateValue.from ? formatDate(dateValue.from) : "";
  const endDate = dateValue.to ? formatDate(dateValue.to) : "";

  const handleDateChange = useCallback((newValue: DateRangePickerValue) => {
    setDateValue(newValue);
  }, []);

  const handleSelectServer = (serverName: string) => {
    setView({ type: "detail", serverName });
  };

  const handleBack = () => {
    setView({ type: "overview" });
  };

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold text-gray-900">MCP Server Monitor</h1>
        <AdvancedDatePicker
          value={dateValue}
          onValueChange={handleDateChange}
          label=""
          showTimeRange={false}
        />
      </div>

      {view.type === "detail" ? (
        <MCPServerDetail
          serverName={view.serverName}
          onBack={handleBack}
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
        />
      ) : (
        <>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              { key: "servers", label: "Servers & Logs" },
              { key: "alerts", label: "Alert Rules" },
            ]}
            className="mb-4"
          />
          {activeTab === "servers" && (
            <MCPOverview
              accessToken={accessToken}
              startDate={startDate}
              endDate={endDate}
              onSelectServer={handleSelectServer}
            />
          )}
          {activeTab === "alerts" && (
            <MCPAlertRules accessToken={accessToken} />
          )}
        </>
      )}
    </div>
  );
}
