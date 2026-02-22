"use client";

import React, { useState } from "react";
import { Title, DateRangePicker, DateRangePickerValue } from "@tremor/react";
import GuardrailsTableView from "@/components/GuardrailsPage/GuardrailsTableView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const GuardrailsMetricsPage = () => {
  const { accessToken } = useAuthorized();
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  // Format dates as YYYY-MM-DD
  const formatDate = (date: Date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  };

  const startDate = dateValue.from ? formatDate(dateValue.from) : "";
  const endDate = dateValue.to ? formatDate(dateValue.to) : "";

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-6">
        <Title>Guardrails Performance</Title>
        <DateRangePicker
          value={dateValue}
          onValueChange={setDateValue}
          enableSelect={true}
        />
      </div>

      {startDate && endDate && (
        <GuardrailsTableView
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
        />
      )}
    </div>
  );
};

export default GuardrailsMetricsPage;
