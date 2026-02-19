"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button, DateRangePicker, DateRangePickerValue } from "@tremor/react";
import GuardrailDetailView from "@/components/GuardrailsPage/GuardrailDetailView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const GuardrailDetailPage = () => {
  const params = useParams();
  const router = useRouter();
  const { accessToken } = useAuthorized();
  const guardrailName = decodeURIComponent(params.name as string);

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
        <div className="flex items-center gap-4">
          <Button
            size="xs"
            variant="secondary"
            onClick={() => router.push("/guardrails/metrics")}
          >
            ← Back to Overview
          </Button>
          <div>
            <h1 className="text-2xl font-bold">{guardrailName}</h1>
          </div>
        </div>
        <DateRangePicker
          value={dateValue}
          onValueChange={setDateValue}
          enableSelect={true}
        />
      </div>

      {startDate && endDate && (
        <GuardrailDetailView
          accessToken={accessToken}
          guardrailName={guardrailName}
          startDate={startDate}
          endDate={endDate}
        />
      )}
    </div>
  );
};

export default GuardrailDetailPage;
