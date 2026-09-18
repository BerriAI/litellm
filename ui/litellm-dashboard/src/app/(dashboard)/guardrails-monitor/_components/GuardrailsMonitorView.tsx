import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { parseAsString, useQueryState } from "nuqs";
import React, { useCallback, useMemo, useState } from "react";
import { formatDate } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { GuardrailDetail } from "./GuardrailDetail";
import { GuardrailsOverview } from "./GuardrailsOverview";

interface GuardrailsMonitorViewProps {
  accessToken?: string | null;
}

const defaultEnd = new Date();
const defaultStart = new Date();
defaultStart.setDate(defaultStart.getDate() - 7);

export default function GuardrailsMonitorView({ accessToken = null }: GuardrailsMonitorViewProps) {
  const [selectedGuardrailId, setSelectedGuardrailId] = useQueryState(
    "guardrail",
    parseAsString.withOptions({ history: "push" }),
  );

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

  const handleSelectGuardrail = (id: string) => {
    void setSelectedGuardrailId(id);
  };

  const handleBack = () => {
    void setSelectedGuardrailId(null, { history: "replace" });
  };

  const dateRangeControl = (
    <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} label="" showTimeRange={false} />
  );

  return (
    <main className="w-full min-w-0 flex-1 p-8">
      {!selectedGuardrailId ? (
        <GuardrailsOverview
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
          onSelectGuardrail={handleSelectGuardrail}
          dateRangeControl={dateRangeControl}
        />
      ) : (
        <>
          <div className="mb-4 flex items-center justify-end">{dateRangeControl}</div>
          <GuardrailDetail
            guardrailId={selectedGuardrailId}
            onBack={handleBack}
            accessToken={accessToken}
            startDate={startDate}
            endDate={endDate}
          />
        </>
      )}
    </main>
  );
}
