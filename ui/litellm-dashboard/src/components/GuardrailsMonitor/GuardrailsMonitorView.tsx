import type { DateRangePickerValue } from "@tremor/react";
import React, { useCallback, useMemo, useState } from "react";
import { formatDate } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { GuardrailDetail } from "./GuardrailDetail";
import { GuardrailsOverview } from "./GuardrailsOverview";

type View =
  | { type: "overview" }
  | { type: "detail"; guardrailId: string };

interface GuardrailsMonitorViewProps {
  accessToken?: string | null;
}

const defaultEnd = new Date();
const defaultStart = new Date();
defaultStart.setDate(defaultStart.getDate() - 7);

export default function GuardrailsMonitorView({ accessToken = null }: GuardrailsMonitorViewProps) {
  const [view, setView] = useState<View>({ type: "overview" });

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
    setView({ type: "detail", guardrailId: id });
  };

  const handleBack = () => {
    setView({ type: "overview" });
  };

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      <div className="flex items-center justify-end mb-4">
        <AdvancedDatePicker
          value={dateValue}
          onValueChange={handleDateChange}
          label=""
          showTimeRange={false}
        />
      </div>
      {view.type === "overview" ? (
        <GuardrailsOverview
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
          onSelectGuardrail={handleSelectGuardrail}
        />
      ) : (
        <GuardrailDetail
          guardrailId={view.guardrailId}
          onBack={handleBack}
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
        />
      )}
    </div>
  );
}
