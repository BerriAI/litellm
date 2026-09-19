import { parseAsString, useQueryState } from "nuqs";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { GuardrailDetail } from "./GuardrailDetail";
import { GuardrailsOverview } from "./GuardrailsOverview";
import { useMonitorDateRange } from "./useMonitorDateRange";

interface GuardrailsMonitorViewProps {
  accessToken?: string | null;
}

export default function GuardrailsMonitorView({ accessToken = null }: GuardrailsMonitorViewProps) {
  const [selectedGuardrailId, setSelectedGuardrailId] = useQueryState(
    "guardrail",
    parseAsString.withOptions({ history: "push" }),
  );
  const { startDate, endDate, pickerValue, setPickerValue } = useMonitorDateRange();

  const handleSelectGuardrail = (id: string) => {
    void setSelectedGuardrailId(id);
  };

  const handleBack = () => {
    void setSelectedGuardrailId(null, { history: "replace" });
  };

  const dateRangeControl = (
    <AdvancedDatePicker value={pickerValue} onValueChange={setPickerValue} label="" showTimeRange={false} />
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
