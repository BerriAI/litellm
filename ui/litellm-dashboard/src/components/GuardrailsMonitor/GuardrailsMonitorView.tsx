import React, { useState } from "react";
import { GuardrailsOverview } from "./GuardrailsOverview";
import { GuardrailDetail } from "./GuardrailDetail";

type View =
  | { type: "overview" }
  | { type: "detail"; guardrailId: string };

export default function GuardrailsMonitorView() {
  const [view, setView] = useState<View>({ type: "overview" });

  const handleSelectGuardrail = (id: string) => {
    setView({ type: "detail", guardrailId: id });
  };

  const handleBack = () => {
    setView({ type: "overview" });
  };

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      {view.type === "overview" ? (
        <GuardrailsOverview onSelectGuardrail={handleSelectGuardrail} />
      ) : (
        <GuardrailDetail guardrailId={view.guardrailId} onBack={handleBack} />
      )}
    </div>
  );
}
