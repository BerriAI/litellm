import React from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getHeuristicFirstContextLimitError } from "./build_complexity_router_config";

type Props = { value: number | undefined; onChange: (value: number | undefined) => void };

export default function HeuristicFirstContextLimitField({ value, onChange }: Props) {
  const { t } = useTranslation();
  const [draft, setDraft] = React.useState<string | null>(null);
  const error = getHeuristicFirstContextLimitError(value);
  const displayDraft = draft !== null && Object.is(value, draft.trim() === "" ? undefined : Number(draft));
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setDraft(raw);
    onChange(raw.trim() === "" ? undefined : Number(raw));
  };
  return (
    <div className="mt-4 space-y-2">
      <Label htmlFor="auto-router-context-limit" className="block font-semibold">
        {t("addModel.classificationMethodConfig.contextLimit.label")}
      </Label>
      <Input
        id="auto-router-context-limit"
        type="text"
        inputMode="numeric"
        placeholder="8000"
        value={displayDraft ? draft : value?.toString() ?? ""}
        onChange={handleChange}
        aria-invalid={Boolean(error)}
        aria-describedby={`auto-router-context-limit-help${error ? " auto-router-context-limit-error" : ""}`}
      />
      <p id="auto-router-context-limit-help" className="text-sm text-muted-foreground">
        {t("addModel.classificationMethodConfig.contextLimit.help")}
      </p>
      {error && (
        <p id="auto-router-context-limit-error" className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
