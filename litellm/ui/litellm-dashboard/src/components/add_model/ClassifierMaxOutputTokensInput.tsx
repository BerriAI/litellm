import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ClassifierLLMConfig } from "./ComplexityRouterConfig";

const INPUT_ID = "classifier-max-output-tokens";

export default function ClassifierMaxOutputTokensInput({
  value,
  onChange,
}: {
  value: ClassifierLLMConfig;
  onChange: (value: ClassifierLLMConfig) => void;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<string | null>(null);
  const handleChange = (raw: string) => {
    setDraft(raw);
    const parsed = Number(raw);
    if (raw.trim() !== "" && (!Number.isSafeInteger(parsed) || parsed < 1)) return;
    const { max_output_tokens: _maxOutputTokens, ...config } = value;
    onChange(raw.trim() === "" ? config : { ...config, max_output_tokens: parsed });
  };

  return (
    <div>
      <Label htmlFor={INPUT_ID} className="block mb-1 font-semibold">
        {t("addModel.classificationMethodConfig.maxOutputTokens.label")}
      </Label>
      <Input
        id={INPUT_ID}
        type="text"
        inputMode="numeric"
        placeholder={t("addModel.classificationMethodConfig.maxOutputTokens.placeholder")}
        value={draft ?? String(value.max_output_tokens ?? "")}
        onChange={(event) => handleChange(event.target.value)}
        onBlur={() => setDraft(null)}
        aria-describedby={`${INPUT_ID}-help`}
        className="w-full"
      />
      <p id={`${INPUT_ID}-help`} className="text-xs text-muted-foreground">
        {t("addModel.classificationMethodConfig.maxOutputTokens.help")}
      </p>
    </div>
  );
}
