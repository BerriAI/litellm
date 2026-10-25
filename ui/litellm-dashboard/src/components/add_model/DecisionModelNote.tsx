import { useLanguage } from "@/contexts/LanguageContext";
import { Info } from "lucide-react";
import { useFormContext, useWatch } from "react-hook-form";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import {
  DECISIONS_DOCS_URL,
  SYSTEM_ONE_PLAYGROUND_ROUTE,
  isDecisionSelection,
  type DecisionCatalog,
} from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";
import type { MountedFormValues } from "../common_components/MountedFormField";

interface DecisionModelNoteProps {
  catalog: DecisionCatalog;
  litellmProvider: string | undefined;
}

const selectedModelNames = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.filter((model): model is string => typeof model === "string");
  }
  return typeof value === "string" && value !== "" ? [value] : [];
};

export default function DecisionModelNote({ catalog, litellmProvider }: DecisionModelNoteProps) {
  const { t } = useLanguage();
  const { control } = useFormContext<MountedFormValues>();
  const model = useWatch({ control, name: "model" });

  if (!isDecisionSelection(catalog, litellmProvider, selectedModelNames(model))) {
    return null;
  }

  return (
    <Alert variant="info" role="note" aria-label={t("Decision model notice")} className="mb-4">
      <Info />
      <AlertTitle>{t("Decision model")}</AlertTitle>
      <AlertDescription>
        <p>
          {t("For decision requests, use {decisions} (OpenAI format) or {systemOne} (System One format).", {
            decisions: "/v1/decisions",
            systemOne: "/v1/systemone",
          })}{" "}
          <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
            {t("How to call decision models")}
          </a>
        </p>
        <p>
          {t("After you add it,")}{" "}
          <a href={uiHref(SYSTEM_ONE_PLAYGROUND_ROUTE)} className="underline">
            {t("test it in the Decisions playground")}
          </a>
        </p>
      </AlertDescription>
    </Alert>
  );
}
