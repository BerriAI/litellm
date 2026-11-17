import { Info } from "lucide-react";
import { useFormContext, useWatch } from "react-hook-form";
import { Trans, useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const model = useWatch({ control, name: "model" });

  if (!isDecisionSelection(catalog, litellmProvider, selectedModelNames(model))) {
    return null;
  }

  return (
    <Alert variant="info" role="note" aria-label={t("addModel.decisionModelNote.ariaLabel")} className="mb-4">
      <Info />
      <AlertTitle>{t("addModel.decisionModelNote.title")}</AlertTitle>
      <AlertDescription>
        <p>
          <Trans
            i18nKey="addModel.decisionModelNote.callInstructions"
            components={{ decisions: <code />, systemOne: <code /> }}
          />{" "}
          <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
            {t("addModel.decisionModelNote.howToCall")}
          </a>
        </p>
        <p>
          {t("addModel.decisionModelNote.afterYouAddIt")}{" "}
          <a href={uiHref(SYSTEM_ONE_PLAYGROUND_ROUTE)} className="underline">
            {t("addModel.decisionModelNote.testInPlayground")}
          </a>
        </p>
      </AlertDescription>
    </Alert>
  );
}
