import React from "react";
import { useTranslation } from "react-i18next";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { ClassifierType } from "./classifier_types";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { restrictedBy } from "./TierRestrictions";

interface ClassifierTypeRadiosProps {
  value: ComplexityRouterConfigValue;
  classifierType: ClassifierType;
  onTypeChange: (classifierType: ClassifierType) => void;
}

const ClassifierTypeRadios: React.FC<ClassifierTypeRadiosProps> = ({ value, classifierType, onTypeChange }) => {
  const { t } = useTranslation();
  const scorerLocked = Boolean(value.custom_tier_set);
  const scorerLockedReason = restrictedBy(value, "heuristicClassifier")?.reason;
  return (
    <RadioGroup
      value={classifierType}
      onValueChange={(nextType: unknown) => onTypeChange(nextType as ClassifierType)}
      className="w-full"
    >
      <div className="flex w-full flex-col items-start gap-2">
        <SimpleTooltip content={scorerLockedReason}>
          <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
            <RadioGroupItem value="heuristic" className="mt-0.5" disabled={scorerLocked} />
            <span>
              <strong className="font-semibold">{t("addModel.classifierTypeRadios.heuristic.label")}</strong>{" "}
              <span className="text-muted-foreground">{t("addModel.classifierTypeRadios.heuristic.description")}</span>
            </span>
          </Label>
        </SimpleTooltip>
        <SimpleTooltip content={scorerLockedReason}>
          <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
            <RadioGroupItem value="heuristic_v2" className="mt-0.5" disabled={scorerLocked} />
            <span>
              <strong className="font-semibold">{t("addModel.classifierTypeRadios.heuristicV2.label")}</strong>{" "}
              <span className="text-muted-foreground">
                {t("addModel.classifierTypeRadios.heuristicV2.description")}
              </span>
            </span>
          </Label>
        </SimpleTooltip>
        <Label className="items-start font-normal leading-normal">
          <RadioGroupItem value="llm" className="mt-0.5" />
          <span>
            <strong className="font-semibold">{t("addModel.classifierTypeRadios.llm.label")}</strong>{" "}
            <span className="text-muted-foreground">{t("addModel.classifierTypeRadios.llm.description")}</span>
          </span>
        </Label>
        <Label className="items-start font-normal leading-normal">
          <RadioGroupItem value="jev" className="mt-0.5" />
          <span>
            <strong className="font-semibold">{t("addModel.classifierTypeRadios.oss.label")}</strong>{" "}
            <span className="text-muted-foreground">{t("addModel.classifierTypeRadios.oss.description")}</span>
          </span>
        </Label>
        <SimpleTooltip content={scorerLockedReason}>
          <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
            <RadioGroupItem value="heuristic_first" className="mt-0.5" disabled={scorerLocked} />
            <span>
              <strong className="font-semibold">{t("addModel.classifierTypeRadios.heuristicFirst.label")}</strong>{" "}
              <span className="text-muted-foreground">
                {t("addModel.classifierTypeRadios.heuristicFirst.description")}
              </span>
            </span>
          </Label>
        </SimpleTooltip>
        <SimpleTooltip content={scorerLockedReason}>
          <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
            <RadioGroupItem value="hybrid" className="mt-0.5" disabled={scorerLocked} />
            <span>
              <strong className="font-semibold">{t("addModel.classifierTypeRadios.hybrid.label")}</strong>{" "}
              <span className="text-muted-foreground">{t("addModel.classifierTypeRadios.hybrid.description")}</span>
            </span>
          </Label>
        </SimpleTooltip>
      </div>
    </RadioGroup>
  );
};

export default ClassifierTypeRadios;
