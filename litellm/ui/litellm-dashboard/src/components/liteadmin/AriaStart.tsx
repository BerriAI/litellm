import type { ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import type { AriaAccess } from "./operations";
import { ChartNoAxesColumnIncreasing, Coins, TrendingDown, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

const QUESTIONS: { label: ParseKeys; question: ParseKeys; Icon: LucideIcon }[] = [
  {
    label: "aria.start.questions.spendOverview.label",
    question: "aria.start.questions.spendOverview.question",
    Icon: Coins,
  },
  {
    label: "aria.start.questions.costByModel.label",
    question: "aria.start.questions.costByModel.question",
    Icon: ChartNoAxesColumnIncreasing,
  },
  {
    label: "aria.start.questions.findSavings.label",
    question: "aria.start.questions.findSavings.question",
    Icon: TrendingDown,
  },
];

export function AriaStart({
  composer,
  onQuestion,
  disabled,
  access,
}: {
  composer: ReactNode;
  onQuestion: (question: string) => void;
  disabled: boolean;
  access: AriaAccess;
}) {
  const { t } = useTranslation();
  const usageDescription =
    access.mode === "usage" && access.scope.type !== "global"
      ? t("aria.start.descriptionUser")
      : t("aria.start.descriptionGateway");
  const description = access.mode === "admin" ? t("aria.start.descriptionAdmin") : usageDescription;
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-10 sm:px-6 sm:pb-[16vh]">
      <div className="my-auto w-full shrink-0 space-y-6">
        <div className="space-y-3">
          <h2 className="text-3xl leading-tight font-medium tracking-[-0.035em] text-foreground sm:text-4xl">
            {t("aria.start.title")}
          </h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <div className="[&>div]:p-0">{composer}</div>
        <div aria-label={t("aria.start.suggestedQuestions")} className="flex flex-wrap gap-2">
          {QUESTIONS.map(({ label, question, Icon }) => (
            <Button
              key={label}
              variant="outline"
              aria-label={t(question)}
              disabled={disabled}
              onClick={() => onQuestion(t(question))}
              className="h-9 gap-2 rounded-full border-border bg-transparent dark:bg-transparent px-3.5 text-[13px] font-normal text-muted-foreground shadow-none hover:border-foreground/20 hover:bg-muted/50 hover:text-foreground"
            >
              <Icon aria-hidden="true" className="size-3.5" />
              {t(label)}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
