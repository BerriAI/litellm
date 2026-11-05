import type { ReactNode } from "react";
import type { AriaAccess } from "./operations";
import { ChartNoAxesColumnIncreasing, Coins, TrendingDown } from "lucide-react";
import { Button } from "@/components/ui/button";

const QUESTIONS = [
  { label: "Spend overview", question: "How much have I spent this week?", Icon: Coins },
  { label: "Cost by model", question: "Which models are costing the most?", Icon: ChartNoAxesColumnIncreasing },
  { label: "Find savings", question: "How can I reduce my spend?", Icon: TrendingDown },
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
  const usageDescription =
    access.mode === "usage" && access.scope.type === "user" ? "Ask about your usage" : "Ask about gateway usage";
  const description = access.mode === "admin" ? "Manage your gateway · Changes require approval" : usageDescription;
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-10 sm:px-6 sm:pb-[16vh]">
      <div className="my-auto w-full shrink-0 space-y-6">
        <div className="space-y-3">
          <h2 className="text-3xl leading-tight font-medium tracking-[-0.035em] text-foreground sm:text-4xl">
            What would you like to know?
          </h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <div className="[&>div]:p-0">{composer}</div>
        <div aria-label="Suggested questions" className="flex flex-wrap gap-2">
          {QUESTIONS.map(({ label, question, Icon }) => (
            <Button
              key={label}
              variant="outline"
              aria-label={question}
              disabled={disabled}
              onClick={() => onQuestion(question)}
              className="h-9 gap-2 rounded-full border-border bg-transparent dark:bg-transparent px-3.5 text-[13px] font-normal text-muted-foreground shadow-none hover:border-foreground/20 hover:bg-muted/50 hover:text-foreground"
            >
              <Icon aria-hidden="true" className="size-3.5" />
              {label}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
