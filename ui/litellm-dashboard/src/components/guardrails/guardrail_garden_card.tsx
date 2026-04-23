import React, { useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { GuardrailCardInfo } from "./guardrail_garden_data";
import { cn } from "@/lib/utils";

const LogoWithFallback: React.FC<{ src: string; name: string }> = ({
  src,
  name,
}) => {
  const [hasError, setHasError] = useState(false);

  if (hasError || !src) {
    return (
      <div className="w-7 h-7 rounded-md bg-muted flex items-center justify-center text-sm font-semibold text-muted-foreground shrink-0">
        {name?.charAt(0) || "?"}
      </div>
    );
  }

  return (
    <img
      src={src}
      alt=""
      className="w-7 h-7 rounded-md object-contain shrink-0"
      onError={() => setHasError(true)}
    />
  );
};

const GuardrailCard: React.FC<{
  card: GuardrailCardInfo;
  onClick: () => void;
}> = ({ card, onClick }) => {
  return (
    <div
      onClick={onClick}
      className={cn(
        "rounded-xl border border-border bg-background px-5 pt-5 pb-4",
        "cursor-pointer transition-colors transition-shadow",
        "hover:border-primary/50 hover:shadow-sm",
        "flex flex-col min-h-[170px]",
      )}
    >
      <div className="flex items-center gap-2.5 mb-2.5">
        <LogoWithFallback src={card.logo} name={card.name} />
        <span className="text-sm font-semibold text-foreground leading-tight">
          {card.name}
        </span>
      </div>

      <p className="line-clamp-3 text-xs text-muted-foreground leading-relaxed m-0 flex-1">
        {card.description}
      </p>

      {card.eval && (
        <div className="mt-2.5 flex items-center gap-1">
          <CheckCircle2 className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
          <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
            F1: {card.eval.f1}% &middot; {card.eval.testCases} test cases
          </span>
        </div>
      )}
    </div>
  );
};

export default GuardrailCard;
