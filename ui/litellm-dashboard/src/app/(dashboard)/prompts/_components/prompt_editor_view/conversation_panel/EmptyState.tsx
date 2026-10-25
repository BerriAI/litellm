import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Bot } from "lucide-react";

interface EmptyStateProps {
  hasVariables: boolean;
}

const EmptyState: React.FC<EmptyStateProps> = ({ hasVariables }) => {
  const { t } = useLanguage();
  return (
    <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
      <Bot className="mb-4 size-12" aria-hidden="true" />
      <span className="text-base">
        {hasVariables
          ? t("Fill in the variables above, then type a message to start testing")
          : t("Type a message below to start testing your prompt")}
      </span>
    </div>
  );
};

export default EmptyState;
