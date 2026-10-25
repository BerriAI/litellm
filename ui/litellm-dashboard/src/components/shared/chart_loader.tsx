import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { UiLoadingSpinner } from "../ui/ui-loading-spinner";

interface ChartLoaderProps {
  isDateChanging?: boolean;
}

export const ChartLoader: React.FC<ChartLoaderProps> = ({ isDateChanging = false }) => {
  const { t } = useLanguage();
  return (
    <div className="flex items-center justify-center h-40">
      <div className="flex items-center justify-center gap-3">
        <UiLoadingSpinner className="size-5" />
        <div className="flex flex-col">
          <span className="text-muted-foreground text-sm font-medium">
            {isDateChanging ? t("Processing date selection...") : t("Loading chart data...")}
          </span>
          <span className="text-muted-foreground text-xs mt-1">
            {isDateChanging ? t("This will only take a moment") : t("Fetching your data")}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ChartLoader;
