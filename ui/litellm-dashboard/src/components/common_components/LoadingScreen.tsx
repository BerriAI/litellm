import { useTranslation } from "react-i18next";
import { cx } from "@/lib/cva.config";
import { UiLoadingSpinner } from "../ui/ui-loading-spinner";
import { productNames } from "@/i18n/untranslated/productNames";

export default function LoadingScreen() {
  const { t } = useTranslation();
  return (
    <div className={cx("h-screen", "flex items-center justify-center gap-4")}>
      <div className="text-lg font-medium py-2 pr-4 border-r border-r-gray-200">{productNames.liteLLMWithLogo}</div>

      <div className="flex items-center justify-center gap-2">
        <UiLoadingSpinner className="size-4" />
        <span className="text-muted-foreground text-sm">{t("shell.loadingScreen.loading")}</span>
      </div>
    </div>
  );
}
