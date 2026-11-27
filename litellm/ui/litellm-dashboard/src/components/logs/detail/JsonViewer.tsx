import { useTheme } from "next-themes";
import { useTranslation } from "react-i18next";
import { JsonView, darkStyles, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";

interface JsonViewerProps {
  data: any;
  mode: "formatted";
}

/**
 * Displays JSON data in formatted tree view.
 * Uses an interactive tree component for easy navigation.
 */
export function JsonViewer({ data }: JsonViewerProps) {
  const { resolvedTheme } = useTheme();
  const { t } = useTranslation();

  if (!data) return <span className="text-muted-foreground">{t("logs.detail.jsonViewer.noData")}</span>;

  return (
    <div className="max-h-[400px] overflow-auto rounded-[4px] bg-background p-3">
      <div className="**:[[role='tree']]:bg-transparent!">
        <JsonView data={data} style={resolvedTheme === "dark" ? darkStyles : defaultStyles} clickToExpandNode={true} />
      </div>
    </div>
  );
}
