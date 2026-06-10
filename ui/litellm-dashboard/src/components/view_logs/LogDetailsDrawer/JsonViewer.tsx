import { Typography } from "antd";
import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import { useTranslation } from "react-i18next";
import { JSON_MAX_HEIGHT, COLOR_BG_LIGHT, SPACING_LARGE } from "./constants";

const { Text } = Typography;

interface JsonViewerProps {
  data: any;
  mode: "formatted";
}

/**
 * Displays JSON data in formatted tree view.
 * Uses an interactive tree component for easy navigation.
 */
export function JsonViewer({ data }: JsonViewerProps) {
  const { t } = useTranslation();
  if (!data) return <Text type="secondary">{t("common.noData")}</Text>;

  return (
    <div
      style={{
        maxHeight: JSON_MAX_HEIGHT,
        overflow: "auto",
        background: COLOR_BG_LIGHT,
        padding: SPACING_LARGE,
        borderRadius: 4,
      }}
    >
      <div className="[&_[role='tree']]:bg-white [&_[role='tree']]:text-slate-900">
        <JsonView data={data} style={defaultStyles} clickToExpandNode={true} />
      </div>
    </div>
  );
}
