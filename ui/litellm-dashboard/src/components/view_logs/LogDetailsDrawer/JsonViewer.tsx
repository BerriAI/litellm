import { Typography } from "antd";
import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import {
  JSON_MAX_HEIGHT,
  FONT_SIZE_SMALL,
  COLOR_BG_LIGHT,
  SPACING_LARGE,
  FONT_FAMILY_MONO,
  VIEW_MODE_JSON,
} from "./constants";

const { Text } = Typography;

export type ViewMode = "formatted" | "json";

interface JsonViewerProps {
  data: any;
  mode: ViewMode;
}

/**
 * Displays JSON data in either formatted tree view or raw JSON format.
 * Formatted view uses an interactive tree, JSON view shows raw stringified output.
 */
export function JsonViewer({ data, mode }: JsonViewerProps) {
  if (!data) return <Text type="secondary">No data</Text>;

  if (mode === VIEW_MODE_JSON) {
    return (
      <pre
        style={{
          margin: 0,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          maxHeight: JSON_MAX_HEIGHT,
          overflow: "auto",
          fontSize: FONT_SIZE_SMALL,
          background: COLOR_BG_LIGHT,
          padding: SPACING_LARGE,
          borderRadius: 4,
          fontFamily: FONT_FAMILY_MONO,
        }}
      >
        {JSON.stringify(data, null, 2)}
      </pre>
    );
  }

  // Formatted tree view
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
