import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import {
  JSON_MAX_HEIGHT,
  COLOR_BG_LIGHT,
  SPACING_LARGE,
} from "./constants";

interface JsonViewerProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any;
  mode: "formatted";
}

/**
 * Displays JSON data in formatted tree view.
 */
export function JsonViewer({ data }: JsonViewerProps) {
  if (!data) return <span className="text-muted-foreground">No data</span>;

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
