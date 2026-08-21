import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import { JSON_MAX_HEIGHT, SPACING_LARGE } from "./constants";

interface JsonViewerProps {
  data: any;
  mode: "formatted";
}

/**
 * Displays JSON data in formatted tree view.
 * Uses an interactive tree component for easy navigation.
 */
export function JsonViewer({ data }: JsonViewerProps) {
  if (!data) return <span className="text-muted-foreground">No data</span>;

  return (
    <div
      className="bg-background"
      style={{
        maxHeight: JSON_MAX_HEIGHT,
        overflow: "auto",
        padding: SPACING_LARGE,
        borderRadius: 4,
      }}
    >
      <div className="**:[[role='tree']]:bg-background! **:[[role='tree']]:text-foreground">
        <JsonView data={data} style={defaultStyles} clickToExpandNode={true} />
      </div>
    </div>
  );
}
