import { Typography } from "antd";
import { COLOR_SECONDARY, FONT_FAMILY_MONO, FONT_SIZE_MEDIUM, SPACING_SMALL, SPACING_MEDIUM } from "./constants";

const { Text } = Typography;

interface TokenFlowProps {
  prompt?: number;
  completion?: number;
  total?: number;
}

/**
 * Displays token usage in a flow format: "prompt → completion (Σ total)"
 * Makes it easy to see the relationship between input, output, and total tokens.
 */
export function TokenFlow({ prompt = 0, completion = 0, total = 0 }: TokenFlowProps) {
  return (
    <Text style={{ fontFamily: FONT_FAMILY_MONO, fontSize: FONT_SIZE_MEDIUM }}>
      {prompt.toLocaleString()}
      <span style={{ color: COLOR_SECONDARY, margin: `0 ${SPACING_SMALL}px` }}>→</span>
      {completion.toLocaleString()}
      <span style={{ color: COLOR_SECONDARY, marginLeft: SPACING_MEDIUM }}>
        (Σ {total.toLocaleString()})
      </span>
    </Text>
  );
}
