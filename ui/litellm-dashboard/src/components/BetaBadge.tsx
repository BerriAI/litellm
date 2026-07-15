import { Badge } from "antd";

export default function BetaBadge({ children, dot = false }: { children?: React.ReactNode; dot?: boolean }) {
  return children ? (
    <Badge color="gold" count={dot ? undefined : "Beta"} dot={dot}>
      {children}
    </Badge>
  ) : (
    <Badge color="gold" count={dot ? undefined : "Beta"} dot={dot} />
  );
}
