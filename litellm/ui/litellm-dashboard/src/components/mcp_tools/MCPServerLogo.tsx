import { type ReactNode, useState } from "react";
import { Logo } from "@/components/molecules/logo/Logo";
import { getMcpServerLogoSrc } from "./mcpServerLogoSrc";

type MCPServerLogoAppearance = {
  name: string;
  className: string;
  fallback?: ReactNode;
};

function MCPServerLogoImage({
  candidates,
  name,
  className,
  fallback,
}: MCPServerLogoAppearance & { candidates: string[] }) {
  const [index, setIndex] = useState(0);
  const src = candidates[index];
  return (
    <Logo
      key={src}
      src={src}
      label={name}
      className={className}
      fallback={fallback}
      referrerPolicy="no-referrer"
      onError={() => setIndex(index + 1)}
    />
  );
}

export function MCPServerLogo({
  server,
  ...appearance
}: MCPServerLogoAppearance & { server: Parameters<typeof getMcpServerLogoSrc>[0] }) {
  const candidates = getMcpServerLogoSrc(server);
  if (candidates.length === 0) return appearance.fallback ?? null;

  return <MCPServerLogoImage key={JSON.stringify(candidates)} candidates={candidates} {...appearance} />;
}
