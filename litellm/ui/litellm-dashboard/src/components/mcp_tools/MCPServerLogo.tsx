import type { ReactNode } from "react";
import { Logo } from "@/components/molecules/logo/Logo";
import { getMcpServerLogoSrc } from "./mcpServerLogoSrc";

export function MCPServerLogo({
  server,
  name,
  className,
  fallback,
}: {
  server: Parameters<typeof getMcpServerLogoSrc>[0];
  name: string;
  className: string;
  fallback?: ReactNode;
}) {
  const src = getMcpServerLogoSrc(server);
  return src ? (
    <Logo src={src} label={name} className={className} fallback={fallback} referrerPolicy="no-referrer" />
  ) : (
    fallback ?? null
  );
}
