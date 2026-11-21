import type { CSSProperties } from "react";
import { useTheme } from "next-themes";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export type SyntaxTheme = Record<string, CSSProperties>;

export const useSyntaxTheme = (light: SyntaxTheme): SyntaxTheme =>
  useTheme().resolvedTheme === "dark" ? oneDark : light;
