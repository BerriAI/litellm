import type { CSSProperties } from "react";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useIsDarkMode } from "./useIsDarkMode";

export type SyntaxTheme = Record<string, CSSProperties>;

export const useSyntaxTheme = (light: SyntaxTheme): SyntaxTheme => (useIsDarkMode() ? oneDark : light);
