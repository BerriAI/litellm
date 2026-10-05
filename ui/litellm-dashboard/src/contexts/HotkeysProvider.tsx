"use client";

import { HotkeysProvider as HotkeysScopeProvider } from "react-hotkeys-hook";

export default function HotkeysProvider({ children }: { children: React.ReactNode }) {
  return <HotkeysScopeProvider>{children}</HotkeysScopeProvider>;
}
