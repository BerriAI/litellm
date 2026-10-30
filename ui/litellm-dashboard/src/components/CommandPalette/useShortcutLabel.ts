"use client";

import { useSyncExternalStore } from "react";
import { shortcutLabel } from "./utils";

const subscribeToShortcut = () => () => {};
const getClientShortcut = () =>
  typeof navigator === "undefined" ? "Ctrl K" : shortcutLabel(navigator.platform, navigator.userAgent);
const getServerShortcut = () => "Ctrl K";
const subscribeToMount = () => () => {};
const getClientMounted = () => true;
const getServerMounted = () => false;

export function useShortcutLabel(): string {
  return useSyncExternalStore(subscribeToShortcut, getClientShortcut, getServerShortcut);
}

export function useClientMounted(): boolean {
  return useSyncExternalStore(subscribeToMount, getClientMounted, getServerMounted);
}
