"use client";

import { useSyncExternalStore } from "react";
import i18n from "@/i18n/i18n";
import { shortcutLabel } from "./utils";

const subscribeToShortcut = () => () => {};
const getClientShortcut = () =>
  typeof navigator === "undefined"
    ? i18n.t("shell.shortcutLabel.ctrlShortcut")
    : shortcutLabel(navigator.platform, navigator.userAgent);
const getServerShortcut = () => i18n.t("shell.shortcutLabel.ctrlShortcut");
const subscribeToMount = () => () => {};
const getClientMounted = () => true;
const getServerMounted = () => false;

export function useShortcutLabel(): string {
  return useSyncExternalStore(subscribeToShortcut, getClientShortcut, getServerShortcut);
}

export function useClientMounted(): boolean {
  return useSyncExternalStore(subscribeToMount, getClientMounted, getServerMounted);
}
