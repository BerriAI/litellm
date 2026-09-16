import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo } from "react";

import { callbackRowMode } from "./LoggingCallbacksTableColumns";
import { AlertingObject } from "./types";

const CALLBACK_MODES = ["success", "failure", "success_and_failure"] as const;
type CallbackMode = (typeof CALLBACK_MODES)[number];

const EDIT_URL_STATE = {
  callback: parseAsString,
  callback_mode: parseAsStringLiteral(CALLBACK_MODES).withDefault("success"),
};

const toCallbackMode = (callback: AlertingObject): CallbackMode =>
  CALLBACK_MODES.find((mode) => mode === callbackRowMode(callback)) ?? "success";

export function useCallbackEditUrlState(callbacks: readonly AlertingObject[], isLoading: boolean) {
  const [{ callback: name, callback_mode: mode }, setTarget] = useQueryStates(EDIT_URL_STATE, { history: "push" });

  const editingCallback = useMemo(
    () =>
      callbacks.find(
        (callback) => !callback.read_only && callback.name === name && toCallbackMode(callback) === mode,
      ) ?? null,
    [callbacks, name, mode],
  );

  useEffect(() => {
    if (!isLoading && name !== null && editingCallback === null) void setTarget(null, { history: "replace" });
  }, [isLoading, name, editingCallback, setTarget]);

  const openEdit = useCallback(
    (callback: AlertingObject) => void setTarget({ callback: callback.name, callback_mode: toCallbackMode(callback) }),
    [setTarget],
  );

  const closeEdit = useCallback(() => void setTarget(null), [setTarget]);

  return { editingCallback, openEdit, closeEdit };
}
