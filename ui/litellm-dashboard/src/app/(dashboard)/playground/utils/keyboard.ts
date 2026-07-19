import type { KeyboardEvent } from "react";

export const isImeComposing = (event: KeyboardEvent<HTMLElement>): boolean =>
  event.nativeEvent.isComposing || event.keyCode === 229;
