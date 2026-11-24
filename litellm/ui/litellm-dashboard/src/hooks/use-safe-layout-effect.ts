import { DependencyList, EffectCallback, useEffect, useLayoutEffect } from "react";

export function useSafeLayoutEffect(effect: EffectCallback, deps?: DependencyList) {
  const isSSR = typeof window === "undefined";
  const safeUseLayoutEffect = isSSR ? useEffect : useLayoutEffect;
  return safeUseLayoutEffect(effect, deps);
}
