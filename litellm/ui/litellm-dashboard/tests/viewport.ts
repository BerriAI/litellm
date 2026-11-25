import { act } from "@testing-library/react";
import { vi } from "vitest";

type Subscription = { media: string; listener: () => void };

const pxBound = (media: string, feature: "min-width" | "max-width") => {
  const match = new RegExp(`${feature}:\\s*([\\d.]+)px`).exec(media);
  return match ? Number(match[1]) : undefined;
};

const matchesWidth = (media: string, width: number) => {
  const min = pxBound(media, "min-width");
  const max = pxBound(media, "max-width");
  if (min === undefined && max === undefined) return false;
  const aboveMin = min === undefined || width >= min;
  const belowMax = max === undefined || width <= max;
  return aboveMin && belowMax;
};

export function stubViewportMatchMedia() {
  const viewport: { cssWidth?: number } = {};
  const subscriptions = new Set<Subscription>();
  const currentWidth = () => viewport.cssWidth ?? window.innerWidth;

  vi.stubGlobal("matchMedia", (media: string) => ({
    media,
    get matches() {
      return matchesWidth(media, currentWidth());
    },
    addEventListener: (_type: string, listener: () => void) => subscriptions.add({ media, listener }),
    removeEventListener: (_type: string, listener: () => void) =>
      subscriptions.forEach((subscription) => {
        if (subscription.media === media && subscription.listener === listener) subscriptions.delete(subscription);
      }),
  }));

  return async function resizeTo(cssWidth: number) {
    const previousWidth = currentWidth();
    const flipped = [...subscriptions].filter(
      ({ media }) => matchesWidth(media, previousWidth) !== matchesWidth(media, cssWidth),
    );
    viewport.cssWidth = cssWidth;
    vi.stubGlobal("innerWidth", Math.floor(cssWidth));
    await act(async () => flipped.forEach(({ listener }) => listener()));
  };
}
