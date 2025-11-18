import React, { useId } from "react";
import { useSafeLayoutEffect } from "@/hooks/use-safe-layout-effect";
import { cx } from "@/lib/cva.config";

type LoadingSpinnerProps = React.SVGProps<SVGSVGElement>;

export function UiLoadingSpinner({ className = "", ...props }: LoadingSpinnerProps) {
  const id = useId();

  useSafeLayoutEffect(() => {
    const animations = document
      .getAnimations()
      .filter((a) => a instanceof CSSAnimation && a.animationName === "spin") as CSSAnimation[];

    const self = animations.find((a) => (a.effect as KeyframeEffect).target?.getAttribute("data-spinner-id") === id);

    const anyOther = animations.find(
      (a) => a.effect instanceof KeyframeEffect && a.effect.target?.getAttribute("data-spinner-id") !== id,
    );

    if (self && anyOther) {
      self.currentTime = anyOther.currentTime;
    }
  }, [id]);

  return (
    <svg
      data-spinner-id={id}
      className={cx("pointer-events-none size-12 animate-spin text-current", className)}
      fill="none"
      viewBox="0 0 24 24"
      {...props}
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
  );
}
