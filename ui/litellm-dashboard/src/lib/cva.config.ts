import { clsx, type ClassValue } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

export { cva, type VariantProps } from "class-variance-authority";

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      z: [{ z: ["raised", "chrome", "sticky", "sticky-pinned", "floating", "overlay", "popup"] }],
    },
  },
});

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const cx = cn;
