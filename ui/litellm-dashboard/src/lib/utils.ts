import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Canonical shadcn helper that merges tailwind classes safely.
 *
 * Use this in every shadcn primitive and in feature code when combining
 * conditional class strings. Prefer this over raw template literals / clsx
 * directly so tailwind-merge has a chance to dedupe conflicting utilities.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
