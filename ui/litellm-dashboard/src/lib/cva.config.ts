import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export { cva, type VariantProps } from "class-variance-authority";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const cx = cn;
