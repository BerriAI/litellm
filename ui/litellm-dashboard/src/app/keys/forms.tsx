import { cx } from "@/lib/cva.config";
import { forwardRef } from "react";

export function UiFormGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cx(
        "flex items-start gap-6 py-5",
        "border-b last-of-type:border-b-0 border-dashed",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormLabelGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cx("flex flex-col gap-1.5 max-w-[320px]", className)}
      {...props}
    />
  );
}

export function UiFormLabel({
  className,
  ...props
}: React.ComponentProps<"label">) {
  return (
    <label
      className={cx(
        "block tracking-tight text-[15px] text-neutral-900",
        "",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormDescription({
  className,
  ...props
}: React.ComponentProps<"p">) {
  return (
    <p
      className={cx(
        "text-[13px]/[1.7] tracking-tight text-neutral-500",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormContent({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div className={cx("flex grow flex-col gap-1.5", className)} {...props} />
  );
}

export const UiFormTextInput = forwardRef<
  HTMLInputElement,
  React.ComponentProps<"input">
>(function UiFormTextInput({ className, ...props }, ref) {
  return (
    <input
      ref={ref}
      className={cx(
        "block w-full max-w-[400px] h-[34px] truncate rounded-md bg-neutral-50 px-4",
        "border-none",
        "ring-[0.7px] ring-black/[0.12]",
        "text-[13px] font-normal tracking-tight text-neutral-900",
        "placeholder:text-neutral-400",
        className,
      )}
      {...props}
    />
  );
});
