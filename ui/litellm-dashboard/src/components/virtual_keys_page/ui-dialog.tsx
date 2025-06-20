import { cx } from "@/lib/cva.config";
import { Dialog, DialogProps } from "@ariakit/react";

type UiDialogProps = DialogProps;

export function UiDialog({ store, className, ...props }: UiDialogProps) {
  return (
    <Dialog
      store={store}
      backdrop={
        <div
          className={cx(
            "fixed inset-0 bg-black/20",
            "transition-[opacity,backdrop-filter] duration-200 ease-out",
            "backdrop-blur-0 opacity-0",
            "data-[enter]:opacity-100 data-[enter]:backdrop-blur-sm",
          )}
        />
      }
      unmountOnHide
      className={cx(
        "fixed bg-[#FBFBFB] z-50 isolate",
        "[--inset:48px] inset-[--inset]",
        "flex flex-col min-h-0",
        "max-h-[calc(100%-var(--inset)*2)] max-w-[1020px]",
        "m-auto h-[fit-content]",
        "outline-none rounded-2xl overflow-x-hidden",
        "ring-[0.5px] ring-black/[0.08]",
        "shadow-2xl",
        "origin-center transition-[opacity,transform] duration-200 ease-out",
        "scale-95 opacity-0",
        "data-[enter]:scale-100 data-[enter]:opacity-100",
        className,
      )}
      {...props}
    />
  );
}
