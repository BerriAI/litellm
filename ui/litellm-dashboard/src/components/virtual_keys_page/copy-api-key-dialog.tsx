import { Button, DialogStore, useDialogContext } from "@ariakit/react";
import { UiDialog } from "./ui-dialog";
import { cx } from "@/lib/cva.config";
import { message } from "antd";

type ContentProps = {
  apiKey: string;
};

function Content(props: ContentProps) {
  const store = useDialogContext();

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="min-h-0 overflow-y-auto px-8 py-8">
        <div className="flex flex-col gap-4 mb-4">
          <h1 className="text-[18px] tracking-tight text-neutral-900">
            Save your Key
          </h1>

          <p className="text-[15px]/[1.6] text-neutral-500 tracking-tight">
            Please save this secret key somewhere safe and accessible.
            <br />
            For security reasons,{" "}
            <span className="font-medium text-neutral-900">
              you will not be able to view it again
            </span>{" "}
            through your LiteLLM account. If you lose this secret key, you will
            need to generate a new one.
          </p>
        </div>

        <div
          className={cx(
            "flex items-center justify-between",
            "bg-white pl-4 py-2 pr-2 rounded-md",
            "ring-[0.5px] ring-black/[0.08]",
            "shadow-md shadow-black/[0.05]",
          )}
        >
          <span className="text-[12px] text-neutral-800 font-medium">
            API Key:
          </span>

          <span
            className={cx(
              "text-[16px] text-neutral-800 font-mono font-medium tracking-tight",
              "py-1 px-2 rounded bg-neutral-100",
            )}
          >
            {props.apiKey}
          </span>
        </div>
      </div>

      <div
        className={cx(
          "flex items-center justify-between mt-auto bg-neutral-100",
          "px-6 h-[64px] shrink-0",
        )}
      >
        <Button
          onClick={store?.hide}
          type="button"
          className={cx(
            "h-[36px] bg-white px-3 rounded-md",
            "flex items-center gap-1",
            "text-[12px] font-medium tracking-tight",
            "bg-white",
          )}
        >
          <span className="text-neutral-900">Close</span>
        </Button>

        <Button
          autoFocus
          type="button"
          onClick={() => {
            navigator.clipboard
              .writeText(props.apiKey)
              .then(() => {
                message.success("API Key copied to clipboard");
              })
              .catch(() => {
                message.error("Failed to copy API Key. Please copy manually!");
              });
          }}
          className={cx(
            "h-[36px] px-3 rounded-md",
            "flex items-center gap-1",
            "text-[12px] font-medium tracking-tight",
            "indigo-button-3d",
          )}
        >
          <span className="text-white">Copy API Key</span>
        </Button>
      </div>
    </div>
  );
}

type CopyApiKeyDialogProps = {
  store?: DialogStore;
  apiKey: string;
};

export function CopyApiKeyDialog(props: CopyApiKeyDialogProps) {
  return (
    <UiDialog store={props.store} className={cx("max-w-[580px]")}>
      <Content apiKey={props.apiKey} />
    </UiDialog>
  );
}
