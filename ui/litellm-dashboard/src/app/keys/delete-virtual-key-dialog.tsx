import { Button, DialogStore, useDialogContext } from "@ariakit/react";
import { UiDialog } from "./ui-dialog";
import { cx } from "@/lib/cva.config";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { keyDeleteCall } from "@/components/networking";
import { useAuthContext } from "./contexts";
import { message } from "antd";
import { parseErrorMessage } from "@/components/shared/errorUtils";

type ContentProps = {
  virtualKey: KeyResponse;
};

function Content(props: ContentProps) {
  const store = useDialogContext();
  const authCtx = useAuthContext();
  const queryClient = useQueryClient();

  const deleteVirtualKeyMutation = useMutation({
    mutationFn: () => {
      return keyDeleteCall(authCtx.key, props.virtualKey.token);
    },
    onSuccess() {
      // Update local cache
      queryClient.setQueriesData(
        { queryKey: ["keys"] },
        (oldData: KeyResponse[] | undefined) => {
          if (oldData === undefined) return undefined;
          return oldData.filter((key) => key.token !== props.virtualKey.token);
        },
      );
      queryClient.invalidateQueries({ queryKey: ["keys"] });

      message.success("Key deleted successfully");
      store?.hide();
    },
    onError(error) {
      message.error(parseErrorMessage(error));
    },
  });

  const busy = deleteVirtualKeyMutation.isPending;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="min-h-0 overflow-y-auto px-8 py-8">
        <div className="flex flex-col gap-4 mb-4">
          <h1 className="text-[18px] tracking-tight text-neutral-900">
            Delete Virtual Key?
          </h1>

          <p className="text-[15px]/[1.6] text-neutral-500 tracking-tight">
            This will immediately revoke API access for any applications using
            this key.{" "}
            <span className="font-medium text-neutral-900">
              The deletion cannot be undone,
            </span>{" "}
            and you will need to create a new key to restore access to affected
            services.
          </p>
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
          <span className="text-neutral-900">Cancel</span>
        </Button>

        <Button
          autoFocus
          disabled={busy}
          type="button"
          onClick={() => {
            deleteVirtualKeyMutation.mutate();
          }}
          className={cx(
            "h-[36px] px-3 rounded-md",
            "flex items-center gap-2",
            "text-[12px] font-medium tracking-tight",
            "bg-red-500 disabled:opacity-50",
          )}
        >
          <span className="text-white">Yes, Delete Permanently</span>
          {busy ? <UiLoadingSpinner className="size-3 text-white" /> : null}
        </Button>
      </div>
    </div>
  );
}

export type DeleteVirtualKeyDialogProps = {
  store?: DialogStore;
  virtualKey: KeyResponse;
};

export function DeleteVirtualKeyDialog(props: DeleteVirtualKeyDialogProps) {
  return (
    <UiDialog store={props.store} className="max-w-[520px]">
      <Content virtualKey={props.virtualKey} />
    </UiDialog>
  );
}
