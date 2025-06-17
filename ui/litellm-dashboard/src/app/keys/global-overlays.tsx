import { useDialogStore } from "@ariakit/react";
import { Fragment, PropsWithChildren } from "react";
import { setAutoFreeze } from "immer";
import { useImmer } from "use-immer";
import {
  GlobalOverlaysContext,
  globalOverlaysContext,
  useGlobalOverlaysContext,
} from "./contexts";
import { DeleteVirtualKeyDialog } from "./delete-virtual-key-dialog";
import { EditVirtualKeyDialog } from "./edit-virtual-key-dialog";
import { useQueryClient } from "@tanstack/react-query";

setAutoFreeze(false);

function GlobalOverlays() {
  const { deleteVirtualKeyDialogProps, editVirtualKeyDialogProps } =
    useGlobalOverlaysContext();

  return (
    <Fragment>
      {deleteVirtualKeyDialogProps ? (
        <DeleteVirtualKeyDialog {...deleteVirtualKeyDialogProps} />
      ) : null}

      {editVirtualKeyDialogProps ? (
        <EditVirtualKeyDialog {...editVirtualKeyDialogProps} />
      ) : null}
    </Fragment>
  );
}

export function GlobalOverlaysProvider(props: PropsWithChildren) {
  const queryClient = useQueryClient();

  const deleteVirtualKeyDialogStore = useDialogStore({
    setMounted(mounted) {
      if (mounted === false) {
        updateValue((currentValue) => {
          currentValue.deleteVirtualKeyDialogProps = null;
        });
      }
    },
  });

  const editVirtualKeyDialogStore = useDialogStore({
    setOpen(open) {
      // Refetch virtual keys whenever this dialog closes
      if (open === false) {
        queryClient.invalidateQueries({ queryKey: ["keys"] });
      }
    },
    setMounted(mounted) {
      if (mounted === false) {
        updateValue((currentValue) => {
          currentValue.editVirtualKeyDialogProps = null;
        });
      }
    },
  });

  const [value, updateValue] = useImmer<GlobalOverlaysContext>({
    deleteVirtualKeyDialogProps: null,
    deleteVirtualKey: (props) => {
      updateValue((currentValue) => {
        currentValue.deleteVirtualKeyDialogProps = {
          store: deleteVirtualKeyDialogStore,
          ...props,
        };
      });
      deleteVirtualKeyDialogStore.show();
    },
    editVirtualKeyDialogProps: null,
    editVirtualKey: (props) => {
      updateValue((currentValue) => {
        currentValue.editVirtualKeyDialogProps = {
          store: editVirtualKeyDialogStore,
          ...props,
        };
      });
      editVirtualKeyDialogStore.show();
    },
  });

  return (
    <globalOverlaysContext.Provider value={value}>
      {props.children}
      <GlobalOverlays />
    </globalOverlaysContext.Provider>
  );
}
