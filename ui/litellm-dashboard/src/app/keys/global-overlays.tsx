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

setAutoFreeze(false);

function GlobalOverlays() {
  const { deleteVirtualKeyDialogProps } = useGlobalOverlaysContext();

  return (
    <Fragment>
      {deleteVirtualKeyDialogProps ? (
        <DeleteVirtualKeyDialog {...deleteVirtualKeyDialogProps} />
      ) : null}
    </Fragment>
  );
}

export function GlobalOverlaysProvider(props: PropsWithChildren) {
  const deleteVirtualKeyDialogStore = useDialogStore({
    setMounted(mounted) {
      if (mounted === false) {
        updateValue((currentValue) => {
          currentValue.deleteVirtualKeyDialogProps = null;
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
  });

  return (
    <globalOverlaysContext.Provider value={value}>
      {props.children}
      <GlobalOverlays />
    </globalOverlaysContext.Provider>
  );
}
