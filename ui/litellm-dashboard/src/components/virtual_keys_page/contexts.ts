import { createContext, useContext } from "react";
import { DeleteVirtualKeyDialogProps } from "./delete-virtual-key-dialog";
import { EditVirtualKeyDialogProps } from "./edit-virtual-key-dialog";

export type AuthContext = {
  auth_header_name: string;
  disabled_non_admin_personal_key_creation: boolean;
  key: string;
  login_method: string;
  premium_user: boolean;
  server_root_path: string;
  user_email: string | null;
  user_id: string;
  user_role: string;
};

export const authContext = createContext<AuthContext | undefined>(undefined);

export function useAuthContext() {
  const context = useContext(authContext);
  if (context === undefined)
    throw new Error(
      "useAuthContext must be used with <authContext.Provider />",
    );

  return context;
}

export type GlobalOverlaysContext = {
  deleteVirtualKeyDialogProps: DeleteVirtualKeyDialogProps | null;
  deleteVirtualKey: (props: DeleteVirtualKeyDialogProps) => void;

  editVirtualKeyDialogProps: EditVirtualKeyDialogProps | null;
  editVirtualKey: (props: EditVirtualKeyDialogProps) => void;
};

export const globalOverlaysContext = createContext<
  GlobalOverlaysContext | undefined
>(undefined);

export function useGlobalOverlaysContext() {
  const context = useContext(globalOverlaysContext);
  if (context === undefined) {
    throw new Error(
      "useGlobalOverlaysContext must be used with <globalOverlaysContext.Provider />",
    );
  }

  return context;
}
