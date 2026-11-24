import { Disc3, Split, Unplug } from "lucide-react";
import { all_admin_roles } from "../utils/roles";
import type { MenuItem } from "./LeftNav";

const ICON = { strokeWidth: 1.75 } as const;

export const whatIfNavItems: MenuItem[] = [
  {
    key: "tape-check",
    page: "tape-check",
    icon: <Disc3 {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.tapeCheck",
  },
  {
    key: "lost-fields",
    page: "lost-fields",
    icon: <Unplug {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.lostFields",
  },
  {
    key: "key-split",
    page: "key-split",
    icon: <Split {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.keySplit",
  },
];
