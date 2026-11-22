import { FlaskConical, SearchX, ShieldCheck } from "lucide-react";
import { all_admin_roles } from "../utils/roles";
import type { MenuItem } from "./LeftNav";

const ICON = { strokeWidth: 1.75 } as const;

export const whatIfNavItems: MenuItem[] = [
  {
    key: "control-canaries",
    page: "control-canaries",
    icon: <ShieldCheck {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.controlCanaries",
  },
  {
    key: "cache-miss",
    page: "cache-miss",
    icon: <SearchX {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.cacheMiss",
  },
  {
    key: "plan-lab",
    page: "plan-lab",
    icon: <FlaskConical {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.planLab",
  },
];
