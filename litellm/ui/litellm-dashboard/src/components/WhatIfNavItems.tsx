import { FlaskConical, Ghost, ReceiptText, ScanLine, SearchX, ShieldCheck } from "lucide-react";
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
  {
    key: "limit-xray",
    page: "limit-xray",
    icon: <ScanLine {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.limitXray",
  },
  {
    key: "ghost-trace",
    page: "ghost-trace",
    icon: <Ghost {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.ghostTrace",
  },
  {
    key: "pr-receipts",
    page: "pr-receipts",
    icon: <ReceiptText {...ICON} />,
    roles: all_admin_roles,
    label: "shell.nav.items.prReceipts",
  },
];
