import type { RoleMappings as RoleMappingsType } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/shared/DataTable";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Users } from "lucide-react";
import { defaultRoleDisplayNames } from "./constants";

const inlineCodeClass = "rounded-sm border border-border bg-muted px-1 py-0.5 font-mono text-xs";

interface RoleMappingRow {
  role: string;
  groups: string[];
}

export default function RoleMappings({ roleMappings }: { roleMappings: RoleMappingsType | undefined }) {
  if (!roleMappings) {
    return null;
  }

  const roleMappingsColumns: ColumnDef<RoleMappingRow>[] = [
    {
      id: "role",
      accessorKey: "role",
      header: "Role",
      cell: ({ row }) => <strong className="font-semibold">{defaultRoleDisplayNames[row.original.role]}</strong>,
    },
    {
      id: "groups",
      accessorKey: "groups",
      header: "Mapped Groups",
      cell: ({ row }) =>
        row.original.groups.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {row.original.groups.map((group, index) => (
              <StatusBadge key={index} tone="info" label={group} />
            ))}
          </div>
        ) : (
          <span className="text-muted-foreground italic">No groups mapped</span>
        ),
    },
  ];
  return (
    <Card>
      <CardContent>
        <div className="flex items-center gap-3">
          <Users className="w-6 h-6 text-muted-foreground mb-2" />
          <h3 className="mb-2 text-2xl font-semibold text-foreground">Role Mappings</h3>
        </div>
        <div className="space-y-8">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="mb-2 text-base font-semibold text-foreground">Group Claim</h5>
              <div>
                <code className={inlineCodeClass}>{roleMappings.group_claim}</code>
              </div>
            </div>
            <div>
              <h5 className="mb-2 text-base font-semibold text-foreground">Default Role</h5>
              <div>
                <strong className="font-semibold">{defaultRoleDisplayNames[roleMappings.default_role]}</strong>
              </div>
            </div>
          </div>
          <Separator className="my-6" />
          <DataTable
            columns={roleMappingsColumns}
            data={Object.entries(roleMappings.roles).map(([role, groups]) => ({
              role,
              groups,
            }))}
            getRowId={(row) => row.role}
            size="compact"
          />
        </div>
      </CardContent>
    </Card>
  );
}
