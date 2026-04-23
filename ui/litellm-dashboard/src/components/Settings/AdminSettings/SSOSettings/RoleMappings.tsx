import type { RoleMappings as RoleMappingsType } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Users } from "lucide-react";
import { defaultRoleDisplayNames } from "./constants";

export default function RoleMappings({
  roleMappings,
}: {
  roleMappings: RoleMappingsType | undefined;
}) {
  if (!roleMappings) return null;

  return (
    <Card className="p-6">
      <div className="flex items-center gap-3">
        <Users className="w-6 h-6 text-muted-foreground mb-2" />
        <h3 className="text-lg font-semibold">Role Mappings</h3>
      </div>
      <div className="space-y-8">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h5 className="text-base font-semibold">Group Claim</h5>
            <div>
              <code className="bg-muted px-1.5 py-0.5 rounded text-sm">
                {roleMappings.group_claim}
              </code>
            </div>
          </div>
          <div>
            <h5 className="text-base font-semibold">Default Role</h5>
            <div>
              <span className="font-semibold">
                {defaultRoleDisplayNames[roleMappings.default_role]}
              </span>
            </div>
          </div>
        </div>
        <Separator />
        <Table className="border border-border rounded-md w-full">
          <TableHeader>
            <TableRow>
              <TableHead>Role</TableHead>
              <TableHead>Mapped Groups</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(roleMappings.roles).map(([role, groups]) => (
              <TableRow key={role}>
                <TableCell>
                  <span className="font-semibold">
                    {defaultRoleDisplayNames[role]}
                  </span>
                </TableCell>
                <TableCell>
                  {groups.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {groups.map((group, index) => (
                        <Badge key={index} variant="default">
                          {group}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground italic">
                      No groups mapped
                    </span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
