import { KeyResponse } from "@/components/key_team_helpers/key_list";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Loader2 } from "lucide-react";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";

interface ProjectKeysTableProps {
  keys: KeyResponse[];
  loading?: boolean;
}

export function ProjectKeysTable({ keys, loading }: ProjectKeysTableProps) {
  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Key Name</TableHead>
            <TableHead>Owner</TableHead>
            <TableHead>Created</TableHead>
            <TableHead>Last Active</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={4} className="text-center py-6">
                <Loader2 className="h-4 w-4 animate-spin mx-auto text-muted-foreground" />
              </TableCell>
            </TableRow>
          ) : keys.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={4}
                className="text-center py-6 text-muted-foreground"
              >
                No keys found
              </TableCell>
            </TableRow>
          ) : (
            keys.map((k) => {
              const alias = k.key_alias || "—";
              const email = k.user?.user_email ?? k.user_id ?? null;
              return (
                <TableRow key={k.token}>
                  <TableCell>{alias}</TableCell>
                  <TableCell>
                    {email ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="inline-flex">
                              <DefaultProxyAdminTag userId={email} />
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>{email}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell>
                    {k.created_at
                      ? new Date(k.created_at).toLocaleDateString()
                      : "—"}
                  </TableCell>
                  <TableCell>
                    {k.last_active
                      ? new Date(k.last_active).toLocaleDateString()
                      : "Never"}
                  </TableCell>
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>
    </div>
  );
}
