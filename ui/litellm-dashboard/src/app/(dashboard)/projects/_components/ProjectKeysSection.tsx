import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { KeyIcon, SearchIcon, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { ProjectKeysTable } from "./ProjectKeysTable";
import { useProjectKeysTableState } from "./useProjectsUrlState";

interface ProjectKeysSectionProps {
  projectId: string;
}

export function ProjectKeysSection({ projectId }: ProjectKeysSectionProps) {
  const {
    search: keyAlias,
    setSearch: setKeyAlias,
    pagination,
    onPaginationChange: setPagination,
  } = useProjectKeysTableState();

  const { data, isLoading, isError } = useKeys(pagination.pageIndex + 1, pagination.pageSize, {
    projectID: projectId,
    selectedKeyAlias: keyAlias || null,
  });

  const keys = data?.keys ?? [];
  const totalCount = data?.total_count ?? 0;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <KeyIcon className="size-4" />
          Keys
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-3 flex items-center">
          <InputGroup className="max-w-[220px]">
            <InputGroupAddon>
              <SearchIcon className="size-3.5 text-muted-foreground" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder="Filter by key name..."
              value={keyAlias}
              onChange={(e) => setKeyAlias(e.target.value)}
            />
            {keyAlias && (
              <InputGroupAddon align="inline-end">
                <InputGroupButton size="icon-xs" aria-label="Clear key filter" onClick={() => setKeyAlias("")}>
                  <X />
                </InputGroupButton>
              </InputGroupAddon>
            )}
          </InputGroup>
        </div>
        <ProjectKeysTable
          keys={keys}
          totalCount={totalCount}
          isLoading={isLoading}
          isError={isError}
          pagination={pagination}
          onPaginationChange={setPagination}
        />
      </CardContent>
    </Card>
  );
}
