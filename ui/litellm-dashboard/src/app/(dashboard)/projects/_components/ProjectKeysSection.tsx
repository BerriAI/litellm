import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { PaginationState } from "@tanstack/react-table";
import { Card, Flex, Input } from "antd";
import { KeyIcon, SearchIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { ProjectKeysTable } from "./ProjectKeysTable";

interface ProjectKeysSectionProps {
  projectId: string;
}

const PAGE_SIZE = 5;

export function ProjectKeysSection({ projectId }: ProjectKeysSectionProps) {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: PAGE_SIZE });
  const [keyAlias, setKeyAlias] = useState<string>("");

  const { data, isLoading } = useKeys(pagination.pageIndex + 1, pagination.pageSize, {
    projectID: projectId,
    selectedKeyAlias: keyAlias || null,
  });

  useEffect(() => {
    setPagination((current) => ({ ...current, pageIndex: 0 }));
  }, [keyAlias]);

  const keys = data?.keys ?? [];
  const totalCount = data?.total_count ?? 0;

  return (
    <Card
      title={
        <Flex align="center" gap={8}>
          <KeyIcon size={16} />
          Keys
        </Flex>
      }
      style={{ height: "100%" }}
    >
      <Flex justify="flex-start" align="center" style={{ marginBottom: 12 }}>
        <Input
          prefix={<SearchIcon size={14} />}
          placeholder="Filter by key name..."
          style={{ maxWidth: 220 }}
          value={keyAlias}
          onChange={(e) => setKeyAlias(e.target.value)}
          allowClear
          size="small"
        />
      </Flex>
      <ProjectKeysTable
        keys={keys}
        totalCount={totalCount}
        isLoading={isLoading}
        pagination={pagination}
        onPaginationChange={setPagination}
      />
    </Card>
  );
}
