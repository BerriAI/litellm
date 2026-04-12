import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { LoadingOutlined } from "@ant-design/icons";
import { Card, Flex, Input, Pagination, Spin } from "antd";
import { KeyIcon, SearchIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { ProjectKeysTable } from "./ProjectKeysTable";

interface ProjectKeysSectionProps {
  projectId: string;
}

const PAGE_SIZE = 5;

export function ProjectKeysSection({ projectId }: ProjectKeysSectionProps) {
  const [page, setPage] = useState(1);
  const [keyAlias, setKeyAlias] = useState<string>("");

  const { data, isLoading } = useKeys(page, PAGE_SIZE, {
    projectID: projectId,
    selectedKeyAlias: keyAlias || null,
  });

  // Reset to page 1 when filter changes
  useEffect(() => {
    setPage(1);
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
      <Flex justify="space-between" align="center" style={{ marginBottom: 12 }}>
        <Input
          prefix={<SearchIcon size={14} />}
          placeholder="Filter by key name..."
          style={{ maxWidth: 220 }}
          value={keyAlias}
          onChange={(e) => setKeyAlias(e.target.value)}
          allowClear
          size="small"
        />
        <Pagination
          current={page}
          total={totalCount}
          pageSize={PAGE_SIZE}
          onChange={setPage}
          size="small"
          showSizeChanger={false}
          showTotal={(total) => `${total} keys`}
        />
      </Flex>
      <ProjectKeysTable
        keys={keys}
        loading={isLoading ? { indicator: <Spin indicator={<LoadingOutlined spin />} /> } : false}
      />
    </Card>
  );
}
