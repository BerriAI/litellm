import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  Key as KeyIcon,
  Search,
} from "lucide-react";
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

  useEffect(() => {
    setPage(1);
  }, [keyAlias]);

  const keys = data?.keys ?? [];
  const totalCount = data?.total_count ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <KeyIcon size={16} />
          Keys
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between items-center mb-3 gap-2">
          <div className="relative max-w-[220px]">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Filter by key name..."
              className="pl-7 h-8"
              value={keyAlias}
              onChange={(e) => setKeyAlias(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-1 text-sm">
            <span className="text-muted-foreground">{totalCount} keys</span>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-muted-foreground">
              {page}/{totalPages}
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <ProjectKeysTable keys={keys} loading={isLoading} />
      </CardContent>
    </Card>
  );
}
