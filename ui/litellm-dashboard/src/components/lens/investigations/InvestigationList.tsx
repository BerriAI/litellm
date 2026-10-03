import { ListRow } from "@/components/shared/ListRow";
import { useState } from "react";
import { ChevronRight, Search } from "lucide-react";
import { Input } from "@/components/ui/input";

import { lensStatus } from "../model/status";
import { runTime, scopeLabel } from "../model/format";
import { type Lens } from "../model/types";

export function InvestigationList({
  lenses,
  connected,
  onSelect,
}: {
  lenses: Lens[];
  connected: boolean;
  onSelect: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const shown = lenses.filter((lens) =>
    `${lens.settings.name} ${scopeLabel(lens.settings)}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <section aria-label="Saved investigations" className="space-y-4">
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
        <Input
          aria-label="Search investigations"
          placeholder="Search investigations"
          className="pl-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <div className="divide-y border-y">
        {shown.map((lens) => (
          <ListRow
            key={lens.id}
            onClick={() => onSelect(lens.id)}
            className="group grid w-full grid-cols-[minmax(0,1fr)_16px] items-center gap-x-4 gap-y-2 py-4 text-left hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring sm:grid-cols-[minmax(0,1fr)_auto_16px]"
          >
            <div className="col-start-1 row-start-1 min-w-0">
              <p className="text-sm font-medium">{lens.settings.name}</p>
              <p className="mt-1 truncate text-xs text-muted-foreground">{scopeLabel(lens.settings)}</p>
            </div>
            <div className="col-start-1 row-start-2 text-xs sm:col-start-2 sm:row-start-1 sm:text-right">
              <p
                data-state={lens.jobs[0]?.status === "failed" ? "failed" : "other"}
                className="data-[state=failed]:text-destructive data-[state=other]:text-muted-foreground"
              >
                {lensStatus(lens, connected)}
              </p>
              <p className="mt-1 text-muted-foreground">
                {lens.jobs[0] ? runTime(lens.jobs[0].created_at) : "Not run yet"}
              </p>
            </div>
            <ChevronRight className="col-start-2 row-start-1 size-4 text-muted-foreground sm:col-start-3" />
          </ListRow>
        ))}
        {!shown.length && <p className="py-8 text-sm text-muted-foreground">No investigations match your search.</p>}
      </div>
    </section>
  );
}
