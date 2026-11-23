"use client";

import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { useDebouncer } from "@tanstack/react-pacer/debouncer";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

export default function CustomerMemberSearch({
  value,
  onSearch,
}: {
  value: string;
  onSearch: (value: string) => void;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState(value);
  const submitted = useRef(value);
  const debouncer = useDebouncer(
    (next: string) => {
      submitted.current = next;
      onSearch(next);
    },
    { wait: DEBOUNCE_WAIT_MS },
  );
  useEffect(() => {
    if (submitted.current === value) return;
    debouncer.cancel();
    submitted.current = value;
    setDraft(value);
  }, [debouncer, value]);
  return (
    <div className="relative w-full max-w-xs">
      <Search aria-hidden="true" className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        type="search"
        className="pl-9"
        value={draft}
        onChange={(event) => {
          setDraft(event.target.value);
          debouncer.maybeExecute(event.target.value.trim());
        }}
        aria-label={t("customerPortal.hierarchy.searchMembers")}
        placeholder={t("customerPortal.hierarchy.searchMembers")}
      />
    </div>
  );
}
