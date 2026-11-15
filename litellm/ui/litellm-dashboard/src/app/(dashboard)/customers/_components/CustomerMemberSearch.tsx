"use client";

import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
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
  const submit = useDebouncedCallback(
    (next: string) => {
      submitted.current = next;
      onSearch(next);
    },
    { wait: DEBOUNCE_WAIT_MS },
  );
  useEffect(() => {
    if (submitted.current === value) return;
    submitted.current = value;
    setDraft(value);
  }, [value]);
  return (
    <div className="relative mb-4 w-full max-w-xs">
      <Search aria-hidden="true" className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        type="search"
        className="h-8 pl-9 text-xs"
        value={draft}
        onChange={(event) => {
          setDraft(event.target.value);
          submit(event.target.value.trim());
        }}
        aria-label={t("customerPortal.hierarchy.searchMembers")}
        placeholder={t("customerPortal.hierarchy.searchMembers")}
      />
    </div>
  );
}
