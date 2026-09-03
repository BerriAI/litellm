"use client";

import { ChevronDown } from "lucide-react";
import { useMemo, useState } from "react";

import { getAvailablePages } from "@/components/page_utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface PageVisibilitySettingsProps {
  enabledPagesInternalUsers: string[] | null | undefined;
  enabledPagesPropertyDescription?: string;
  isUpdating: boolean;
  onUpdate: (settings: { enabled_ui_pages_internal_users: string[] | null }) => void;
}

export default function PageVisibilitySettings({
  enabledPagesInternalUsers,
  enabledPagesPropertyDescription,
  isUpdating,
  onUpdate,
}: PageVisibilitySettingsProps) {
  const isPageVisibilitySet = enabledPagesInternalUsers !== null && enabledPagesInternalUsers !== undefined;
  const availablePages = useMemo(() => getAvailablePages(), []);
  const pagesByGroup = useMemo(() => {
    const grouped: Record<string, typeof availablePages> = {};
    availablePages.forEach((page) => {
      if (!grouped[page.group]) {
        grouped[page.group] = [];
      }
      grouped[page.group].push(page);
    });
    return grouped;
  }, [availablePages]);
  const [selectedPages, setSelectedPages] = useState<string[]>(enabledPagesInternalUsers || []);

  useMemo(() => {
    setSelectedPages(enabledPagesInternalUsers || []);
  }, [enabledPagesInternalUsers]);

  const togglePage = (page: string, checked: boolean) => {
    setSelectedPages((current) => (checked ? [...current, page] : current.filter((item) => item !== page)));
  };

  const handleSavePageVisibility = () => {
    onUpdate({ enabled_ui_pages_internal_users: selectedPages.length > 0 ? selectedPages : null });
  };

  const handleResetToDefault = () => {
    setSelectedPages([]);
    onUpdate({ enabled_ui_pages_internal_users: null });
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium text-foreground">Internal User Page Visibility</p>
          <Badge variant={isPageVisibilitySet ? "secondary" : "outline"}>
            {isPageVisibilitySet
              ? `${selectedPages.length} page${selectedPages.length !== 1 ? "s" : ""} selected`
              : "Not set (all pages visible)"}
          </Badge>
        </div>
        {enabledPagesPropertyDescription && (
          <p className="text-sm text-muted-foreground">{enabledPagesPropertyDescription}</p>
        )}
        <p className="text-xs italic text-muted-foreground">
          By default, all pages are visible to internal users. Select specific pages to restrict visibility.
        </p>
        <p className="text-xs text-primary">
          Note: Only pages accessible to internal user roles are shown here. Admin-only pages are excluded as they
          cannot be made visible to internal users regardless of this setting.
        </p>
      </div>

      <Collapsible className="rounded-lg border border-border">
        <CollapsibleTrigger className="group flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted">
          Configure Page Visibility
          <ChevronDown className="size-4 transition-transform group-data-[panel-open]:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="border-t border-border p-4">
          <div className="space-y-4">
            {Object.entries(pagesByGroup).map(([groupName, pages]) => (
              <fieldset key={groupName} className="space-y-2">
                <legend className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                  {groupName}
                </legend>
                <div className="ml-4 space-y-2">
                  {pages.map((page) => {
                    const checkboxId = `page-visibility-${page.page}`;
                    return (
                      <label key={page.page} htmlFor={checkboxId} className="flex cursor-pointer items-start gap-2">
                        <Checkbox
                          id={checkboxId}
                          checked={selectedPages.includes(page.page)}
                          onCheckedChange={(checked) => togglePage(page.page, checked === true)}
                        />
                        <span className="space-y-0.5">
                          <span className="block text-sm text-foreground">{page.label}</span>
                          <span className="block text-xs text-muted-foreground">{page.description}</span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              </fieldset>
            ))}

            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={handleSavePageVisibility} disabled={isUpdating}>
                Save Page Visibility Settings
              </Button>
              {isPageVisibilitySet && (
                <Button type="button" variant="outline" onClick={handleResetToDefault} disabled={isUpdating}>
                  Reset to Default (All Pages)
                </Button>
              )}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
