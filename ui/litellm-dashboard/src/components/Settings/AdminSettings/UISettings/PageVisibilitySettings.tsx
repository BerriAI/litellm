"use client";

import { getAvailablePages } from "@/components/page_utils";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useMemo, useState } from "react";

interface PageVisibilitySettingsProps {
  enabledPagesInternalUsers: string[] | null | undefined;
  enabledPagesPropertyDescription?: string;
  isUpdating: boolean;
  onUpdate: (settings: {
    enabled_ui_pages_internal_users: string[] | null;
  }) => void;
}

export default function PageVisibilitySettings({
  enabledPagesInternalUsers,
  enabledPagesPropertyDescription,
  isUpdating,
  onUpdate,
}: PageVisibilitySettingsProps) {
  const isPageVisibilitySet =
    enabledPagesInternalUsers !== null &&
    enabledPagesInternalUsers !== undefined;

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

  const [selectedPages, setSelectedPages] = useState<string[]>(
    enabledPagesInternalUsers || [],
  );

  useMemo(() => {
    if (enabledPagesInternalUsers) {
      setSelectedPages(enabledPagesInternalUsers);
    } else {
      setSelectedPages([]);
    }
  }, [enabledPagesInternalUsers]);

  const togglePage = (page: string, checked: boolean) => {
    if (checked) {
      setSelectedPages((prev) => [...prev, page]);
    } else {
      setSelectedPages((prev) => prev.filter((p) => p !== page));
    }
  };

  const handleSavePageVisibility = () => {
    onUpdate({
      enabled_ui_pages_internal_users:
        selectedPages.length > 0 ? selectedPages : null,
    });
  };

  const handleResetToDefault = () => {
    setSelectedPages([]);
    onUpdate({ enabled_ui_pages_internal_users: null });
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="font-bold">Internal User Page Visibility</span>
          {!isPageVisibilitySet && (
            <Badge variant="secondary">Not set (all pages visible)</Badge>
          )}
          {isPageVisibilitySet && (
            <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              {selectedPages.length} page{selectedPages.length !== 1 ? "s" : ""}{" "}
              selected
            </Badge>
          )}
        </div>
        {enabledPagesPropertyDescription && (
          <p className="text-muted-foreground text-sm">
            {enabledPagesPropertyDescription}
          </p>
        )}
        <p className="text-muted-foreground text-xs italic">
          By default, all pages are visible to internal users. Select specific
          pages to restrict visibility.
        </p>
        <p className="text-purple-600 dark:text-purple-400 text-xs">
          Note: Only pages accessible to internal user roles are shown here.
          Admin-only pages are excluded as they cannot be made visible to
          internal users regardless of this setting.
        </p>
      </div>

      <Accordion type="single" collapsible>
        <AccordionItem value="page-visibility">
          <AccordionTrigger>Configure Page Visibility</AccordionTrigger>
          <AccordionContent>
            <div className="flex flex-col gap-4 w-full">
              {Object.entries(pagesByGroup).map(([groupName, pages]) => (
                <div key={groupName}>
                  <div className="font-bold text-[11px] text-muted-foreground tracking-wider block mb-2">
                    {groupName}
                  </div>
                  <div className="ml-4 flex flex-col gap-2 w-full">
                    {pages.map((page) => (
                      <label
                        key={page.page}
                        className="flex items-start gap-2 cursor-pointer"
                      >
                        <Checkbox
                          checked={selectedPages.includes(page.page)}
                          onCheckedChange={(c) =>
                            togglePage(page.page, c === true)
                          }
                          className="mt-1"
                        />
                        <div className="flex flex-col">
                          <span>{page.label}</span>
                          <span className="text-muted-foreground text-xs">
                            {page.description}
                          </span>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              ))}

              <div className="flex gap-2">
                <Button
                  onClick={handleSavePageVisibility}
                  disabled={isUpdating}
                >
                  {isUpdating
                    ? "Saving…"
                    : "Save Page Visibility Settings"}
                </Button>
                {isPageVisibilitySet && (
                  <Button
                    variant="outline"
                    onClick={handleResetToDefault}
                    disabled={isUpdating}
                  >
                    Reset to Default (All Pages)
                  </Button>
                )}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
