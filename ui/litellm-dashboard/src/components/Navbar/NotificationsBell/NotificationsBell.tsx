"use client";

import {
  HIDE_AUTO_ROUTER_ANNOUNCEMENT_KEY,
  useHideAutoRouterAnnouncement,
} from "@/app/(dashboard)/hooks/useHideAutoRouterAnnouncement";
import { emitLocalStorageChange, setLocalStorageItem } from "@/utils/localStorageUtils";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverDescription, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";
import { Bell } from "lucide-react";
import React, { useState } from "react";

export const AUTO_ROUTER_DOCS_URL = "https://docs.litellm.ai/docs/proxy/auto_routing";

export const NotificationsBell: React.FC = () => {
  const hidden = useHideAutoRouterAnnouncement();
  const hasUnread = !hidden;
  const [open, setOpen] = useState(false);

  const markDismissed = () => {
    setLocalStorageItem(HIDE_AUTO_ROUTER_ANNOUNCEMENT_KEY, "true");
    emitLocalStorageChange(HIDE_AUTO_ROUTER_ANNOUNCEMENT_KEY);
    setOpen(false);
  };

  const content = (
    <div className="max-w-[280px]">
      <PopoverTitle className="mt-0! mb-2!">LiteLLM Auto Router</PopoverTitle>
      <PopoverDescription className="mb-3! text-sm leading-snug">
        Route every request to the cheapest model that can handle it, no prompt changes needed.
      </PopoverDescription>
      <div className="flex flex-wrap items-center gap-2">
        <a
          className={cn(buttonVariants({ size: "sm" }))}
          href={AUTO_ROUTER_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Read the docs
        </a>
        {hasUnread ? (
          <Button variant="link" size="sm" className="px-1!" onClick={markDismissed}>
            Mark as read
          </Button>
        ) : null}
      </div>
    </div>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className="flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!"
        aria-label="Notifications"
      >
        <span className="relative inline-flex">
          <Bell className="size-4" aria-hidden />
          {hasUnread ? <Badge className="absolute -top-0.5 -right-1 size-1.5 p-0" aria-hidden /> : null}
        </span>
      </PopoverTrigger>
      <PopoverContent align="end">{content}</PopoverContent>
    </Popover>
  );
};
