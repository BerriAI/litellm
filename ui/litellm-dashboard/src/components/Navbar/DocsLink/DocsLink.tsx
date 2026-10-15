import { getBreadcrumb } from "@/components/leftnav";
import { NAV_PRODUCT_LINK_CLASS } from "@/components/Navbar/navProductLinkClass";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { pageDocsFor } from "@/lib/pageDocs";
import { ChevronDown } from "lucide-react";
import { usePathname } from "next/navigation";
import React from "react";

export const DOCS_URL = "https://docs.litellm.ai/docs/";

const ChevronWidthSpacer: React.FC = () => (
  <ChevronDown className="pointer-events-none size-2.5 opacity-0" aria-hidden />
);

export const DocsLink: React.FC = () => {
  const pathname = usePathname();
  const topics = pageDocsFor(pathname);

  if (topics.length === 0) {
    return (
      <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className={NAV_PRODUCT_LINK_CLASS}>
        Docs
        <ChevronWidthSpacer />
      </a>
    );
  }

  const { title } = getBreadcrumb(pathname);

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger
        openOnHover
        closeDelay={100}
        render={<Button variant="ghost" className={`${NAV_PRODUCT_LINK_CLASS} border-0!`} />}
      >
        Docs
        <ChevronDown className="size-2.5 text-muted-foreground" aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" side="bottom" className="w-auto">
        <DropdownMenuGroup>
          <DropdownMenuLabel>{title} docs</DropdownMenuLabel>
        </DropdownMenuGroup>
        {topics.map((topic) => (
          <DropdownMenuItem key={topic.href}>
            <a href={topic.href} target="_blank" rel="noopener noreferrer" style={{ display: "block", width: 260 }}>
              {topic.title}
            </a>
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem>
          <a href={DOCS_URL} target="_blank" rel="noopener noreferrer">
            All docs
          </a>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default DocsLink;
