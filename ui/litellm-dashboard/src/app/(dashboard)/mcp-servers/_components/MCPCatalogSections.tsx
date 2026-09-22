import React from "react";
import { FileText, MessageSquareText, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { MCPPrompt, MCPResource, MCPResourceTemplate } from "@/components/mcp_tools/types";

interface CatalogSectionProps {
  title: string;
  icon: LucideIcon;
  count: number;
  isLoading: boolean;
  errorMessage?: string | null;
  emptyText: string;
  children: React.ReactNode;
}

const CatalogSection = ({
  title,
  icon: Icon,
  count,
  isLoading,
  errorMessage,
  emptyText,
  children,
}: CatalogSectionProps) => {
  const isSettled = !isLoading && !errorMessage;
  return (
    <section aria-label={title} className="mt-4 flex flex-col">
      <p className="mb-3 flex items-center text-sm font-medium">
        <Icon className="mr-2 size-4" /> {title}
        {count > 0 && (
          <Badge variant="secondary" className="ml-2">
            {count}
          </Badge>
        )}
      </p>
      {isLoading && (
        <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-6">
          <UiLoadingSpinner className="mb-2 size-5 text-muted-foreground" />
          <p className="text-xs font-medium">Loading {title.toLowerCase()}...</p>
        </div>
      )}
      {!isLoading && errorMessage && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive">
          <p className="font-medium">Error: {errorMessage}</p>
        </div>
      )}
      {isSettled && count === 0 && (
        <div className="rounded-lg border border-border bg-card p-3 text-center">
          <p className="text-xs text-muted-foreground">{emptyText}</p>
        </div>
      )}
      {isSettled && count > 0 && <div className="max-h-60 min-h-0 space-y-2 overflow-y-auto">{children}</div>}
    </section>
  );
};

interface MCPPromptsSectionProps {
  prompts: MCPPrompt[];
  isLoading: boolean;
  errorMessage?: string | null;
}

export const MCPPromptsSection = ({ prompts, isLoading, errorMessage }: MCPPromptsSectionProps) => (
  <CatalogSection
    title="Prompts"
    icon={MessageSquareText}
    count={prompts.length}
    isLoading={isLoading}
    errorMessage={errorMessage}
    emptyText="No prompts found for this server"
  >
    {prompts.map((prompt) => (
      <div key={prompt.name} className="rounded-lg border border-border bg-card p-3">
        <h4 className="truncate font-mono text-xs font-medium">{prompt.name}</h4>
        {prompt.description && (
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{prompt.description}</p>
        )}
        {prompt.arguments && prompt.arguments.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {prompt.arguments.map((argument) => (
              <Badge
                key={argument.name}
                variant={argument.required ? "default" : "outline"}
                title={argument.description ?? (argument.required ? "required" : "optional")}
                className="font-mono text-[10px]"
              >
                {argument.name}
              </Badge>
            ))}
          </div>
        )}
      </div>
    ))}
  </CatalogSection>
);

interface MCPResourcesSectionProps {
  resources: MCPResource[];
  resourceTemplates: MCPResourceTemplate[];
  isLoading: boolean;
  errorMessage?: string | null;
}

export const MCPResourcesSection = ({
  resources,
  resourceTemplates,
  isLoading,
  errorMessage,
}: MCPResourcesSectionProps) => (
  <CatalogSection
    title="Resources"
    icon={FileText}
    count={resources.length + resourceTemplates.length}
    isLoading={isLoading}
    errorMessage={errorMessage}
    emptyText="No resources found for this server"
  >
    {resources.map((resource) => (
      <div key={resource.uri} className="rounded-lg border border-border bg-card p-3">
        <div className="flex items-start justify-between gap-2">
          <h4 className="truncate text-xs font-medium">{resource.title ?? resource.name}</h4>
          {resource.mimeType && (
            <Badge variant="outline" className="shrink-0 font-mono text-[10px]">
              {resource.mimeType}
            </Badge>
          )}
        </div>
        <p className="truncate font-mono text-xs text-muted-foreground" title={resource.uri}>
          {resource.uri}
        </p>
        {resource.description && (
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{resource.description}</p>
        )}
      </div>
    ))}
    {resourceTemplates.map((template) => (
      <div key={template.uriTemplate} className="rounded-lg border border-dashed border-border bg-card p-3">
        <div className="flex items-start justify-between gap-2">
          <h4 className="truncate text-xs font-medium">{template.title ?? template.name}</h4>
          <Badge variant="secondary" className="shrink-0 text-[10px]">
            template
          </Badge>
        </div>
        <p className="truncate font-mono text-xs text-muted-foreground" title={template.uriTemplate}>
          {template.uriTemplate}
        </p>
        {template.description && (
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{template.description}</p>
        )}
      </div>
    ))}
  </CatalogSection>
);
