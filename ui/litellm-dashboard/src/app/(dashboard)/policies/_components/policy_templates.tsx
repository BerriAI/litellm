import React, { useState, useEffect, useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import MessageManager from "@/components/molecules/message_manager";
import { ShieldCheck, ShieldAlert, FlaskConical, CircleDollarSign, CheckCircle2 } from "lucide-react";
import { getPolicyTemplates } from "@/components/networking";

interface PolicyTemplateCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  iconColor: string;
  iconBg: string;
  guardrails: string[];
  tags: string[];
  inherits?: string;
  complexity: "Low" | "Medium" | "High";
  onUseTemplate: () => void;
}

const PolicyTemplateCard: React.FC<PolicyTemplateCardProps> = ({
  title,
  description,
  icon: Icon,
  iconColor,
  iconBg,
  guardrails,
  tags,
  inherits,
  complexity,
  onUseTemplate,
}) => {
  return (
    <Card className="h-full transition-shadow hover:shadow-md">
      <CardContent className="flex h-full flex-col">
        <div className="mb-4 flex items-start justify-between">
          <div className={`rounded-lg p-2 ${iconBg}`}>
            <Icon className={`size-6 ${iconColor}`} />
          </div>
          <Badge variant="outline">{complexity} Complexity</Badge>
        </div>

        <h3 className="mb-2 text-base font-semibold">{title}</h3>
        <p className="mb-4 grow text-sm text-muted-foreground">{description}</p>

        {tags.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-1.5">
            {tags.map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        )}

        {inherits && (
          <div className="mb-4 text-xs">
            <span className="text-muted-foreground">Inherits from: </span>
            <span className="rounded-sm bg-muted px-2 py-0.5 font-medium">{inherits}</span>
          </div>
        )}

        <div className="mb-6">
          <span className="mb-2 block text-xs font-medium tracking-wider text-muted-foreground uppercase">
            Included Guardrails
          </span>
          <div className="flex flex-wrap gap-2">
            {guardrails.map((g) => (
              <Badge key={g} variant="outline">
                {g}
              </Badge>
            ))}
          </div>
        </div>

        <Button className="mt-auto w-full" onClick={onUseTemplate}>
          Use Template
        </Button>
      </CardContent>
    </Card>
  );
};

interface PolicyTemplatesProps {
  onUseTemplate: (templateData: any) => void;
  onOpenAiSuggestion: () => void;
  onTemplatesLoaded?: (templates: any[]) => void;
  accessToken: string | null;
}

// Map icon names from JSON to actual icon components
const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  ShieldCheckIcon: ShieldCheck,
  ShieldExclamationIcon: ShieldAlert,
  BeakerIcon: FlaskConical,
  CurrencyDollarIcon: CircleDollarSign,
  CheckCircleIcon: CheckCircle2,
};

const PolicyTemplates: React.FC<PolicyTemplatesProps> = ({
  onUseTemplate,
  onOpenAiSuggestion,
  onTemplatesLoaded,
  accessToken,
}) => {
  const [templates, setTemplates] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  // Compute all unique tags with counts
  const tagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    templates.forEach((t) => {
      const tags: string[] = t.tags || [];
      tags.forEach((tag: string) => {
        counts[tag] = (counts[tag] || 0) + 1;
      });
    });
    // Sort alphabetically
    return Object.entries(counts).sort(([a], [b]) => a.localeCompare(b));
  }, [templates]);

  // Filter templates: show templates that have ALL selected tags (AND logic)
  const filteredTemplates = useMemo(() => {
    if (selectedTags.size === 0) return templates;
    return templates.filter((t) => {
      const tags: string[] = t.tags || [];
      return Array.from(selectedTags).every((selectedTag) => tags.includes(selectedTag));
    });
  }, [templates, selectedTags]);

  const handleTagToggle = (tag: string) => {
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) {
        next.delete(tag);
      } else {
        next.add(tag);
      }
      return next;
    });
  };

  const handleClearAll = () => {
    setSelectedTags(new Set());
  };

  useEffect(() => {
    const fetchTemplates = async () => {
      if (!accessToken) return;

      setIsLoading(true);
      try {
        const data = await getPolicyTemplates(accessToken);
        setTemplates(data);
        onTemplatesLoaded?.(data);
      } catch (error) {
        console.error("Error fetching policy templates:", error);
        MessageManager.error("Failed to fetch policy templates");
      } finally {
        setIsLoading(false);
      }
    };

    fetchTemplates();
  }, [accessToken]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-6 py-20 md:grid-cols-2 xl:grid-cols-3">
        <Skeleton className="h-72 w-full" />
        <Skeleton className="h-72 w-full" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-lg font-medium">Policy Templates</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Start with a pre-configured policy template to quickly set up guardrails for your organization.
          </p>
        </div>
        <Button variant="outline" onClick={onOpenAiSuggestion}>
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1l1.5 3.5L13 6l-3.5 1.5L8 11 6.5 7.5 3 6l3.5-1.5L8 1zm4 7l.75 1.75L14.5 10.5l-1.75.75L12 13l-.75-1.75L9.5 10.5l1.75-.75L12 8zM4 9l.75 1.75L6.5 11.5l-1.75.75L4 14l-.75-1.75L1.5 11.5l1.75-.75L4 9z" />
          </svg>
          Use AI to find templates
        </Button>
      </div>

      <div className="flex gap-6">
        {/* Left sidebar - tag filters */}
        {tagCounts.length > 0 && (
          <div className="w-52 shrink-0">
            <div className="sticky top-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold">Categories</span>
                {selectedTags.size > 0 && (
                  <button onClick={handleClearAll} className="text-xs text-primary hover:underline">
                    Clear all
                  </button>
                )}
              </div>
              <div className="space-y-1">
                {tagCounts.map(([tag, count]) => (
                  <label
                    key={tag}
                    className={`flex items-center justify-between px-2 py-1.5 rounded-md cursor-pointer transition-colors ${
                      selectedTags.has(tag) ? "bg-accent" : "hover:bg-muted"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Checkbox checked={selectedTags.has(tag)} onCheckedChange={() => handleTagToggle(tag)} />
                      <span className="text-sm">{tag}</span>
                    </div>
                    <span className="text-xs font-medium text-muted-foreground">{count}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Right content - template cards */}
        <div className="flex-1">
          {selectedTags.size > 0 && (
            <div className="mb-4 text-sm text-muted-foreground">
              Showing {filteredTemplates.length} of {templates.length} templates
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredTemplates.map((template, index) => (
              <PolicyTemplateCard
                key={template.id || index}
                title={template.title}
                description={template.description}
                icon={iconMap[template.icon] || ShieldCheck}
                iconColor={template.iconColor}
                iconBg={template.iconBg}
                guardrails={template.guardrails}
                tags={template.tags || []}
                inherits={template.inherits}
                complexity={template.complexity}
                onUseTemplate={() => onUseTemplate(template)}
              />
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="py-12 text-center text-muted-foreground">
              <p>No templates match the selected filters.</p>
              <button onClick={handleClearAll} className="mt-2 text-sm text-primary hover:underline">
                Clear all filters
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PolicyTemplates;
