import React, { useState, useEffect, useMemo } from "react";
import { Card, Button, Spin, message, Checkbox } from "antd";
import {
  ShieldCheckIcon,
  ShieldExclamationIcon,
  BeakerIcon,
  CurrencyDollarIcon,
  CheckCircleIcon,
} from "@heroicons/react/outline";
import { getPolicyTemplates } from "../networking";

interface PolicyTemplateCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
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
  const getComplexityStyle = () => {
    switch (complexity) {
      case "Low":
        return "bg-gray-50 text-gray-600 border-gray-200";
      case "Medium":
        return "bg-blue-50 text-blue-600 border-blue-100";
      case "High":
        return "bg-purple-50 text-purple-600 border-purple-100";
    }
  };

  return (
    <Card
      className="h-full hover:shadow-md transition-shadow"
      bodyStyle={{ display: "flex", flexDirection: "column", height: "100%" }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`p-2 rounded-lg ${iconBg}`}>
          <Icon className={`h-6 w-6 ${iconColor}`} />
        </div>
        <span
          className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${getComplexityStyle()}`}
        >
          {complexity} Complexity
        </span>
      </div>

      <h3 className="text-base font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-500 mb-4 flex-grow">{description}</p>

      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-100"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {inherits && (
        <div className="mb-4 text-xs">
          <span className="text-gray-500">Inherits from: </span>
          <span className="font-medium text-gray-700 bg-gray-100 px-2 py-0.5 rounded">
            {inherits}
          </span>
        </div>
      )}

      <div className="mb-6">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider block mb-2">
          Included Guardrails
        </span>
        <div className="flex flex-wrap gap-2">
          {guardrails.map((g) => (
            <span
              key={g}
              className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-50 text-gray-700 border border-gray-200"
            >
              {g}
            </span>
          ))}
        </div>
      </div>

      <Button
        type="primary"
        block
        className="mt-auto"
        onClick={onUseTemplate}
      >
        Use Template
      </Button>
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
const iconMap: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {
  ShieldCheckIcon: ShieldCheckIcon,
  ShieldExclamationIcon: ShieldExclamationIcon,
  BeakerIcon: BeakerIcon,
  CurrencyDollarIcon: CurrencyDollarIcon,
  CheckCircleIcon: CheckCircleIcon,
};

const PolicyTemplates: React.FC<PolicyTemplatesProps> = ({ onUseTemplate, onOpenAiSuggestion, onTemplatesLoaded, accessToken }) => {
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
        message.error("Failed to fetch policy templates");
      } finally {
        setIsLoading(false);
      }
    };

    fetchTemplates();
  }, [accessToken]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Spin size="large" tip="Loading policy templates..." />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-lg font-medium text-gray-900">
            Policy Templates
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Start with a pre-configured policy template to quickly set up
            guardrails for your organization.
          </p>
        </div>
        <Button
          type="default"
          onClick={onOpenAiSuggestion}
          className="flex items-center gap-1.5"
        >
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1l1.5 3.5L13 6l-3.5 1.5L8 11 6.5 7.5 3 6l3.5-1.5L8 1zm4 7l.75 1.75L14.5 10.5l-1.75.75L12 13l-.75-1.75L9.5 10.5l1.75-.75L12 8zM4 9l.75 1.75L6.5 11.5l-1.75.75L4 14l-.75-1.75L1.5 11.5l1.75-.75L4 9z" />
          </svg>
          Use AI to find templates
        </Button>
      </div>

      <div className="flex gap-6">
        {/* Left sidebar - tag filters */}
        {tagCounts.length > 0 && (
          <div className="w-52 flex-shrink-0">
            <div className="sticky top-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-gray-900">
                  Categories
                </span>
                {selectedTags.size > 0 && (
                  <button
                    onClick={handleClearAll}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    Clear all
                  </button>
                )}
              </div>
              <div className="space-y-1">
                {tagCounts.map(([tag, count]) => (
                  <label
                    key={tag}
                    className={`flex items-center justify-between px-2 py-1.5 rounded-md cursor-pointer transition-colors ${
                      selectedTags.has(tag)
                        ? "bg-blue-50"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Checkbox
                        checked={selectedTags.has(tag)}
                        onChange={() => handleTagToggle(tag)}
                      />
                      <span className="text-sm text-gray-700">{tag}</span>
                    </div>
                    <span className="text-xs text-gray-400 font-medium">
                      {count}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Right content - template cards */}
        <div className="flex-1">
          {selectedTags.size > 0 && (
            <div className="mb-4 text-sm text-gray-500">
              Showing {filteredTemplates.length} of {templates.length} templates
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredTemplates.map((template, index) => (
              <PolicyTemplateCard
                key={template.id || index}
                title={template.title}
                description={template.description}
                icon={iconMap[template.icon] || ShieldCheckIcon}
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
            <div className="text-center py-12 text-gray-500">
              <p>No templates match the selected filters.</p>
              <button
                onClick={handleClearAll}
                className="text-blue-600 hover:text-blue-800 mt-2 text-sm"
              >
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
