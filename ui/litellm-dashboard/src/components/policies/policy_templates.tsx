import React, { useState, useEffect, useMemo } from "react";
import { Card, Button, Spin, message, Radio } from "antd";
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
      <p className="text-sm text-gray-500 mb-6 flex-grow">{description}</p>

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

const PolicyTemplates: React.FC<PolicyTemplatesProps> = ({ onUseTemplate, accessToken }) => {
  const [templates, setTemplates] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<string>("All");
  const [selectedType, setSelectedType] = useState<string>("All");

  const availableRegions = useMemo(() => {
    const regions = new Set(templates.map(t => t.region || "Global"));
    return ["All", ...Array.from(regions).sort()];
  }, [templates]);

  const availableTypes = useMemo(() => {
    const types = new Set(templates.map(t => t.type || "General"));
    return ["All", ...Array.from(types).sort()];
  }, [templates]);

  const filteredTemplates = useMemo(() => {
    return templates.filter(t => {
      const regionMatch = selectedRegion === "All" || (t.region || "Global") === selectedRegion;
      const typeMatch = selectedType === "All" || (t.type || "General") === selectedType;
      return regionMatch && typeMatch;
    });
  }, [templates, selectedRegion, selectedType]);

  useEffect(() => {
    const fetchTemplates = async () => {
      if (!accessToken) return;

      setIsLoading(true);
      try {
        const data = await getPolicyTemplates(accessToken);
        setTemplates(data);
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
      </div>

      <div className="flex items-center gap-6 mb-4">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-700">Region:</span>
          <Radio.Group
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            buttonStyle="solid"
          >
            {availableRegions.map(region => (
              <Radio.Button key={region} value={region}>
                {region}
              </Radio.Button>
            ))}
          </Radio.Group>
        </div>
        {availableTypes.length > 2 && (
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">Type:</span>
            <Radio.Group
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              buttonStyle="solid"
            >
              {availableTypes.map(type => (
                <Radio.Button key={type} value={type}>
                  {type}
                </Radio.Button>
              ))}
            </Radio.Group>
          </div>
        )}
      </div>

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
            inherits={template.inherits}
            complexity={template.complexity}
            onUseTemplate={() => onUseTemplate(template)}
          />
        ))}
      </div>
    </div>
  );
};

export default PolicyTemplates;
