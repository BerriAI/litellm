import React, { useMemo } from "react";
import { Spin, Checkbox } from "antd";
import { useClaudeCodeMarketplaces } from "@/app/(dashboard)/hooks/claudeCodeMarketplaces/useClaudeCodeMarketplaces";
import { useClaudeCodePlugins } from "@/app/(dashboard)/hooks/claudeCodePlugins/useClaudeCodePlugins";
import { PluginListItem } from "./types";

interface SkillPermissionsPickerProps {
  accessToken: string;
  value?: string[];
  onChange: (skills: string[]) => void;
  disabled?: boolean;
}

const marketplacePrefix = (marketplaceName: string) => `${marketplaceName}--`;

const SkillPermissionsPicker: React.FC<SkillPermissionsPickerProps> = ({ value, onChange, disabled = false }) => {
  const { data: marketplaces = [], isLoading: marketplacesLoading } = useClaudeCodeMarketplaces();
  const { data: allSkills = [], isLoading: skillsLoading } = useClaudeCodePlugins();

  const selected = useMemo(() => value ?? [], [value]);
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const skillsByMarketplace = useMemo(() => {
    return marketplaces.reduce<Record<string, PluginListItem[]>>((acc, marketplace) => {
      const prefix = marketplacePrefix(marketplace.name);
      return { ...acc, [marketplace.name]: allSkills.filter((skill) => skill.name.startsWith(prefix)) };
    }, {});
  }, [marketplaces, allSkills]);

  const handleToggleMarketplace = (marketplaceName: string, checked: boolean) => {
    if (disabled) return;
    const prefix = marketplacePrefix(marketplaceName);
    if (checked) {
      const skillNames = (skillsByMarketplace[marketplaceName] || []).map((skill) => skill.name);
      onChange(Array.from(new Set([...selected, ...skillNames])));
    } else {
      onChange(selected.filter((name) => !name.startsWith(prefix)));
    }
  };

  const handleToggleSkill = (skillName: string, checked: boolean) => {
    if (disabled) return;
    onChange(checked ? Array.from(new Set([...selected, skillName])) : selected.filter((name) => name !== skillName));
  };

  if (marketplacesLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spin size="large" />
        <span className="ml-3 text-gray-500">Loading marketplaces...</span>
      </div>
    );
  }

  if (marketplaces.length === 0) {
    return <p className="text-gray-500">No marketplaces imported yet. Add one from the Skills page.</p>;
  }

  return (
    <div className="space-y-4">
      {marketplaces.map((marketplace) => {
        const skills = skillsByMarketplace[marketplace.name] || [];
        const selectedCount = skills.filter((skill) => selectedSet.has(skill.name)).length;
        const checked = skills.length > 0 && selectedCount === skills.length;
        const indeterminate = selectedCount > 0 && selectedCount < skills.length;

        return (
          <div key={marketplace.name} className="border rounded-lg bg-gray-50">
            <div className="flex items-center justify-between p-4 border-b bg-white rounded-t-lg">
              <Checkbox
                checked={checked}
                indeterminate={indeterminate}
                disabled={disabled || skills.length === 0}
                onChange={(e) => handleToggleMarketplace(marketplace.name, e.target.checked)}
              >
                <p className="font-semibold text-gray-900">{marketplace.display_name || marketplace.name}</p>
                <p className="text-sm text-gray-500">
                  {marketplace.source_type} · {selectedCount}/{skills.length} skills selected
                </p>
              </Checkbox>
            </div>

            <div className="p-4">
              {skillsLoading && (
                <div className="flex items-center justify-center py-4">
                  <Spin />
                  <span className="ml-3 text-gray-500">Loading skills...</span>
                </div>
              )}

              {!skillsLoading && skills.length === 0 && (
                <p className="text-gray-500">No skills found for this marketplace</p>
              )}

              {!skillsLoading && skills.length > 0 && (
                <div className="space-y-2">
                  {skills.map((skill) => {
                    const isSelected = selectedSet.has(skill.name);
                    return (
                      <label key={skill.name} className="flex items-start gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => handleToggleSkill(skill.name, !isSelected)}
                          disabled={disabled}
                          className="mt-0.5"
                        />
                        <span className="flex-1 min-w-0 flex items-center gap-2">
                          <span className="font-medium text-gray-900">{skill.name}</span>
                          <span className="text-sm text-gray-500">- {skill.description || "No description"}</span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SkillPermissionsPicker;
