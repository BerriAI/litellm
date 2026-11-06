import React from "react";
import { useTranslation } from "react-i18next";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

interface ProjectDropdownProps {
  projects?: ProjectResponse[] | null;
  value?: string | null;
  onChange?: (value: string | null) => void;
  disabled?: boolean;
  loading?: boolean;
  /** When set, only show projects belonging to this team */
  teamId?: string | null;
  id?: string;
}

const ProjectDropdown: React.FC<ProjectDropdownProps> = ({
  projects,
  value,
  onChange,
  disabled,
  loading,
  teamId,
  id,
}) => {
  const { t } = useTranslation();
  const filtered = teamId ? projects?.filter((p) => p.team_id === teamId) : projects;

  return (
    <SearchSelect
      options={
        loading
          ? []
          : (filtered ?? []).map((project) => ({
              label: project.project_alias || project.project_id,
              value: project.project_id,
              sublabel: project.project_id,
            }))
      }
      value={value}
      onValueChange={(projectId) => onChange?.(projectId)}
      placeholder={t("projectDropdown.placeholder")}
      emptyText={loading ? t("projectDropdown.loading") : t("projectDropdown.empty")}
      disabled={disabled}
      inputId={id}
    />
  );
};

export default ProjectDropdown;
