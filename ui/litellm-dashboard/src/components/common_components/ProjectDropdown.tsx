import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

interface ProjectDropdownProps {
  projects?: ProjectResponse[] | null;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  loading?: boolean;
  /** When set, only show projects belonging to this team */
  teamId?: string | null;
}

const ALL = "__all__";

const ProjectDropdown: React.FC<ProjectDropdownProps> = ({
  projects,
  value,
  onChange,
  disabled,
  loading,
  teamId,
}) => {
  const filtered = teamId
    ? projects?.filter((p) => p.team_id === teamId)
    : projects;

  return (
    <Select
      value={value ?? ALL}
      onValueChange={(v) => onChange?.(v === ALL ? "" : v)}
      disabled={disabled || loading}
    >
      <SelectTrigger>
        <SelectValue placeholder="Search or select a project" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>All Projects</SelectItem>
        {!loading &&
          filtered?.map((project) => (
            <SelectItem key={project.project_id} value={project.project_id}>
              <span className="font-medium">
                {project.project_alias || project.project_id}
              </span>{" "}
              <span className="text-muted-foreground">
                ({project.project_id})
              </span>
            </SelectItem>
          ))}
      </SelectContent>
    </Select>
  );
};

export default ProjectDropdown;
