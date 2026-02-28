import React from "react";
import { Select, Spin } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
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
      showSearch
      placeholder="Search or select a project"
      value={value}
      onChange={onChange}
      disabled={disabled}
      loading={loading}
      allowClear
      notFoundContent={loading ? <Spin indicator={<LoadingOutlined spin />} size="small" /> : undefined}
      filterOption={(input, option) => {
        if (!option) return false;
        const project = filtered?.find((p) => p.project_id === option.key);
        if (!project) return false;

        const searchTerm = input.toLowerCase().trim();
        const alias = (project.project_alias || "").toLowerCase();
        const id = (project.project_id || "").toLowerCase();

        return alias.includes(searchTerm) || id.includes(searchTerm);
      }}
      optionFilterProp="children"
    >
      {!loading &&
        filtered?.map((project) => (
          <Select.Option key={project.project_id} value={project.project_id}>
            <span className="font-medium">{project.project_alias || project.project_id}</span>{" "}
            <span className="text-gray-500">({project.project_id})</span>
          </Select.Option>
        ))}
    </Select>
  );
};

export default ProjectDropdown;
