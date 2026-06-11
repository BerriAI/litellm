"use client";

import { ProjectsPage } from "@/components/Projects/ProjectsPage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Projects() {
  useAuthorized();
  return <ProjectsPage />;
}
