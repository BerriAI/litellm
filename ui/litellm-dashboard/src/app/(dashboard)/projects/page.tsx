"use client";

import { ProjectsPage } from "./_components/ProjectsPage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Projects() {
  useAuthorized();
  return <ProjectsPage />;
}
