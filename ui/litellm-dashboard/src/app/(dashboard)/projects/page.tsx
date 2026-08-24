"use client";

import { ProjectsPage } from "./components/ProjectsPage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Projects() {
  useAuthorized();
  return <ProjectsPage />;
}
