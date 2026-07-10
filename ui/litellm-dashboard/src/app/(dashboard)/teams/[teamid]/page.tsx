import TeamDetailPage from "./TeamDetailPage";

export function generateStaticParams() {
  return [{ teamid: "placeholder" }];
}

export default function TeamDetailRoute() {
  return <TeamDetailPage />;
}
