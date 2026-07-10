import KeyDetailPage from "./KeyDetailPage";

export function generateStaticParams() {
  return [{ keyid: "placeholder" }];
}

export default function ApiKeyDetailPage() {
  return <KeyDetailPage />;
}
