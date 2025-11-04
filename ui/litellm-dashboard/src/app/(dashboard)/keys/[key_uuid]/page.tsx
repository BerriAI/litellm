import KeyDetailClient from "./KeyDetailClient";

export async function generateStaticParams() {
  return [{ key_uuid: "index" }];
}

const KeyDetailPage = () => {
  return <KeyDetailClient />;
};

export default KeyDetailPage;
