export type ModelViewType = "groups" | "individual";

const MODEL_VIEW_OPTIONS: readonly { value: ModelViewType; label: string }[] = [
  { value: "groups", label: "Public Model Name" },
  { value: "individual", label: "Litellm Model Name" },
];

interface ModelViewToggleProps {
  value: ModelViewType;
  onChange: (value: ModelViewType) => void;
}

export default function ModelViewToggle({ value, onChange }: ModelViewToggleProps) {
  return (
    <div className="flex bg-gray-100 rounded-lg p-1">
      {MODEL_VIEW_OPTIONS.map((option) => (
        <button
          key={option.value}
          className={`px-3 py-1 text-sm rounded-md transition-colors ${
            value === option.value ? "bg-white shadow-xs text-gray-900" : "text-gray-600 hover:text-gray-900"
          }`}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
