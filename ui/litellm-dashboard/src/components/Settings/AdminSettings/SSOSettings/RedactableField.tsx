import { useState } from "react";
import { Button } from "antd";
import { Eye, EyeOff } from "lucide-react";

export default function RedactableField({
  defaultHidden = true,
  value,
}: {
  defaultHidden?: boolean;
  value: string | null;
}) {
  const [isHidden, setIsHidden] = useState(defaultHidden);

  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-gray-600 flex-1">
        {value ? (
          isHidden ? (
            "•".repeat(value.length)
          ) : (
            value
          )
        ) : (
          <span className="text-gray-400 italic">Not configured</span>
        )}
      </span>
      {value && (
        <Button
          type="text"
          size="small"
          icon={isHidden ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          onClick={() => setIsHidden(!isHidden)}
          className="text-gray-400 hover:text-gray-600"
        />
      )}
    </div>
  );
}
