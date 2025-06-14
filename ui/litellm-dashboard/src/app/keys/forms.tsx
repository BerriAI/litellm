import { cx } from "@/lib/cva.config";
import {
  Combobox,
  ComboboxItem,
  ComboboxList,
  ComboboxProvider,
  Radio,
  RadioGroup,
  RadioGroupProps,
  RadioProps,
  RadioProvider,
  RadioProviderProps,
  Select,
  SelectItem,
  SelectPopover,
  SelectProps,
  SelectProvider,
  useRadioContext,
} from "@ariakit/react";
import { CheckIcon, ChevronDown, XIcon } from "lucide-react";
import { forwardRef, startTransition, useMemo, useState } from "react";
import { AiModel, models } from "./data";
import { title } from "process";

export function UiFormGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cx(
        "flex items-start gap-6 py-5",
        "border-b last-of-type:border-b-0 border-dashed",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormLabelGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cx("flex flex-col gap-1.5 shrink-0 max-w-[320px]", className)}
      {...props}
    />
  );
}

export function UiFormLabel({
  className,
  ...props
}: React.ComponentProps<"label">) {
  return (
    <label
      className={cx(
        "block tracking-tight text-[15px] text-neutral-900",
        "",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormDescription({
  className,
  ...props
}: React.ComponentProps<"p">) {
  return (
    <p
      className={cx(
        "text-[13px]/[1.7] tracking-tight text-neutral-500",
        className,
      )}
      {...props}
    />
  );
}

export function UiFormContent({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div className={cx("flex grow flex-col gap-4", className)} {...props} />
  );
}

export const UiFormTextInput = forwardRef<
  HTMLInputElement,
  React.ComponentProps<"input">
>(function UiFormTextInput({ className, ...props }, ref) {
  return (
    <input
      ref={ref}
      className={cx(
        "block w-full max-w-[400px] h-[34px] truncate rounded-md bg-neutral-50 px-4",
        "border-none",
        "ring-[0.7px] ring-black/[0.12]",
        "shadow-md shadow-black/[0.05]",
        "text-[13px] font-normal tracking-tight text-neutral-900",
        "placeholder:text-neutral-400",
        "transition-colors",
        "focus:bg-white",
        className,
      )}
      {...props}
    />
  );
});

type UiFormRadioGroupProps = Omit<RadioGroupProps, "defaultValue"> & {
  defaultValue?: RadioProviderProps["defaultValue"];
};

export function UiFormRadioGroup({
  className,
  defaultValue,
  ...props
}: UiFormRadioGroupProps) {
  return (
    <RadioProvider defaultValue={defaultValue}>
      <RadioGroup className={cx("flex flex-col gap-2", className)} {...props} />
    </RadioProvider>
  );
}

function UiFormCircle() {
  return (
    <span
      className={cx(
        "inline-flex items-center justify-center shrink-0",
        "size-[16px] rounded-full",
        "ring-[0.5px] ring-black/[0.12]",
        "shadow-md shadow-black/[0.06]",
        "transition-colors duration-200 ease-out",
        "bg-white group-aria-checked:bg-indigo-500",
      )}
    >
      <span
        className={cx(
          "size-[7px] bg-white rounded-full",
          "transition-[opacity,transform] origin-center duration-200 ease-out",
          "shadow-sm shadow-black/30",
          "scale-0 group-aria-checked:scale-100",
        )}
      />
    </span>
  );
}

function UiFormCheck() {
  return (
    <span
      className={cx(
        "inline-flex items-center justify-center shrink-0",
        "size-[16px] rounded",
        "ring-[0.5px] ring-black/[0.12]",
        "shadow-md shadow-black/[0.06]",
        "transition-colors duration-200 ease-out",
        "bg-white group-aria-selected:bg-indigo-500",
      )}
    >
      <CheckIcon
        size={10}
        strokeWidth={4}
        className={cx(
          "text-white",
          "transition-[opacity,transform] mt-px origin-center duration-200 ease-out",
          "scale-0 group-aria-selected:scale-100",
        )}
      />
    </span>
  );
}

export function UiFormRadio({ className, children, ...props }: RadioProps) {
  const store = useRadioContext();

  return (
    <Radio
      onFocusVisible={() => {
        store?.setValue(props.value);
      }}
      render={<button />}
      className={cx("group outline-none", className)}
      {...props}
    >
      <div className="text-start flex items-center gap-2">
        <UiFormCircle />
        <span className="text-[13px] tracking-tight text-neutral-800">
          {children}
        </span>
      </div>
    </Radio>
  );
}

export type UiFormComboboxItem = {
  title: string;
  subtitle?: string;
};

type UiFormComboboxProps = SelectProps & {
  placeholder?: string;
  items: UiFormComboboxItem[];
};

export function UiFormCombobox({
  className,
  placeholder,
  items,
  ...props
}: UiFormComboboxProps) {
  const [searchValue, setSearchValue] = useState("");
  const resolvedPlaceholder = placeholder || "Select an item...";

  const matches = useMemo(() => {
    return items.filter((item) => {
      return (
        item.title.toLowerCase().includes(searchValue) ||
        (item.subtitle
          ? item.subtitle.toLowerCase().includes(searchValue)
          : false)
      );
    });
  }, [searchValue, items]);

  return (
    <ComboboxProvider
      includesBaseElement={false}
      resetValueOnHide
      setValue={(value) => {
        startTransition(() => {
          setSearchValue(value);
        });
      }}
    >
      <SelectProvider>
        <Select
          className={cx(
            "relative block w-full max-w-[400px] h-[34px] truncate rounded-md bg-neutral-50 px-4",
            "border-none",
            "ring-[0.7px] ring-black/[0.12]",
            "shadow-md shadow-black/[0.05]",
            "transition-colors",
            "text-start",
            className,
          )}
          {...props}
        >
          <span className="text-[13px] font-normal tracking-tight text-neutral-400">
            {resolvedPlaceholder}
          </span>

          <span
            className={cx(
              "absolute right-0 inset-y-0 pr-3 pointer-events-none",
              "flex items-center justify-center",
            )}
          >
            <ChevronDown size={16} className="text-neutral-500" />
          </span>
        </Select>

        <SelectPopover
          gutter={8}
          sameWidth
          unmountOnHide
          className={cx(
            "bg-white rounded-md overflow-x-hidden z-10",
            "flex flex-col",
            "max-h-[min(320px,var(--popover-available-height,320px))]",
            "ring-[0.7px] ring-black/[0.12] outline-none",
            "shadow-lg shadow-black/[0.08]",
            "transition-transform duration-[150ms] ease-out",
            "translate-y-[-2%] opacity-0",
            "data-[enter]:translate-y-0 data-[enter]:opacity-100",
          )}
        >
          <div className={cx("", "border-b border-neutral-200")}>
            <Combobox
              autoFocus
              style={{ boxShadow: "none" }}
              autoSelect
              className={cx(
                "block w-full max-w-[400px] h-[34px] truncate rounded-none bg-transparent px-4",
                "border-none outline-none",
                "text-[13px] font-normal tracking-tight text-neutral-900",
                "placeholder:text-neutral-400",
              )}
              placeholder="Search..."
            />
          </div>

          <ComboboxList className="overflow-y-auto grow outline-none">
            {matches.map((item) => (
              <SelectItem
                key={item.title}
                render={<ComboboxItem />}
                className={cx(
                  "px-4 py-2.5",
                  "border-b border-neutral-100",
                  "flex items-center gap-2 shrink-0 outline-none",
                  "data-[active-item]:bg-neutral-100 cursor-default",
                )}
              >
                <span className="text-[13px] text-neutral-800 tracking-tight truncate">
                  {item.title}
                </span>

                {item.subtitle ? (
                  <span className="text-[11px] text-neutral-500 truncate">
                    {item.subtitle}
                  </span>
                ) : null}
              </SelectItem>
            ))}
          </ComboboxList>
        </SelectPopover>
      </SelectProvider>
    </ComboboxProvider>
  );
}

type AiModelOption = {
  title: string;
  value: string;
};

function getModelTitle(id: string): string {
  if (id.endsWith("/*")) {
    return "All " + id.split("/*")[0] + " models";
  }

  return id;
}

function transformModel(id: string): AiModelOption {
  return { title: getModelTitle(id), value: id };
}

function transformModels(ids: string[]): AiModelOption[] {
  const modelOptions: Record<string, AiModelOption> = {};

  for (const id of ids) {
    if (id in modelOptions === false) {
      modelOptions[id] = transformModel(id);
    }
  }

  return Object.values(modelOptions).sort((a, b) => {
    const aEndsWithWildcard = a.value.endsWith("/*");
    const bEndsWithWildcard = b.value.endsWith("/*");

    // If one ends with /* and the other doesn't, prioritize the wildcard
    if (aEndsWithWildcard && !bEndsWithWildcard) {
      return -1;
    }

    if (!aEndsWithWildcard && bEndsWithWildcard) {
      return 1;
    }

    // If both are wildcards or both are not wildcards, sort alphabetically
    return a.title.localeCompare(b.title);
  });
}

type UiModelSelectProps = SelectProps & {};

export function UiModelSelect({ className, ...props }: UiModelSelectProps) {
  const [searchValue, setSearchValue] = useState("");
  const [selectedValues, setSelectedValues] = useState<string[]>([]);
  console.log(selectedValues);
  const placeholder = "Select Models";

  const items = useMemo(() => {
    return transformModels(models);
  }, []);

  const matches = useMemo(() => {
    return items.filter((item) => {
      return item.title.toLowerCase().includes(searchValue);
    });
  }, [searchValue, items]);

  return (
    <ComboboxProvider
      includesBaseElement={false}
      resetValueOnHide
      setValue={(value) => {
        startTransition(() => {
          setSearchValue(value);
        });
      }}
    >
      <SelectProvider value={selectedValues} setValue={setSelectedValues}>
        <Select
          className={cx(
            "relative block w-full min-h-[34px] truncate rounded-md",
            "border-none text-start",
            "ring-[0.7px] ring-black/[0.12]",
            "shadow-md shadow-black/[0.05] transition-colors",
            selectedValues.length === 0 ? "bg-neutral-50" : "bg-white",
            className,
          )}
          {...props}
        >
          {selectedValues.length === 0 ? (
            <span className="text-[13px] font-normal tracking-tight text-neutral-400 px-4">
              {placeholder}
            </span>
          ) : (
            <div className="flex flex-wrap items-center gap-1 px-1 py-1">
              {selectedValues.map(transformModel).map((option, optionIndex) => (
                <div
                  key={option.value}
                  className={cx(
                    "bg-black/[0.07] rounded inline-flex items-center gap-1",
                  )}
                >
                  <span className="text-[12px] inline-block py-1 pl-2">
                    {option.title}
                  </span>

                  <div
                    className={cx(
                      "flex items-center group/close",
                      "shrink-0 pr-2",
                    )}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={(event) => {
                      event.preventDefault();
                      setSelectedValues((v) => {
                        return v.reduce<string[]>((result, item, index) => {
                          if (index !== optionIndex) {
                            result.push(item);
                          }
                          return result;
                        }, []);
                      });
                    }}
                  >
                    <XIcon
                      size={14}
                      strokeWidth={3}
                      className="text-neutral-400 group-hover/close:text-neutral-600"
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          {selectedValues.length === 0 ? (
            <span
              className={cx(
                "absolute right-0 inset-y-0 pr-3 pointer-events-none",
                "flex items-center justify-center",
              )}
            >
              <ChevronDown size={16} className="text-neutral-500" />
            </span>
          ) : null}
        </Select>

        <SelectPopover
          gutter={8}
          sameWidth
          unmountOnHide
          className={cx(
            "bg-white rounded-md overflow-x-hidden",
            "flex flex-col",
            "max-h-[min(320px,var(--popover-available-height,320px))]",
            "ring-[0.7px] ring-black/[0.12] outline-none",
            "shadow-lg shadow-black/[0.08]",
            "transition-transform duration-[150ms] ease-out",
            "translate-y-[-2%] opacity-0",
            "data-[enter]:translate-y-0 data-[enter]:opacity-100 z-10",
          )}
        >
          <div className={cx("", "border-b border-neutral-200")}>
            <Combobox
              autoFocus
              style={{ boxShadow: "none" }}
              autoSelect
              className={cx(
                "block w-full h-[34px] truncate rounded-none bg-transparent px-4",
                "border-none outline-none",
                "text-[13px] font-normal tracking-tight text-neutral-900",
                "placeholder:text-neutral-400",
              )}
              placeholder="Search..."
            />
          </div>

          <ComboboxList className="overflow-y-auto grow outline-none">
            {matches.map((item) => (
              <SelectItem
                key={item.title}
                render={<ComboboxItem />}
                value={item.value}
                className={cx(
                  "px-4 py-2.5",
                  "border-b border-neutral-100",
                  "flex items-center gap-2 shrink-0 outline-none",
                  "data-[active-item]:bg-neutral-100 cursor-default group",
                )}
              >
                <UiFormCheck />
                <span className="text-[13px] text-neutral-800 tracking-tight truncate">
                  {item.title}
                </span>
              </SelectItem>
            ))}
          </ComboboxList>
        </SelectPopover>
      </SelectProvider>
    </ComboboxProvider>
  );
}
