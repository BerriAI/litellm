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
import { ChevronDown } from "lucide-react";
import { forwardRef, startTransition, useMemo, useState } from "react";

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
      className={cx("flex flex-col gap-1.5 max-w-[320px]", className)}
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
    <div className={cx("flex grow flex-col gap-1.5", className)} {...props} />
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

function UiFormRadioCircle() {
  return (
    <span
      className={cx(
        "inline-flex items-center justify-center shrink-0",
        "size-4 rounded-full",
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
        <UiFormRadioCircle />
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
            "bg-white rounded-md overflow-x-hidden",
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
