import React from "react";
import { NumberInput } from "@tremor/react";

/**
 * A reusable numerical input component
 * @param {Object} props - Component props
 * @param {number} [props.step=0.01] - Step increment for the input
 * @param {Object} [props.style] - Custom styles to apply
 * @param {string} [props.placeholder="Enter a numerical value"] - Placeholder text
 * @param {number} [props.min] - Minimum value
 * @param {number} [props.max] - Maximum value
 * @param {Function} [props.onChange] - On change handler
 * @param {any} props.rest - Additional props passed to NumberInput
 */
const NumericalInput = ({
  step = 0.01,
  style = { width: "100%" },
  placeholder = "Enter a numerical value",
  min,
  max,
  onChange,
  ...rest
}) => {
  return (
    <NumberInput
      step={step}
      style={style}
      placeholder={placeholder}
      min={min}
      max={max}
      onChange={onChange}
      {...rest}
    />
  );
};

export default NumericalInput; 