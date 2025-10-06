import * as React from "react";
import { getCountryFromIP } from "./ip_lookup";

interface CountryCellProps {
  ipAddress: string | null;
}

export const CountryCell: React.FC<CountryCellProps> = ({ ipAddress }) => {
  const [country, setCountry] = React.useState<string>("-");

  React.useEffect(() => {
    if (!ipAddress) return;

    let mounted = true;
    getCountryFromIP(ipAddress)
      .then((result) => {
        if (mounted) setCountry(result);
      })
      .catch(() => {
        if (mounted) setCountry("-");
      });

    return () => {
      mounted = false;
    };
  }, [ipAddress]);

  return <span>{country}</span>;
};
