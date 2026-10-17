import { useCallback, useEffect, useState } from "react";
import type { Airport, Plane } from "../types";
import { loadAirports } from "../lib/airports";
import { loadPlanes } from "../lib/planes";

export function useAirports() {
  const [airports, setAirports] = useState<Airport[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadAirports()
      .then((a) => !cancelled && setAirports(a))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  return { airports, error };
}


export function usePlanes(startDate = "", endDate = "") {
  const [planes, setPlanes] = useState<Plane[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setPlanes(
        await loadPlanes(
          startDate || undefined,
          endDate || undefined,
        ),
      );
      setLastUpdatedAt(new Date());
    } finally {
      setLoading(false);
    }
  }, [startDate, endDate]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { planes, loading, refresh, lastUpdatedAt };
}
