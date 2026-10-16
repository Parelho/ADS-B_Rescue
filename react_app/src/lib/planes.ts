import type { Plane, RawPlane } from "../types";
import { getHeading, makeCircle } from "./geo";
import { dateToUnixSeconds, dateAfterToUnixSeconds } from "./format";

const env = import.meta.env;

async function loadFallbackPlanes(): Promise<RawPlane[]> {
  const path: string = env.VITE_PLANE_DATA_FILE || "/response.json";
  try {
    const res = await fetch(path);
    if (!res.ok) return [];
    const payload: unknown = await res.json();
    return Array.isArray(payload) ? (payload as RawPlane[]) : [];
  } catch (err) {
    console.warn(`Unable to read fallback plane data from ${path}:`, err);
    return [];
  }
}

/** Port of `get_lista()` in planes.py. */
export async function fetchRawPlanes(): Promise<RawPlane[]> {
  const apiUrl: string | undefined = env.VITE_SUPABASE_URL;
  if (!apiUrl) return loadFallbackPlanes();

  const headers: Record<string, string> = {};
  if (env.VITE_SUPABASE_API_KEY) headers["apikey"] = env.VITE_SUPABASE_API_KEY;
  if (env.VITE_SUPABASE_AUTHORIZATION) {
    headers["Authorization"] = `Bearer ${env.VITE_SUPABASE_AUTHORIZATION}`;
  }

  try {
    const res = await fetch(`${apiUrl}/rest/v1/trajectory`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload: unknown = await res.json();
    return Array.isArray(payload) ? (payload as RawPlane[]) : [];
  } catch (err) {
    console.warn("Unable to fetch remote plane data, using fallback data:", err);
    return loadFallbackPlanes();
  }
}

export function normalizePlanes(
  raw: RawPlane[],
  startDate?: string,
  endDate?: string,
): Plane[] {
  const startTimestamp = startDate
    ? dateToUnixSeconds(startDate)
    : undefined;

  const endTimestamp = endDate
    ? dateAfterToUnixSeconds(endDate)
    : undefined;

  return raw
    .map((p) => {
      const trajectory = (p.trajectory ?? []).filter((t) => {
        if (typeof t.timestamp !== "number") return false;

        return (
          (startTimestamp === undefined ||
            t.timestamp >= startTimestamp) &&
          (endTimestamp === undefined ||
            t.timestamp < endTimestamp)
        );
      });

      const route = trajectory.filter((pt) => pt.pred === false);
      const routePoints = route.map(
        (pt) => [pt.lat, pt.lon] as [number, number],
      );
      const predPoints = trajectory
        .filter((pt) => pt.pred === true)
        .map((pt) => [pt.lat, pt.lon] as [number, number]);

      const last = routePoints[routePoints.length - 1];

      return {
        icao: p.icao,
        callsign: p.callsign,
        time: route.length
          ? (route[route.length - 1].timestamp ?? null)
          : null,
        routePoints,
        predPoints,
        searchArea: last
          ? makeCircle(last[0], last[1], 0.08)
          : null,
        heading: getHeading(routePoints),
      };
    });
}

export async function loadPlanes(
  startDate?: string,
  endDate?: string,
): Promise<Plane[]> {
  return normalizePlanes(
    await fetchRawPlanes(),
    startDate,
    endDate,
  );
}
