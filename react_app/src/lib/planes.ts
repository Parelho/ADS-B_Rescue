import type { Plane, RawPlane } from "../types";
import { getHeading, makeCircle } from "./geo";

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

/** Equivalent to the `plane_data` construction in `MapWindow.update_map`. */
export function normalizePlanes(raw: RawPlane[]): Plane[] {
  return raw.map((p) => {
    const trajectory = p.trajectory ?? [];
    const route = trajectory.filter((pt) => pt.pred === false);
    const routePoints = route.map((pt) => [pt.lat, pt.lon] as [number, number]);
    const predPoints = trajectory
      .filter((pt) => pt.pred === true)
      .map((pt) => [pt.lat, pt.lon] as [number, number]);
    const last = routePoints[routePoints.length - 1];

    return {
      icao: p.icao,
      callsign: p.callsign,
      time: route.length ? (route[route.length - 1].timestamp ?? null) : null,
      routePoints,
      predPoints,
      searchArea: last ? makeCircle(last[0], last[1], 0.08) : null,
      heading: getHeading(routePoints),
    };
  });
}

export async function loadPlanes(): Promise<Plane[]> {
  return normalizePlanes(await fetchRawPlanes());
}
