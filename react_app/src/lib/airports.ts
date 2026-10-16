import Papa from "papaparse";
import type { Airport } from "../types";

const AIRPORT_TYPES = new Set(["small_airport", "medium_airport", "large_airport"]);

interface CsvRow {
  id?: string;
  type?: string;
  name?: string;
  latitude_deg?: string;
  longitude_deg?: string;
}

/** Loads airports.csv (OurAirports format) keeping only real airports. */
export async function loadAirports(url = "/airports.csv"): Promise<Airport[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  const { data } = Papa.parse<CsvRow>(text, { header: true, skipEmptyLines: true });

  const airports: Airport[] = [];
  for (const row of data) {
    if (!row.type || !AIRPORT_TYPES.has(row.type)) continue;
    const lat = parseFloat(row.latitude_deg ?? "");
    const lon = parseFloat(row.longitude_deg ?? "");
    if (Number.isNaN(lat) || Number.isNaN(lon)) continue;
    airports.push({ id: row.id ?? `${lat},${lon}`, name: row.name ?? "", lat, lon });
  }
  return airports;
}

/** First airport inside a +/- delta degree box (same as `get_nearby_airports(...).iloc[0]`). */
export function findNearbyAirport(
  airports: Airport[],
  lat: number,
  lon: number,
  delta = 0.2,
): Airport | null {
  return (
    airports.find(
      (a) =>
        a.lat >= lat - delta &&
        a.lat <= lat + delta &&
        a.lon >= lon - delta &&
        a.lon <= lon + delta,
    ) ?? null
  );
}
