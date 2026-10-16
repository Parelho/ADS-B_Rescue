export type LatLon = [number, number];

/** One point of a trajectory as returned by Supabase `/rest/v1/trajectory`. */
export interface RawTrajectoryPoint {
  lat: number;
  lon: number;
  pred?: boolean;
  timestamp?: number | string | null;
}

export interface RawPlane {
  icao: string;
  callsign: string;
  trajectory?: RawTrajectoryPoint[] | null;
}

/** A plane after normalisation (equivalent to `plane_data` in map_window.py). */
export interface Plane {
  icao: string;
  callsign: string;
  time: number | string | null;
  routePoints: LatLon[];
  predPoints: LatLon[];
  searchArea: LatLon[] | null;
  heading: number;
}

export interface Airport {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export type Selection =
  | { kind: "plane"; plane: Plane }
  | { kind: "airport"; airport: Airport };
