import { useCallback, useMemo, useState } from "react";
import "leaflet/dist/leaflet.css";
import "./App.css";
import type { Airport, Plane, Selection } from "./types";
import { useAirports, usePlanes } from "./hooks/useData";
import PlaneMap, { type FocusRequest } from "./components/PlaneMap";
import SearchBox from "./components/SearchBox";
import InfoPanel from "./components/InfoPanel";

export default function App() {
  const { planes, loading, refresh } = usePlanes();
  const { airports, error: airportsError } = useAirports();
  const [selection, setSelection] = useState<Selection | null>(null);
  const [focus, setFocus] = useState<FocusRequest | null>(null);

  const requestFocus = useCallback(
    (lat: number, lon: number, minZoom?: number) =>
      setFocus((prev) => ({ lat, lon, minZoom, nonce: (prev?.nonce ?? 0) + 1 })),
    [],
  );

  const selectPlane = useCallback(
    (plane: Plane, minZoom?: number) => {
      const last = plane.routePoints[plane.routePoints.length - 1];
      setSelection({ kind: "plane", plane });
      requestFocus(last[0], last[1], minZoom);
    },
    [requestFocus],
  );

  const selectAirport = useCallback(
    (airport: Airport) => {
      setSelection({ kind: "airport", airport });
      requestFocus(airport.lat, airport.lon);
    },
    [requestFocus],
  );

  // Only planes that have a position on the map are searchable.
  const searchOptions = useMemo(
    () =>
      planes
        .filter((p) => p.icao && p.routePoints.length > 0)
        .map((p) => String(p.icao).toUpperCase()),
    [planes],
  );

  const handleSearchSelect = (icao: string) => {
    const plane = planes.find((p) => String(p.icao).toUpperCase() === icao);
    if (plane) selectPlane(plane, 6);
  };

  // After a refresh, keep the selection only if that plane still exists (using the fresh data).
  const currentSelection = useMemo<Selection | null>(() => {
    if (selection?.kind !== "plane") return selection;
    const fresh = planes.find((p) => p.icao === selection.plane.icao);
    return fresh && fresh.routePoints.length > 0
      ? { kind: "plane", plane: fresh }
      : null;
  }, [selection, planes]);

  return (
    <div className="app bg-gray-200">
      <header className="toolbar">
        <button className="refresh-button" onClick={refresh} disabled={loading}>
          {loading ? "Atualizando..." : "Atualizar dados"}
        </button>
        <SearchBox options={searchOptions} onSelect={handleSearchSelect} />
      </header>

      {airportsError && (
        <div className="banner" role="alert">
          Não foi possível carregar airports.csv ({airportsError}). Coloque o arquivo em
          <code> public/airports.csv</code>.
        </div>
      )}

      <main className="content">
        <PlaneMap
          planes={planes}
          airports={airports}
          selection={currentSelection}
          focus={focus}
          onSelectPlane={(p) => selectPlane(p)}
          onSelectAirport={selectAirport}
        />
        <InfoPanel
          selection={currentSelection}
          airports={airports}
          refreshing={loading}
        />
      </main>
    </div>
  );
}
