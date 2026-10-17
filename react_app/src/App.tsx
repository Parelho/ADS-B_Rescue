import { useCallback, useMemo, useState } from "react";
import "leaflet/dist/leaflet.css";
import "./App.css";
import type { Airport, Plane, Selection } from "./types";
import { useAirports, usePlanes } from "./hooks/useData";
import PlaneMap, { type FocusRequest } from "./components/PlaneMap";
import SearchBox from "./components/SearchBox";
import InfoPanel from "./components/InfoPanel";
import DateRangePicker from "./components/DateRangePicker";

export default function App() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const { planes, loading, refresh } = usePlanes(startDate, endDate);
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
    <div className="flex h-screen flex-col gap-2.5 bg-gray-200 p-2.5 font-[Arial,sans-serif] text-gray-800">
      <header className="grid grid-cols-[1fr_auto_1fr] items-center gap-2.5">
        <button
          className="min-h-[50px] min-w-[190px] justify-self-start rounded-[18px] border border-black bg-white px-[22px] py-3 text-[17px] font-bold text-black hover:bg-gray-100 active:bg-gray-200 disabled:cursor-default disabled:bg-gray-100 max-[800px]:min-w-0"
          onClick={refresh}
          disabled={loading}
        >
          {loading ? "Atualizando..." : "Atualizar dados"}
        </button>

        <DateRangePicker
          onApply={(start, end) => {
            setStartDate(start);
            setEndDate(end);
          }}
        />

        <SearchBox options={searchOptions} onSelect={handleSearchSelect} />
      </header>

      {airportsError && (
        <div
          className="rounded-lg border border-amber-500 bg-amber-100 px-3 py-2 text-sm"
          role="alert"
        >
          Não foi possível carregar airports.csv ({airportsError}). Coloque o arquivo em
          <code> public/airports.csv</code>.
        </div>
      )}

      <main className="flex min-h-0 flex-1 gap-2.5 max-[800px]:flex-col">
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
