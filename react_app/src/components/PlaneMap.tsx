import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  Marker,
  Pane,
  Polygon,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import type { Airport, Plane, Selection } from "../types";
import { findNearbyAirport } from "../lib/airports";
import { airportIcon, planeIcon } from "./icons";

export interface FocusRequest {
  lat: number;
  lon: number;
  /** When set, zoom in to at least this level (used by the search box). */
  minZoom?: number;
  /** Changes on every request so the same target can be focused twice. */
  nonce: number;
}

const BRAZIL_CENTER: [number, number] = [-14.235, -51.925];
// Esri.WorldTopoMap (same basemap folium used)
const TILE_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}";
const TILE_ATTRIBUTION =
  "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community";
const AIRPORT_MARKER_MIN_ZOOM = 7;

function AirportMarkers({
  airports,
  onSelectAirport,
}: {
  airports: Airport[];
  onSelectAirport: (airport: Airport) => void;
}) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());

  useMapEvents({
    zoomend: () => setZoom(map.getZoom()),
  });

  if (zoom < AIRPORT_MARKER_MIN_ZOOM) return null;

  return airports.map((airport) => (
    <Marker
      key={`airport-${airport.id}`}
      position={[airport.lat, airport.lon]}
      icon={airportIcon}
      eventHandlers={{ click: () => onSelectAirport(airport) }}
    >
      <Tooltip>{airport.name}</Tooltip>
    </Marker>
  ));
}

function MapController({ focus }: { focus: FocusRequest | null }) {
  const map = useMap();
  useEffect(() => {
    if (!focus) return;
    const zoom = focus.minZoom
      ? Math.max(map.getZoom(), focus.minZoom)
      : map.getZoom();
    map.setView([focus.lat, focus.lon], zoom, { animate: true, duration: 0.7 });
  }, [focus, map]);
  return null;
}

function EmptyMapClickHandler({ onClear }: { onClear: () => void }) {
  useMapEvents({
    click(event) {
      const target = event.originalEvent.target;
      if (
        target instanceof Element &&
        target.closest(".leaflet-marker-icon, .leaflet-interactive")
      ) {
        return;
      }
      onClear();
    },
  });
  return null;
}

interface Props {
  planes: Plane[];
  airports: Airport[];
  selection: Selection | null;
  focus: FocusRequest | null;
  onSelectPlane: (plane: Plane) => void;
  onSelectAirport: (airport: Airport) => void;
  onClearSelection: () => void;
}

export default function PlaneMap({
  planes,
  airports,
  selection,
  focus,
  onSelectPlane,
  onSelectAirport,
  onClearSelection,
}: Props) {
  const plottable = useMemo(
    () => planes.filter((p) => p.routePoints.length > 0),
    [planes],
  );

  // One blue airport marker per distinct origin airport (nearest to each route's first point).
  const originAirports = useMemo(() => {
    const seen = new Map<string, Airport>();
    for (const p of plottable) {
      const [lat, lon] = p.routePoints[0];
      const a = findNearbyAirport(airports, lat, lon);
      if (a) seen.set(a.id, a);
    }
    return [...seen.values()];
  }, [plottable, airports]);

  const selectedPlane = selection?.kind === "plane" ? selection.plane : null;

  return (
    <MapContainer
      center={BRAZIL_CENTER}
      zoom={4}
      className="h-full min-w-0 flex-1 rounded-[10px] max-[800px]:min-h-0"
      worldCopyJump
    >
      <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
      <MapController focus={focus} />
      <EmptyMapClickHandler onClear={onClearSelection} />

      <Pane name="airportMarkers" style={{ zIndex: 610 }}>
        <AirportMarkers
          airports={originAirports}
          onSelectAirport={onSelectAirport}
        />
      </Pane>

      <Pane name="planeMarkers" style={{ zIndex: 620 }}>
        {plottable.map((p) => {
          const last = p.routePoints[p.routePoints.length - 1];
          return (
            <Marker
              key={`plane-${p.icao}`}
              position={last}
              icon={planeIcon(
                p.heading,
                selectedPlane !== null && selectedPlane.icao !== p.icao,
              )}
              eventHandlers={{ click: () => onSelectPlane(p) }}
            >
              <Tooltip>{p.icao}</Tooltip>
            </Marker>
          );
        })}
      </Pane>

      {selectedPlane && (
        <>
          <Polyline
            positions={selectedPlane.routePoints}
            pathOptions={{ color: "blue", weight: 5, opacity: 0.8 }}
          />
          {selectedPlane.predPoints.length > 0 && (
            <Polyline
              positions={selectedPlane.predPoints}
              pathOptions={{
                color: "#6b7280",
                weight: 3,
                opacity: 0.9,
                dashArray: "6, 8",
              }}
            />
          )}
          {selectedPlane.searchArea && (
            <Polygon
              positions={selectedPlane.searchArea}
              pathOptions={{
                color: "#3b82f6",
                fillColor: "#3b82f6",
                fillOpacity: 0.15,
                weight: 2,
              }}
            />
          )}
        </>
      )}
    </MapContainer>
  );
}
