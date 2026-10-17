import L from "leaflet";

export const planeGlyph = "✈";

const planeIconCache = new Map<string, L.DivIcon>();

/** Red plane rotated to its heading. Port of `_build_plane_icon`. */
export function planeIcon(heading: number, muted = false): L.DivIcon {
  const rotation = Math.round(((heading - 90) % 360 + 360) % 360);
  const cacheKey = `${rotation}-${muted}`;
  let icon = planeIconCache.get(cacheKey);
  if (!icon) {
    icon = L.divIcon({
      className: "leaflet-div-icon !border-0 !bg-transparent",
      html:
        `<div style="width:36px;height:36px;line-height:36px;text-align:center;` +
        `font-size:36px;color:#dc2626;opacity:${muted ? 0.18 : 1};` +
        `-webkit-text-stroke:2.5px #000;` +
        `paint-order:stroke fill;display:block;` +
        `transform:rotate(${rotation}deg);transform-origin:18px 18px;">${planeGlyph}</div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });
    planeIconCache.set(cacheKey, icon);
  }
  return icon;
}

/** Blue airport marker. Port of `_build_airport_icon`. */
export const airportIcon = L.divIcon({
  className: "leaflet-div-icon !border-0 !bg-transparent",
  html: `<div style="font-size:24px;color:#3b82f6;display:inline-block;">${planeGlyph}</div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});
