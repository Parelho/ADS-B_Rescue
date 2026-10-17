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

/** Small white air traffic control tower inside a blue map pin. */
export const airportIcon = L.divIcon({
  className: "leaflet-div-icon !border-0 !bg-transparent",
  html:
    `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="28" viewBox="0 0 32 40" ` +
    `aria-hidden="true" style="display:block;filter:drop-shadow(0 1px 2px #0005)">` +
    `<path d="M16 1C7.7 1 1 7.7 1 16c0 6.5 7.8 15.7 13.8 21.5Q16 39 16.6 37.5C23.2 31.7 31 22.5 31 16 ` +
    `C31 7.7 24.3 1 16 1Z" fill="#3b82f6" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/>` +
    `<g fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">` +
    `<path d="M10 10h12v4H10zM13.5 14v9h5v-9z"/>` +
    `<path d="M14 10v4m4-4v4"/>` +
    `</g>` +
    `</svg>`,
  iconSize: [22, 28],
  iconAnchor: [11, 26],
});
