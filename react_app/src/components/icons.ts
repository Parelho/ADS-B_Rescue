import L from "leaflet";

const planeIconCache = new Map<number, L.DivIcon>();

/** Red plane rotated to its heading. Port of `_build_plane_icon`. */
export function planeIcon(heading: number): L.DivIcon {
  const rotation = Math.round(((heading - 90) % 360 + 360) % 360);
  let icon = planeIconCache.get(rotation);
  if (!icon) {
    icon = L.divIcon({
      className: "marker-icon",
      html:
        `<div style="width:36px;height:36px;line-height:36px;text-align:center;` +
        `font-size:36px;color:#dc2626;display:block;` +
        `transform:rotate(${rotation}deg);transform-origin:18px 18px;">✈</div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });
    planeIconCache.set(rotation, icon);
  }
  return icon;
}

/** Blue airport marker. Port of `_build_airport_icon`. */
export const airportIcon = L.divIcon({
  className: "marker-icon",
  html: '<div style="font-size:24px;color:#3b82f6;display:inline-block;">✈</div>',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});
