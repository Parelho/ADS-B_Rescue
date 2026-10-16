import { describe, expect, it } from "vitest";
import { formatTimestamp } from "./format";
import { getHeading, makeCircle } from "./geo";

describe("ports match the Python implementation", () => {
  it("heading", () => {
    expect(getHeading([[-23.25, -46.7], [-22.9, -47.05]])).toBeCloseTo(317.31745127140067, 9);
    expect(getHeading([[-15.5, -47.6], [-15.84, -47.9]])).toBeCloseTo(220.30944884893216, 9);
    expect(getHeading([[1, 1]])).toBe(0);
  });
  it("circle", () => {
    const c = makeCircle(-22.4, -47.5);
    expect(c).toHaveLength(73);
    expect(c[0][0]).toBeCloseTo(-22.4, 9);
    expect(c[0][1]).toBeCloseTo(-47.41347104731234, 9);
    expect(c[18][0]).toBeCloseTo(-22.32, 9);
    expect(c[18][1]).toBeCloseTo(-47.5, 9);
  });
  it("timestamp", () => {
    expect(formatTimestamp(1760620180)).toBe("16/10/2025 13:09:40 UTC");
    expect(formatTimestamp(1760620180000)).toBe("16/10/2025 13:09:40 UTC");
    expect(formatTimestamp(null)).toBe("N/A");
    expect(formatTimestamp("abc")).toBe("abc");
  });
});
