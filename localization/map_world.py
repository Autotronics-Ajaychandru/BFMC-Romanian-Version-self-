"""
World / map coordinate conventions (BFMC competition scale).

Authoritative state estimation uses **GraphML node coordinates in meters**, matching
`config.REAL_WIDTH_M`, `REAL_HEIGHT_M`, and `MapEngine.to_pixel` / `to_meter`.

The SVG track is display-only in this stack: MapEngine rasterizes it; do not treat raw
SVG user units as meters without a separate calibration transform.
"""
