# P1 audit clarification

P1_20261124-001 remaps original distorted THIN_PRISM_FISHEYE laser depths into undistorted PINHOLE image coordinates using ETH3D calibration. Its `center_mapping_error_px: 0` field was an analytic assertion, not an image-wide inverse remap validation. This stage numerically evaluates that projection equation at the optical center for all 25 selected frames. The maximum numerical error is 0 px at double precision. This only validates the optical-center mapping; it does not establish image-wide remap accuracy.
