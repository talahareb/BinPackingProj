import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

from solver_000000.abstract_solver import AbstractSolver


@dataclass
class Placement:
    item_id: str
    orient: str
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    weight: float
    value: float


class solver_000000(AbstractSolver):
    """
    Corner-based constructive 3D packing heuristic.

    Idea:
    - Sort items by difficulty (bigger / heavier / more restrictive first).
    - Try to place each item into existing opened bins first.
    - If not possible, open the cheapest feasible new bin.
    - Candidate positions are only corners:
        (0,0,0), and the right/front/top corners of already placed items.
    - Enforce:
        * inside bounds
        * no overlap
        * bin maxWeight
        * bin maxValue (if defined)
        * allowed rotations
        * gravityStrength support ratio
    """

    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_000000"

        # Normalize vehicles into a sorted table:
        # x-axis = depth, y-axis = width, z-axis = height
        self.vehicles = []
        for vtype, row in self.inst.df_vehicles.iterrows():
            self.vehicles.append(
                {
                    "type": vtype,
                    "W": float(row["width"]),   # along y
                    "D": float(row["depth"]),   # along x
                    "H": float(row["height"]),  # along z
                    "maxWeight": float(row["maxWeight"]),
                    "cost": float(row["cost"]),
                    "maxValue": None if pd.isna(row.get("maxValue", None)) else float(row["maxValue"]),
                    "gravityStrength": float(row.get("gravityStrength", 0.0)),
                }
            )

        # Sort vehicles by cheaper first, then volume ascending
        self.vehicles.sort(key=lambda v: (v["cost"], v["W"] * v["D"] * v["H"]))

        # Opened bins
        self.open_bins: List[Dict[str, Any]] = []
        self.next_bin_idx = 0

    # ---------------------------
    # Main solve
    # ---------------------------
    def solve(self):
        items = self._prepare_items()

        for item in items:
            placed = self._place_item(item)
            if not placed:
                raise RuntimeError(f"Could not place item {item['id']} in any vehicle.")

        self.write_solution_to_file()

    # ---------------------------
    # Item preparation / sorting
    # ---------------------------
    def _prepare_items(self) -> List[Dict[str, Any]]:
        items = []

        for item_id, row in self.inst.df_items.iterrows():
            w = float(row["width"])
            d = float(row["depth"])
            h = float(row["height"])
            weight = float(row["weight"])
            value = float(row["value"])
            allowed = str(row["allowedRotations"])

            volume = w * d * h
            base_area = max(w * d, w * h, d * h)

            items.append(
                {
                    "id": item_id,
                    "width": w,
                    "depth": d,
                    "height": h,
                    "weight": weight,
                    "value": value,
                    "allowed": allowed,
                    "volume": volume,
                    "base_area": base_area,
                    "rot_count": len(set(allowed)),
                }
            )

        # Harder items first:
        # bigger volume, then heavier, then fewer allowed rotations
        items.sort(
            key=lambda it: (
                -it["volume"],
                -it["weight"],
                it["rot_count"],
                -it["base_area"],
            )
        )
        return items

    # ---------------------------
    # Placement workflow
    # ---------------------------
    def _place_item(self, item: Dict[str, Any]) -> bool:
        best_existing = None

        # 1) Try existing bins first
        for bin_obj in self.open_bins:
            candidate = self._best_placement_in_bin(item, bin_obj, prefer_existing=True)
            if candidate is not None:
                if best_existing is None or candidate["score"] < best_existing["score"]:
                    best_existing = candidate

        if best_existing is not None:
            self._commit(best_existing)
            return True

        # 2) Otherwise open a new cheapest feasible bin
        best_new = None
        for vehicle in self.vehicles:
            new_bin = self._make_empty_bin(vehicle)
            candidate = self._best_placement_in_bin(item, new_bin, prefer_existing=False)
            if candidate is not None:
                if best_new is None or candidate["score"] < best_new["score"]:
                    best_new = candidate

        if best_new is not None:
            # actually open that bin
            real_bin = self._make_empty_bin(best_new["bin"]["vehicle"])
            self.open_bins.append(real_bin)
            best_new["bin"] = real_bin
            self._commit(best_new)
            return True

        return False

    def _best_placement_in_bin(
        self,
        item: Dict[str, Any],
        bin_obj: Dict[str, Any],
        prefer_existing: bool,
    ) -> Optional[Dict[str, Any]]:
        best = None
        candidates = self._candidate_positions(bin_obj)

        seen_positions = set()
        for x, y, z in candidates:
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key in seen_positions:
                continue
            seen_positions.add(key)

            for orient, (dx, dy, dz) in self._allowed_orientations(item):
                if not self._fits_inside(bin_obj, x, y, z, dx, dy, dz):
                    continue
                if not self._respects_bin_capacity(bin_obj, item):
                    continue
                if self._overlaps(bin_obj, x, y, z, dx, dy, dz):
                    continue
                if not self._supported(bin_obj, x, y, z, dx, dy, dz):
                    continue

                score = self._score_placement(
                    bin_obj=bin_obj,
                    x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz,
                    prefer_existing=prefer_existing
                )

                cand = {
                    "bin": bin_obj,
                    "item": item,
                    "orient": orient,
                    "x": x,
                    "y": y,
                    "z": z,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "score": score,
                }

                if best is None or cand["score"] < best["score"]:
                    best = cand

        return best

    # ---------------------------
    # Bin helpers
    # ---------------------------
    def _make_empty_bin(self, vehicle: Dict[str, Any]) -> Dict[str, Any]:
        return {
        "vehicle": vehicle,
        "placements": [],
        "total_weight": 0.0,
        "total_value": 0.0,
    }

    def _respects_bin_capacity(self, bin_obj: Dict[str, Any], item: Dict[str, Any]) -> bool:
        vehicle = bin_obj["vehicle"]

        if bin_obj["total_weight"] + item["weight"] > vehicle["maxWeight"] + 1e-9:
            return False

        if vehicle["maxValue"] is not None:
            if bin_obj["total_value"] + item["value"] > vehicle["maxValue"] + 1e-9:
                return False

        return True

    def _fits_inside(self, bin_obj, x, y, z, dx, dy, dz) -> bool:
        v = bin_obj["vehicle"]
        return (
            x >= -1e-9 and y >= -1e-9 and z >= -1e-9 and
            x + dx <= v["D"] + 1e-9 and
            y + dy <= v["W"] + 1e-9 and
            z + dz <= v["H"] + 1e-9
        )

    def _overlaps(self, bin_obj, x, y, z, dx, dy, dz) -> bool:
        for p in bin_obj["placements"]:
            if (
                x < p.x + p.dx - 1e-9 and x + dx > p.x + 1e-9 and
                y < p.y + p.dy - 1e-9 and y + dy > p.y + 1e-9 and
                z < p.z + p.dz - 1e-9 and z + dz > p.z + 1e-9
            ):
                return True
        return False

    # ---------------------------
    # Gravity / support
    # ---------------------------
    def _supported(self, bin_obj, x, y, z, dx, dy, dz) -> bool:
        # If item touches the floor, it is fully supported
        if z <= 1e-9:
            return True

        required_ratio = bin_obj["vehicle"]["gravityStrength"] / 100.0
        if required_ratio <= 1e-12:
            return True

        base_area = dx * dy
        supported_area = 0.0

        # Only objects exactly below at height z can support
        for p in bin_obj["placements"]:
            top_z = p.z + p.dz
            if abs(top_z - z) > 1e-9:
                continue

            x_overlap = max(0.0, min(x + dx, p.x + p.dx) - max(x, p.x))
            y_overlap = max(0.0, min(y + dy, p.y + p.dy) - max(y, p.y))
            supported_area += x_overlap * y_overlap

        ratio = supported_area / base_area if base_area > 0 else 1.0
        return ratio + 1e-9 >= required_ratio

    # ---------------------------
    # Candidate corner positions
    # ---------------------------
    def _candidate_positions(self, bin_obj) -> List[Tuple[float, float, float]]:
        pts = [(0.0, 0.0, 0.0)]

        for p in bin_obj["placements"]:
            pts.append((p.x + p.dx, p.y, p.z))      # right
            pts.append((p.x, p.y + p.dy, p.z))      # front / side
            pts.append((p.x, p.y, p.z + p.dz))      # top

        # Sort low and near origin first
        pts = sorted(set((round(x, 6), round(y, 6), round(z, 6)) for x, y, z in pts),
                     key=lambda t: (t[2], t[0], t[1]))
        return pts

    # ---------------------------
    # Rotations
    # ---------------------------
    def _allowed_orientations(self, item: Dict[str, Any]) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Return dimensions as (dx, dy, dz) where:
        - dx along container x = depth
        - dy along container y = width
        - dz along container z = height

        Starting from item (width, depth, height):
        no-rotation maps to (depth, width, height)
        """
        w = item["width"]
        d = item["depth"]
        h = item["height"]

        orientation_map = {
            "0": (d, w, h),
            "1": (w, d, h),
            "2": (d, h, w),
            "3": (h, d, w),
            "4": (h, w, d),
            "5": (w, h, d),
        }

        out = []
        seen_dims = set()

        for code in item["allowed"]:
            if code not in orientation_map:
                continue
            dims = orientation_map[code]
            key = tuple(round(v, 9) for v in dims)
            if key not in seen_dims:
                seen_dims.add(key)
                out.append((code, dims))

        return out

    # ---------------------------
    # Scoring
    # ---------------------------
    def _score_placement(self, bin_obj, x, y, z, dx, dy, dz, prefer_existing: bool):
        v = bin_obj["vehicle"]

        # Primary priority:
        # - if existing bin, strongly preferred over opening new cost
        # - if new bin, cheaper vehicle is better
        new_bin_penalty = 0.0 if prefer_existing else v["cost"] * 1_000_000.0

        # Then pack low and compact
        height_penalty = z * 10_000.0
        compact_penalty = (x + dx) + (y + dy) + (z + dz) * 100.0

        # Slightly prefer smaller leftover volume
        leftover = (v["D"] - (x + dx)) + (v["W"] - (y + dy)) + (v["H"] - (z + dz))

        return new_bin_penalty + height_penalty + compact_penalty + leftover

    # ---------------------------
    # Commit chosen placement
    # ---------------------------
    def _commit(self, cand: Dict[str, Any]):
        bin_obj = cand["bin"]
        item = cand["item"]

        # If this is a truly new bin not yet indexed in self.open_bins, index it now
        if "assigned_idx" not in bin_obj:
            bin_obj["assigned_idx"] = self.next_bin_idx
            self.next_bin_idx += 1

        placement = Placement(
            item_id=item["id"],
            orient=cand["orient"],
            x=float(cand["x"]),
            y=float(cand["y"]),
            z=float(cand["z"]),
            dx=float(cand["dx"]),
            dy=float(cand["dy"]),
            dz=float(cand["dz"]),
            weight=float(item["weight"]),
            value=float(item["value"]),
        )

        bin_obj["placements"].append(placement)
        bin_obj["total_weight"] += item["weight"]
        bin_obj["total_value"] += item["value"]

        self.sol["type_vehicle"].append(bin_obj["vehicle"]["type"])
        self.sol["idx_vehicle"].append(bin_obj["assigned_idx"])
        self.sol["id_item"].append(item["id"])
        self.sol["x_origin"].append(placement.x)
        self.sol["y_origin"].append(placement.y)
        self.sol["z_origin"].append(placement.z)
        self.sol["orient"].append(placement.orient)