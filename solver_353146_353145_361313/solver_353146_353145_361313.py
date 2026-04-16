import time
import math
import random
from copy import deepcopy

from solver_353146_353145_361313.abstract_solver import AbstractSolver


class solver_353146_353145_361313(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_353146_353145_361313"

        # 30 for testing, 590 for final; assignment allows up to 10 minutes
        self.TIME_LIMIT_SECONDS = 100.0

        self.EPS = 1e-9
        self.RNG = random.Random(42)

        self.items_df = self.inst.df_items
        self.vehicles_df = self.inst.df_vehicles

        self.items = []
        for item_id, row in self.items_df.iterrows():
            self.items.append({
                "id": item_id,
                "width": float(row["width"]),
                "depth": float(row["depth"]),
                "height": float(row["height"]),
                "weight": float(row["weight"]),
                "value": float(row["value"]),
                "allowedRotations": str(row["allowedRotations"]),
                "volume": float(row["width"]) * float(row["depth"]) * float(row["height"]),
                "base_area": float(row["width"]) * float(row["depth"]),
            })

        self.vehicle_types = []
        for vehicle_type, row in self.vehicles_df.iterrows():
            max_value = row["maxValue"]
            if max_value is None or (isinstance(max_value, float) and math.isnan(max_value)):
                max_value = float("inf")

            self.vehicle_types.append({
                "type": vehicle_type,
                "width": float(row["width"]),
                "depth": float(row["depth"]),
                "height": float(row["height"]),
                "maxWeight": float(row["maxWeight"]),
                "cost": float(row["cost"]),
                "maxValue": float(max_value),
                "gravityStrength": float(row["gravityStrength"]),
                "volume": float(row["width"]) * float(row["depth"]) * float(row["height"]),
            })

        self.vehicle_types_sorted = sorted(
            self.vehicle_types,
            key=lambda v: (v["cost"], v["volume"], v["height"])
        )

    # ------------------------------------------------------------
    # Geometry / rotations (must match results_checker.get_dims)
    # Tuple (a,b,c): extent along y, x, z respectively; checker uses
    # x2=x+b, y2=y+a, z2=z+c and bounds x by vehicle depth, y by width.
    # ------------------------------------------------------------
    def get_dims_checker(self, item, orient):
        w, d, h = item["width"], item["depth"], item["height"]
        rotations = [
            (w, d, h),
            (d, w, h),
            (h, d, w),
            (d, h, w),
            (w, h, d),
            (h, w, d),
        ]
        return rotations[int(orient)]

    def overlap_1d(self, a_min, a_max, b_min, b_max):
        return max(a_min, b_min) < min(a_max, b_max) - self.EPS

    def boxes_overlap(self, a, b):
        return (
            self.overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
            self.overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
            self.overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"])
        )

    # ------------------------------------------------------------
    # Vehicle helpers
    # ------------------------------------------------------------
    def make_empty_bin(self, vehicle_type_dict, idx_vehicle):
        return {
            "type_vehicle": vehicle_type_dict["type"],
            "idx_vehicle": idx_vehicle,
            "spec": vehicle_type_dict,
            "placed": [],
            "points": {(0.0, 0.0, 0.0)},
            "used_weight": 0.0,
            "used_value": 0.0,
        }

    def canonicalize_points(self, points):
        cleaned = set()
        for x, y, z in points:
            cleaned.add((round(x, 6), round(y, 6), round(z, 6)))
        return cleaned

    def prune_points(self, bin_state):
        spec = bin_state["spec"]
        valid = set()

        for p in self.canonicalize_points(bin_state["points"]):
            x, y, z = p

            if x < -self.EPS or y < -self.EPS or z < -self.EPS:
                continue
            if x > spec["depth"] + self.EPS:
                continue
            if y > spec["width"] + self.EPS:
                continue
            if z > spec["height"] + self.EPS:
                continue

            inside_existing = False
            for box in bin_state["placed"]:
                if (
                    box["x1"] + self.EPS < x < box["x2"] - self.EPS and
                    box["y1"] + self.EPS < y < box["y2"] - self.EPS and
                    box["z1"] + self.EPS < z < box["z2"] - self.EPS
                ):
                    inside_existing = True
                    break

            if not inside_existing:
                valid.add((x, y, z))

        bin_state["points"] = valid

    def support_area(self, box, placed_boxes):
        if box["z1"] <= self.EPS:
            return box["base_area"]

        support = 0.0
        for other in placed_boxes:
            if abs(other["z2"] - box["z1"]) > 1e-6:
                continue

            dx = max(0.0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
            dy = max(0.0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
            support += dx * dy

        return support

    def feasible_placement(self, bin_state, item, orient, point):
        spec = bin_state["spec"]
        x, y, z = point
        ext_y, ext_x, ext_z = self.get_dims_checker(item, orient)

        if x + ext_x > spec["depth"] + self.EPS:
            return None
        if y + ext_y > spec["width"] + self.EPS:
            return None
        if z + ext_z > spec["height"] + self.EPS:
            return None

        if bin_state["used_weight"] + item["weight"] > spec["maxWeight"] + self.EPS:
            return None
        if bin_state["used_value"] + item["value"] > spec["maxValue"] + self.EPS:
            return None

        box = {
            "id_item": item["id"],
            "orient": int(orient),
            "x1": x,
            "y1": y,
            "z1": z,
            "x2": x + ext_x,
            "y2": y + ext_y,
            "z2": z + ext_z,
            "width": ext_y,
            "depth": ext_x,
            "height": ext_z,
            "base_area": ext_y * ext_x,
            "volume": ext_y * ext_x * ext_z,
            "weight": item["weight"],
            "value": item["value"],
        }

        for other in bin_state["placed"]:
            if self.boxes_overlap(box, other):
                return None

        gravity = spec["gravityStrength"]
        if z > self.EPS and gravity > 0:
            support = self.support_area(box, bin_state["placed"])
            required = box["base_area"] * (gravity / 100.0)
            if support + self.EPS < required:
                return None
        else:
            support = box["base_area"]

        residual_x = spec["depth"] - box["x2"]
        residual_y = spec["width"] - box["y2"]
        residual_z = spec["height"] - box["z2"]
        support_ratio = support / max(box["base_area"], self.EPS)

        score = (
            1.2 * residual_x +
            1.2 * residual_y +
            0.25 * residual_z +
            0.10 * z -
            2.0 * support_ratio
        )

        return {"box": box, "score": score}

    def add_box_to_bin(self, bin_state, placement):
        box = placement["box"]
        bin_state["placed"].append(box)
        bin_state["used_weight"] += box["weight"]
        bin_state["used_value"] += box["value"]

        new_points = set(bin_state["points"])
        new_points.discard((round(box["x1"], 6), round(box["y1"], 6), round(box["z1"], 6)))

        candidates = [
            (box["x2"], box["y1"], box["z1"]),
            (box["x1"], box["y2"], box["z1"]),
            (box["x1"], box["y1"], box["z2"]),
        ]

        for p in candidates:
            new_points.add((round(p[0], 6), round(p[1], 6), round(p[2], 6)))

        bin_state["points"] = new_points
        self.prune_points(bin_state)

    def try_place_in_existing_bins(self, bins, item):
        best = None

        for b_idx, bin_state in enumerate(bins):
            candidate_points = sorted(
                list(bin_state["points"]),
                key=lambda p: (p[2], p[0] + p[1], p[0], p[1])
            )

            for orient_char in item["allowedRotations"]:
                orient = int(orient_char)

                for point in candidate_points:
                    placement = self.feasible_placement(bin_state, item, orient, point)
                    if placement is None:
                        continue

                    total_score = placement["score"]

                    if best is None or total_score < best["total_score"]:
                        best = {
                            "mode": "existing",
                            "bin_index": b_idx,
                            "placement": placement,
                            "total_score": total_score,
                        }

        return best

    def cheapest_new_bin_option(self, next_idx_vehicle, item):
        best = None

        for v in self.vehicle_types_sorted:
            temp_bin = self.make_empty_bin(v, next_idx_vehicle)
            best_for_vehicle = None

            for orient_char in item["allowedRotations"]:
                orient = int(orient_char)
                placement = self.feasible_placement(temp_bin, item, orient, (0.0, 0.0, 0.0))
                if placement is None:
                    continue

                total_score = 100000.0 + 1000.0 * v["cost"] + placement["score"]

                if best_for_vehicle is None or total_score < best_for_vehicle["total_score"]:
                    best_for_vehicle = {
                        "mode": "new",
                        "vehicle_spec": v,
                        "placement": placement,
                        "total_score": total_score,
                    }

            if best_for_vehicle is not None:
                if best is None or best_for_vehicle["total_score"] < best["total_score"]:
                    best = best_for_vehicle

        return best

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------
    def item_sort_key(self, item, strategy_name):
        vol = item["volume"]
        base = item["base_area"]
        max_edge = max(item["width"], item["depth"], item["height"])

        if strategy_name == "volume":
            return (-vol, -item["weight"], -base, -max_edge, item["id"])
        if strategy_name == "base":
            return (-base, -vol, -item["weight"], -max_edge, item["id"])
        if strategy_name == "height":
            return (-item["height"], -vol, -base, -item["weight"], item["id"])
        if strategy_name == "weight":
            return (-item["weight"], -vol, -base, -max_edge, item["id"])
        if strategy_name == "maxedge":
            return (-max_edge, -vol, -base, -item["weight"], item["id"])

        return (-vol, -base, item["id"])

    def build_order(self, strategy_name):
        items = self.items[:]

        if strategy_name == "shuffled_volume":
            items.sort(key=lambda x: self.item_sort_key(x, "volume"))
            self.RNG.shuffle(items)
            items = sorted(
                items,
                key=lambda x: (-(x["volume"]) + 0.05 * self.RNG.random(), -x["weight"])
            )
            return items

        items.sort(key=lambda x: self.item_sort_key(x, strategy_name))
        return items

    def solution_cost(self, bins):
        return sum(b["spec"]["cost"] for b in bins)

    def convert_bins_to_sol(self, bins):
        sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }

        for bin_state in bins:
            for box in bin_state["placed"]:
                sol["type_vehicle"].append(bin_state["type_vehicle"])
                sol["idx_vehicle"].append(bin_state["idx_vehicle"])
                sol["id_item"].append(box["id_item"])
                sol["x_origin"].append(box["x1"])
                sol["y_origin"].append(box["y1"])
                sol["z_origin"].append(box["z1"])
                sol["orient"].append(box["orient"])

        return sol

    def constructive_attempt(self, ordered_items, start_time):
        bins = []
        next_idx_vehicle = 0

        for item in ordered_items:
            if time.time() - start_time >= self.TIME_LIMIT_SECONDS:
                break

            best_existing = self.try_place_in_existing_bins(bins, item)
            best_new = self.cheapest_new_bin_option(next_idx_vehicle, item)

            if best_existing is None and best_new is None:
                return None

            if best_existing is None:
                choice = best_new
            elif best_new is None:
                choice = best_existing
            else:
                choice = best_existing if best_existing["total_score"] <= best_new["total_score"] else best_new

            if choice["mode"] == "existing":
                self.add_box_to_bin(bins[choice["bin_index"]], choice["placement"])
            else:
                new_bin = self.make_empty_bin(choice["vehicle_spec"], next_idx_vehicle)
                self.add_box_to_bin(new_bin, choice["placement"])
                bins.append(new_bin)
                next_idx_vehicle += 1

        placed_ids = set()
        for b in bins:
            for box in b["placed"]:
                placed_ids.add(box["id_item"])

        if len(placed_ids) != len(self.items):
            return None

        return bins

    def fallback_solution(self):
        bins = []
        next_idx_vehicle = 0

        for item in self.items:
            option = self.cheapest_new_bin_option(next_idx_vehicle, item)
            if option is None:
                raise RuntimeError(f"Could not place item {item['id']} in any vehicle.")

            new_bin = self.make_empty_bin(option["vehicle_spec"], next_idx_vehicle)
            self.add_box_to_bin(new_bin, option["placement"])
            bins.append(new_bin)
            next_idx_vehicle += 1

        return bins

    def solve(self):
        start_time = time.time()

        strategies = [
            "volume",
            "base",
            "height",
            "weight",
            "maxedge",
            "shuffled_volume",
        ]

        best_bins = None
        best_cost = float("inf")
        best_num_bins = float("inf")

        strat_index = 0
        while time.time() - start_time < self.TIME_LIMIT_SECONDS:
            strategy = strategies[strat_index % len(strategies)]
            strat_index += 1

            ordered_items = self.build_order(strategy)
            bins = self.constructive_attempt(ordered_items, start_time)

            if bins is None:
                continue

            cost = self.solution_cost(bins)
            num_bins = len(bins)

            if best_bins is None:
                best_bins = deepcopy(bins)
                best_cost = cost
                best_num_bins = num_bins
            else:
                if cost < best_cost - self.EPS:
                    best_bins = deepcopy(bins)
                    best_cost = cost
                    best_num_bins = num_bins
                elif abs(cost - best_cost) <= self.EPS and num_bins < best_num_bins:
                    best_bins = deepcopy(bins)
                    best_num_bins = num_bins

        if best_bins is None:
            best_bins = self.fallback_solution()

        self.sol = self.convert_bins_to_sol(best_bins)
        self.write_solution_to_file()