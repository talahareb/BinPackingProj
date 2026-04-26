"""One-off batch: run solver on every dataset folder and check feasibility."""
import os
import sys
import math
import pandas as pd
from instances import Instance
from solver_353146_353145_361313 import solver_353146_353145_361313

SOLVER_NAME = "solver_353146_353145_361313"
# solve() runs until this elapses each call.
TIME_LIMIT = 100.0

DATASETS = sorted(
    d for d in os.listdir("datasets")
    if os.path.isdir(os.path.join("datasets", d))
)


def get_dims(item, orient):
    w, d, h = item["width"], item["depth"], item["height"]
    rotations = [
        (w, d, h),
        (d, w, h),
        (h, d, w),
        (d, h, w),
        (w, h, d),
        (h, w, d),
    ]
    return rotations[orient]


def overlap_1d(a_min, a_max, b_min, b_max):
    return max(a_min, b_min) < min(a_max, b_max)


def boxes_overlap(a, b):
    return (
        overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
        overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
        overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"])
    )


def check_feasible(inst, solution_path):
    items = inst.df_items
    vehicles = inst.df_vehicles
    solution = pd.read_csv(solution_path)
    items_dict = items.to_dict("index")
    vehicles_dict = vehicles.to_dict("index")
    groups = solution.groupby("idx_vehicle")

    feasible = True
    placed_item_ids = set()

    for vidx, group in groups:
        vehicle_type = group.iloc[0]["type_vehicle"]
        vehicle = vehicles_dict[vehicle_type]
        placed_boxes = []
        total_weight = 0.0
        total_value = 0.0

        for _, row in group.iterrows():
            item_id = row["id_item"]
            placed_item_ids.add(item_id)
            item = items_dict[item_id]
            orient = int(row["orient"])
            w, d, h = get_dims(item, orient)
            x, y, z = row["x_origin"], row["y_origin"], row["z_origin"]
            box = {
                "id": item_id,
                "x1": x, "y1": y, "z1": z,
                "x2": x + d, "y2": y + w, "z2": z + h,
                "base_area": w * d,
            }
            if box["x2"] > vehicle["depth"] or box["y2"] > vehicle["width"] or box["z2"] > vehicle["height"]:
                feasible = False
            for other in placed_boxes:
                if boxes_overlap(box, other):
                    feasible = False
            placed_boxes.append(box)
            total_weight += item["weight"]
            total_value += item["value"]

        if total_weight > vehicle["maxWeight"]:
            feasible = False
        mv = vehicle["maxValue"]
        if mv is not None and not (isinstance(mv, float) and math.isnan(mv)):
            if total_value > float(mv):
                feasible = False

        gravity = vehicle["gravityStrength"]
        for i, box in enumerate(placed_boxes):
            if box["z1"] == 0:
                continue
            support_area = 0.0
            for j, other in enumerate(placed_boxes):
                if i == j:
                    continue
                if abs(other["z2"] - box["z1"]) < 1e-6:
                    dx = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                    dy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                    support_area += dx * dy
            required = box["base_area"] * (gravity / 100.0)
            if support_area < required:
                feasible = False

    all_items = set(items.index)
    missing = all_items - placed_item_ids
    if missing:
        feasible = False

    return feasible, len(missing)


def main():
    results = []
    for name in DATASETS:
        path = os.path.join("results", f"sol_{name}_{SOLVER_NAME}.csv")
        try:
            inst = Instance(name)
            sol = solver_353146_353145_361313(inst)
            sol.TIME_LIMIT_SECONDS = TIME_LIMIT
            sol.solve()
            ok, nmiss = check_feasible(inst, path)
            df = pd.read_csv(path)
            seen = set()
            cost = 0.0
            for _, r in df.iterrows():
                key = (r["type_vehicle"], r["idx_vehicle"])
                if key not in seen:
                    seen.add(key)
                    cost += inst.df_vehicles.loc[r["type_vehicle"], "cost"]
            results.append((name, ok, nmiss, cost))
            status = "OK" if ok else "FAIL"
            print(f"{status}  {name}  missing={nmiss}  cost={cost:.2f}")
        except Exception as e:
            results.append((name, False, -1, 0.0))
            print(f"ERR   {name}  {e!r}")

    failed = [r for r in results if not r[1]]
    print()
    print(f"Datasets checked: {len(results)}")
    print(f"Feasible: {sum(1 for r in results if r[1])}")
    if failed:
        print(f"Infeasible or error: {[r[0] for r in failed]}")
        sys.exit(1)
    print("All feasible.")
    sys.exit(0)


if __name__ == "__main__":
    main()
