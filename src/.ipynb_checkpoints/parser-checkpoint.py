"""
SC2 replay parser — converts a .SC2Replay file into structured DataFrames
suitable for feature engineering and model training.

Public API
----------
load_replay(path)                          -> sc2reader.Replay
extract_tracker_rows(replay)               -> list[dict]   raw tracker events
build_snapshot_df(replay, interval=30)     -> pd.DataFrame per-player time-series
build_event_features(replay)               -> dict         event-based features (timings, tech)
build_checkpoint_row(replay, checkpoint_sec, snap, event_feats) -> dict  one ML row
build_all_checkpoints(replay, checkpoints) -> list[dict]   one ML row per checkpoint

Feature categories
------------------
1. Snapshot stats     — minerals, vespene, workers, army at each checkpoint
2. Cumulative totals  — resources collected up to checkpoint (integral)
3. Rate-of-change     — how fast each stat is growing (slope over last 2 min)
4. Expansion timing   — when natural / third base started
5. Tech path timing   — when key tech buildings were placed
6. Combat signal      — army supply lost per minute up to checkpoint
"""

import sc2reader
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_replay(path: str | Path) -> sc2reader.resources.Replay:
    """Load and return an sc2reader Replay object."""
    return sc2reader.load_replay(str(path))


def _player_index(replay) -> dict:
    """Return {player_id: player} for human players only."""
    return {p.pid: p for p in replay.players}


def _get_winner(replay) -> int | None:
    """Return pid of the winning player, or None if unknown."""
    for p in replay.players:
        if hasattr(p, "result") and p.result == "Win":
            return p.pid
    return None


# ---------------------------------------------------------------------------
# Town-hall unit names — used to detect expansion timing
# ---------------------------------------------------------------------------

TOWN_HALLS = {
    "CommandCenter", "OrbitalCommand", "PlanetaryFortress",
    "Nexus",
    "Hatchery", "Lair", "Hive",
}

# Tech buildings we care about timing for, grouped by race
TECH_BUILDINGS = {
    # Terran
    "Barracks":        "terran_barracks",
    "Factory":         "terran_factory",
    "Starport":        "terran_starport",
    "EngineeringBay":  "terran_engbay",
    "Armory":          "terran_armory",
    "GhostAcademy":    "terran_ghost_academy",
    # Protoss
    "CyberneticsCore": "protoss_cyber",
    "RoboticsFacility":"protoss_robo",
    "Stargate":        "protoss_stargate",
    "TwilightCouncil": "protoss_twilight",
    "TemplarArchive":  "protoss_templar",
    "DarkShrine":      "protoss_dark_shrine",
    "RoboticsBay":     "protoss_robo_bay",
    "FleetBeacon":     "protoss_fleet_beacon",
    "Forge":           "protoss_forge",
    # Zerg
    "SpawningPool":    "zerg_pool",
    "RoachWarren":     "zerg_roach_warren",
    "BanelingNest":    "zerg_baneling_nest",
    "HydraliskDen":    "zerg_hydra_den",
    "Lair":            "zerg_lair",
    "Hive":            "zerg_hive",
    "Spire":           "zerg_spire",
    "InfestationPit":  "zerg_infestation_pit",
    "UltraliskCavern": "zerg_ultra_cavern",
    "EvolutionChamber":"zerg_evo_chamber",
}

# Worker unit names per race
WORKER_NAMES = {"SCV", "Probe", "Drone"}

# Army unit names to exclude (non-combat)
NON_ARMY = WORKER_NAMES | TOWN_HALLS | {
    "MULE", "Larva", "Egg", "Cocoon", "ChangelingMarine",
    "ChangelingZealot", "ChangelingZergling",
    "AdeptPhaseShift", "ParasiticBombDummy",
}

# ---------------------------------------------------------------------------
# Upgrade names → (category, level)
# ---------------------------------------------------------------------------

WEAPONS_UPGRADES: dict[str, int] = {
    f"TerranInfantryWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"TerranVehicleWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"TerranShipWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"ProtossGroundWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"ProtossAirWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"ZergMeleeWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"ZergMissileWeaponsLevel{i}": i for i in range(1, 4)
} | {
    f"ZergFlyerWeaponsLevel{i}": i for i in range(1, 4)
}

ARMOR_UPGRADES: dict[str, int] = {
    f"TerranInfantryArmorsLevel{i}": i for i in range(1, 4)
} | {
    f"TerranVehicleAndShipArmorsLevel{i}": i for i in range(1, 4)
} | {
    f"ProtossGroundArmorsLevel{i}": i for i in range(1, 4)
} | {
    f"ProtossAirArmorsLevel{i}": i for i in range(1, 4)
} | {
    f"ProtossShieldsLevel{i}": i for i in range(1, 4)
} | {
    f"ZergGroundArmorsLevel{i}": i for i in range(1, 4)
} | {
    f"ZergFlyerArmorsLevel{i}": i for i in range(1, 4)
}

# ---------------------------------------------------------------------------
# Unit composition categories
# ---------------------------------------------------------------------------

ARMY_UNIT_CATEGORIES: dict[str, str] = {
    # Terran
    "Marine": "bio", "Marauder": "bio", "Ghost": "bio", "Reaper": "bio",
    "SiegeTank": "mech", "SiegeTankSieged": "mech",
    "Thor": "mech", "ThorHighImpactMode": "mech",
    "Hellion": "mech", "Hellbat": "mech", "Cyclone": "mech", "WidowMine": "mech",
    "Viking": "air", "VikingAssault": "air", "Banshee": "air",
    "Raven": "air", "Battlecruiser": "air",
    "Liberator": "air", "LiberatorAG": "air", "Medivac": "air",
    # Protoss
    "Zealot": "gateway", "Stalker": "gateway", "Sentry": "gateway",
    "Adept": "gateway", "HighTemplar": "gateway", "DarkTemplar": "gateway", "Archon": "gateway",
    "Immortal": "robo", "Colossus": "robo", "Disruptor": "robo", "WarpPrism": "robo",
    "Phoenix": "stargate", "VoidRay": "stargate", "Oracle": "stargate",
    "Carrier": "stargate", "Tempest": "stargate",
    # Zerg
    "Zergling": "ling_bane", "Baneling": "ling_bane",
    "Roach": "roach_ravager", "Ravager": "roach_ravager",
    "Hydralisk": "hydra_lurker", "LurkerMP": "hydra_lurker", "LurkerMPBurrowed": "hydra_lurker",
    "Mutalisk": "muta_corr", "Corruptor": "muta_corr", "BroodLord": "muta_corr",
    "Ultralisk": "ultra",
    "Infestor": "caster", "SwarmHost": "caster", "Viper": "caster",
    "Queen": "queen",
}

ALL_COMPOSITION_KEYS: list[str] = [
    "bio", "mech", "air",                                          # Terran
    "gateway", "robo", "stargate",                                 # Protoss
    "ling_bane", "roach_ravager", "hydra_lurker",                  # Zerg ground
    "muta_corr", "ultra", "caster", "queen",                       # Zerg other
]


# ---------------------------------------------------------------------------
# Tracker event → tidy rows  (economy snapshots)
# ---------------------------------------------------------------------------

def extract_tracker_rows(replay) -> list[dict]:
    """Walk PlayerStatsEvents and return a flat list of economy snapshot dicts."""
    rows = []
    for event in replay.tracker_events:
        if not hasattr(event, "player") or event.player is None:
            continue
        if type(event).__name__ != "PlayerStatsEvent":
            continue

        pid       = event.player.pid
        workers   = getattr(event, "workers_active_count", None)
        food_used = getattr(event, "food_used", None)
        army      = (food_used - workers
                     if food_used is not None and workers is not None
                     else None)

        rows.append({
            "second":                   event.second,
            "player_id":                pid,
            "event_type":               "PlayerStatsEvent",
            "minerals_current":         getattr(event, "minerals_current", None),
            "vespene_current":          getattr(event, "vespene_current", None),
            "minerals_collection_rate": getattr(event, "minerals_collection_rate", None),
            "vespene_collection_rate":  getattr(event, "vespene_collection_rate", None),
            "workers_active":           workers,
            "army_supply":              army,
            "supply_used":              food_used,
            "supply_cap":               getattr(event, "food_made", None),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-player time-series snapshots (regular grid)
# ---------------------------------------------------------------------------

def build_snapshot_df(replay, interval: int = 30) -> pd.DataFrame:
    """
    Resample PlayerStatsEvents onto a regular time grid (every `interval` seconds).
    Returns a DataFrame indexed by (second, player_id).
    """
    rows = extract_tracker_rows(replay)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop(columns=["event_type"])

    stat_cols = [c for c in df.columns if c not in ("second", "player_id")]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    game_end = int(replay.real_length.total_seconds())
    grid     = list(range(0, game_end + interval, interval))

    results = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("second").groupby("second").last()
        grp = grp.reindex(grp.index.union(grid)).interpolate(method="index").reindex(grid)
        grp["second"]    = grid
        grp["player_id"] = pid
        results.append(grp.reset_index(drop=True))

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Event-based features  (timings, tech path, combat)
# ---------------------------------------------------------------------------

def build_event_features(replay) -> dict:
    """
    Extract event-based features for both players from a replay.

    Returns a flat dict with keys like:
        p1_natural_sec, p1_third_sec,
        p1_terran_factory_sec, p2_protoss_cyber_sec, ...
        p1_army_lost_per_min, p2_army_lost_per_min,
        p1_worker_lost_count, p2_worker_lost_count

    All timing values are in seconds; NaN if that event never occurred.
    """
    players    = list(_player_index(replay).values())
    game_end   = int(replay.real_length.total_seconds())

    # --- Expansion tracking ---
    # Town-hall locations born/init at t=0 are the starting bases.
    # We track location to distinguish main from natural/third.
    starting_locations: dict[int, list] = {p.pid: [] for p in players}
    expansion_times:    dict[int, list] = {p.pid: [] for p in players}

    # --- Tech timing: first occurrence of each building per player ---
    tech_times: dict[int, dict] = {p.pid: {} for p in players}

    # --- Army lost and worker lost (from UnitDiedEvent) ---
    army_lost:   dict[int, list] = {p.pid: [] for p in players}  # list of (second,)
    worker_lost: dict[int, int]  = {p.pid: 0 for p in players}

    # --- Upgrade tracking ---
    upgrade_events: dict[int, list] = {p.pid: [] for p in players}  # (second, cat, level)

    # --- Unit composition (births per category) ---
    unit_births: dict[int, dict] = {
        p.pid: {cat: [] for cat in ALL_COMPOSITION_KEYS} for p in players
    }

    for event in replay.tracker_events:
        etype = type(event).__name__
        t     = event.second

        # --- Town halls: track expansions ---
        if etype in ("UnitBornEvent", "UnitInitEvent"):
            unit  = event.unit
            owner = getattr(unit, "owner", None)
            if owner is None:
                continue
            pid = owner.pid
            if pid not in expansion_times:
                continue

            name = unit.name
            loc  = getattr(unit, "location", None)

            if name in TOWN_HALLS and loc is not None:
                if t == 0:
                    starting_locations[pid].append(loc)
                else:
                    expansion_times[pid].append((t, loc))

            # Tech buildings — first time only
            if name in TECH_BUILDINGS:
                key = TECH_BUILDINGS[name]
                if key not in tech_times[pid]:
                    tech_times[pid][key] = t

            # Unit composition — count army units born per category
            if etype == "UnitBornEvent":
                cat = ARMY_UNIT_CATEGORIES.get(name)
                if cat and pid in unit_births:
                    unit_births[pid][cat].append(t)

        # --- Units died: army and worker losses ---
        if etype == "UnitDiedEvent":
            unit  = event.unit
            owner = getattr(unit, "owner", None)
            if owner is None:
                continue
            pid = owner.pid
            if pid not in army_lost:
                continue
            name = unit.name
            if name in WORKER_NAMES:
                worker_lost[pid] += 1
            elif name not in NON_ARMY and not name.startswith("Mineral") \
                    and not name.startswith("Vespene") \
                    and not name.startswith("Destructible") \
                    and not name.startswith("Beacon") \
                    and not name.startswith("Unbuildable"):
                army_lost[pid].append(t)

        # --- Upgrades ---
        if etype == "UpgradeCompleteEvent":
            player = getattr(event, "player", None)
            if player is None:
                continue
            pid   = player.pid
            uname = getattr(event, "upgrade_type_name", "")
            if pid in upgrade_events:
                if uname in WEAPONS_UPGRADES:
                    upgrade_events[pid].append((t, "weapons", WEAPONS_UPGRADES[uname]))
                elif uname in ARMOR_UPGRADES:
                    upgrade_events[pid].append((t, "armor", ARMOR_UPGRADES[uname]))

    # Build flat feature dict
    result: dict = {}

    for p in players:
        pid   = p.pid
        label = f"p{pid}"

        # --- Expansion timing ---
        # Sort by time; natural = 1st expansion, third = 2nd
        exps = sorted(expansion_times[pid], key=lambda x: x[0])
        result[f"{label}_natural_sec"] = exps[0][0] if len(exps) >= 1 else np.nan
        result[f"{label}_third_sec"]   = exps[1][0] if len(exps) >= 2 else np.nan

        # --- Tech timings ---
        for key in TECH_BUILDINGS.values():
            result[f"{label}_{key}_sec"] = tech_times[pid].get(key, np.nan)

        # --- Army lost per minute (over full game) ---
        if army_lost[pid] and game_end > 0:
            result[f"{label}_army_lost_per_min"] = len(army_lost[pid]) / (game_end / 60)
        else:
            result[f"{label}_army_lost_per_min"] = 0.0

        # --- Worker deaths ---
        result[f"{label}_worker_lost_count"] = worker_lost[pid]

        # --- Army loss timings as a list (used in checkpoint function) ---
        result[f"_army_lost_times_{pid}"] = army_lost[pid]  # internal, prefixed _

        # --- Upgrade events (used in checkpoint function) ---
        result[f"_upgrade_events_{pid}"] = upgrade_events[pid]

        # --- Unit births per composition category ---
        result[f"_unit_births_{pid}"] = unit_births[pid]

    return result


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

STAT_COLS = [
    "minerals_current",
    "vespene_current",
    "minerals_collection_rate",
    "vespene_collection_rate",
    "workers_active",
    "army_supply",
]


def _snap_at(snap: pd.DataFrame, pid: int, target_sec: int, col: str) -> float:
    """Value of `col` for `pid` at `target_sec`. NaN if game ended before checkpoint."""
    sub = snap[snap["player_id"] == pid]
    if sub.empty:
        return np.nan
    if sub["second"].max() < target_sec - 30:
        return np.nan
    idx = (sub["second"] - target_sec).abs().idxmin()
    return sub.loc[idx, col]


def _integral_up_to(snap: pd.DataFrame, pid: int, target_sec: int, col: str) -> float:
    """Area under `col` curve from 0 to `target_sec`, in units-per-minute."""
    sub = snap[(snap["player_id"] == pid) & (snap["second"] <= target_sec)].sort_values("second")
    if len(sub) < 2:
        return np.nan
    return np.trapezoid(sub[col].fillna(0), sub["second"]) / 60


def _rate_of_change(snap: pd.DataFrame, pid: int, target_sec: int,
                    col: str, window_sec: int = 120) -> float:
    """
    Slope of `col` over the last `window_sec` seconds before `target_sec`.
    Computed as (value_now - value_then) / window_minutes.
    Positive = stat is growing; negative = stat is falling.
    """
    sub = snap[snap["player_id"] == pid].sort_values("second")
    if sub.empty:
        return np.nan
    now_rows  = sub[sub["second"] <= target_sec]
    past_rows = sub[sub["second"] <= max(0, target_sec - window_sec)]
    if now_rows.empty or past_rows.empty:
        return np.nan
    val_now  = now_rows.iloc[-1][col]
    val_past = past_rows.iloc[-1][col]
    if pd.isna(val_now) or pd.isna(val_past):
        return np.nan
    return (val_now - val_past) / (window_sec / 60)


def _army_lost_up_to(army_lost_times: list, target_sec: int) -> int:
    """Count army unit deaths that occurred up to `target_sec`."""
    return sum(1 for t in army_lost_times if t <= target_sec)


# ---------------------------------------------------------------------------
# Feature engineering — one row per (game, checkpoint)
# ---------------------------------------------------------------------------

def build_checkpoint_row(
    replay,
    checkpoint_sec: int,
    snap: pd.DataFrame | None = None,
    event_feats: dict | None = None,
) -> dict:
    """
    Build one ML-ready feature dict for a single point in time.

    Pass `snap` and `event_feats` when calling in a loop to avoid re-parsing.
    Returns {} if the game ended before this checkpoint or is not a 1v1.
    """
    players = list(_player_index(replay).values())
    if len(players) != 2:
        return {}

    game_length = int(replay.real_length.total_seconds())
    if game_length < checkpoint_sec - 30:
        return {}

    if snap is None:
        snap = build_snapshot_df(replay, interval=30)
    if event_feats is None:
        event_feats = build_event_features(replay)

    snap_window = snap[snap["second"] <= checkpoint_sec]

    row: dict = {
        "replay_file":    Path(replay.filename).name,
        "game_length_s":  game_length,
        "map_name":       replay.map_name,
        "checkpoint_sec": checkpoint_sec,
        "game_minute":    checkpoint_sec / 60,
        "p1_race":        players[0].play_race,
        "p2_race":        players[1].play_race,
        "winner":         _get_winner(replay),
    }

    for p in players:
        pid   = p.pid
        label = f"p{pid}"

        # 1. Snapshot stats at checkpoint
        for col in STAT_COLS:
            row[f"{label}_{col}"] = _snap_at(snap_window, pid, checkpoint_sec, col)

        # 2. Cumulative resource totals
        row[f"{label}_minerals_total"] = _integral_up_to(
            snap_window, pid, checkpoint_sec, "minerals_collection_rate")
        row[f"{label}_vespene_total"]  = _integral_up_to(
            snap_window, pid, checkpoint_sec, "vespene_collection_rate")

        # 3. Rate-of-change over last 2 minutes
        for col in ["workers_active", "army_supply", "minerals_collection_rate"]:
            row[f"{label}_{col}_roc"] = _rate_of_change(
                snap_window, pid, checkpoint_sec, col, window_sec=120)

        # 4. Expansion timing — seconds since natural/third started (negative = not yet)
        nat  = event_feats.get(f"{label}_natural_sec", np.nan)
        thrd = event_feats.get(f"{label}_third_sec",   np.nan)
        row[f"{label}_natural_age"]  = (checkpoint_sec - nat)  if not np.isnan(nat)  else np.nan
        row[f"{label}_third_age"]    = (checkpoint_sec - thrd) if not np.isnan(thrd) else np.nan
        row[f"{label}_has_natural"]  = int(not np.isnan(nat)  and nat  <= checkpoint_sec)
        row[f"{label}_has_third"]    = int(not np.isnan(thrd) and thrd <= checkpoint_sec)

        # 5. Tech path — has each building appeared yet? (binary)
        for tech_key in TECH_BUILDINGS.values():
            t_built = event_feats.get(f"{label}_{tech_key}_sec", np.nan)
            row[f"{label}_{tech_key}"] = int(
                not np.isnan(t_built) and t_built <= checkpoint_sec)

        # 6. Combat signal — army units lost up to this checkpoint
        lost_times = event_feats.get(f"_army_lost_times_{pid}", [])
        lost_count = _army_lost_up_to(lost_times, checkpoint_sec)
        elapsed_min = checkpoint_sec / 60
        row[f"{label}_army_lost_count"]   = lost_count
        row[f"{label}_army_lost_per_min"] = lost_count / elapsed_min if elapsed_min > 0 else 0.0
        row[f"{label}_worker_lost_count"] = event_feats.get(f"{label}_worker_lost_count", 0)

        # 7. Spending efficiency — floating resources relative to income
        #    High value = sitting on unspent resources (bad macro)
        m_curr = _snap_at(snap_window, pid, checkpoint_sec, "minerals_current") or 0.0
        v_curr = _snap_at(snap_window, pid, checkpoint_sec, "vespene_current") or 0.0
        m_rate = _snap_at(snap_window, pid, checkpoint_sec, "minerals_collection_rate") or 0.0
        v_rate = _snap_at(snap_window, pid, checkpoint_sec, "vespene_collection_rate") or 0.0
        total_income = max(1.0, m_rate + v_rate)
        row[f"{label}_spending_quotient"] = (m_curr + v_curr) / total_income

        # 8. Supply blocks — fraction of 30s snapshot intervals where supply was capped
        sub_p = snap[snap["player_id"] == pid]
        sub_up = sub_p[sub_p["second"] <= checkpoint_sec]
        if len(sub_up) > 0 and "supply_used" in sub_up.columns and "supply_cap" in sub_up.columns:
            blocked = (sub_up["supply_used"] >= sub_up["supply_cap"] - 1).sum()
            row[f"{label}_supply_block_pct"] = float(blocked) / len(sub_up)
        else:
            row[f"{label}_supply_block_pct"] = 0.0

        # 9. Upgrade levels — highest weapons/armor upgrade completed so far
        upgrade_evts = event_feats.get(f"_upgrade_events_{pid}", [])
        row[f"{label}_weapons_level"] = max(
            (lvl for t, cat, lvl in upgrade_evts if cat == "weapons" and t <= checkpoint_sec),
            default=0,
        )
        row[f"{label}_armor_level"] = max(
            (lvl for t, cat, lvl in upgrade_evts if cat == "armor" and t <= checkpoint_sec),
            default=0,
        )

        # 10. Unit composition — army units produced per category up to checkpoint
        births = event_feats.get(f"_unit_births_{pid}", {})
        for cat in ALL_COMPOSITION_KEYS:
            row[f"{label}_{cat}_produced"] = sum(
                1 for t in births.get(cat, []) if t <= checkpoint_sec
            )

    # Delta features — p1 minus p2 for every numeric stat
    delta_cols = (
        STAT_COLS
        + ["minerals_total", "vespene_total"]
        + [f"{c}_roc" for c in ["workers_active", "army_supply", "minerals_collection_rate"]]
        + ["natural_age", "third_age", "has_natural", "has_third",
           "army_lost_count", "army_lost_per_min", "worker_lost_count"]
        + ["spending_quotient", "supply_block_pct", "weapons_level", "armor_level"]
        + [f"{cat}_produced" for cat in ALL_COMPOSITION_KEYS]
    )
    for col in delta_cols:
        p1_val = row.get(f"p1_{col}")
        p2_val = row.get(f"p2_{col}")
        if p1_val is not None and p2_val is not None:
            try:
                row[f"delta_{col}"] = float(p1_val) - float(p2_val)
            except (TypeError, ValueError):
                row[f"delta_{col}"] = np.nan
        else:
            row[f"delta_{col}"] = np.nan

    return row


def parse_replay_file(args: tuple) -> tuple[list[dict], dict | None]:
    """
    Worker function for parallel parsing — safe to use with ProcessPoolExecutor.
    Args: (path, checkpoints_or_None)
    Returns: (list_of_rows, error_dict_or_None)
    """
    path, checkpoints = args
    try:
        r = load_replay(path)
        if len(r.players) != 2:
            return [], None
        return build_all_checkpoints(r, checkpoints=checkpoints), None
    except Exception as exc:
        return [], {"file": Path(path).name, "error": str(exc)}


def build_all_checkpoints(
    replay,
    checkpoints: list[int] | None = None,
) -> list[dict]:
    """
    Build one feature row per checkpoint for a single replay.
    Parses the replay once and reuses the snapshot and event features.

    checkpoints : seconds. Defaults to every 60s from 3 min to 15 min.
    """
    if checkpoints is None:
        checkpoints = list(range(180, 901, 60))

    snap        = build_snapshot_df(replay, interval=30)
    event_feats = build_event_features(replay)

    rows = []
    for t in checkpoints:
        row = build_checkpoint_row(replay, t, snap=snap, event_feats=event_feats)
        if row:
            rows.append(row)
    return rows
