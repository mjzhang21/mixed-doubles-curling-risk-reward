from __future__ import annotations


# typing helpers for annotations
from typing import Dict, List, Optional, Tuple

# this is easier i recommend if you end up changing stuff in your branch it will still find stuff.
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "end_level.csv"


#basically i do not ordinarily preprocess the data with inplace modification or making
# progressive edits to one dataframe, etc. for context, i made progressive edits to a single
# data frame for the in-class presentation i did with this. the code took longer to run,
# used significantly more of my computer's memory, and it was much more difficult to track
# down errors.

# i build a few  functions first which each preprocess their respective dataframe.
# the last function uses the other functions.

# imo if we decide to make this into a quarto file, it will be much easier to follow this way.

# dictionary from csas
TASK_LABELS = {
    -1: "Unknown / No statistics",
    0: "Draw",
    1: "Front",
    2: "Guard",
    3: "Raise / Tap-back",
    4: "Wick / Soft Peeling",
    5: "Freeze",
    6: "Take-out",
    7: "Hit and Roll",
    8: "Clearing",
    9: "Double Take-out",
    10: "Promotion Take-out",
    11: "Through",
    13: "No Statistics",
}


def _task_col_name(task_value: int) -> str:
    """Create a safe column name for a task code (handles negatives)."""
    return f"task_{task_value}" if task_value >= 0 else f"task_neg{abs(task_value)}"


# because this isnt a quarto and a .py, i use functions to load data because this 
# is just one scrit. i think its technically more optimal.
def _load_games() -> pd.DataFrame:
    games = pd.read_csv(DATA_DIR / "Games.csv").rename(
        columns={
            "CompetitionID": "competition_id",
            "SessionID": "session_id",
            "GameID": "game_id",
            "GroupID": "group_id",
            "Sheet": "sheet",
            "NOC1": "noc1",
            "NOC2": "noc2",
            "ResultStr1": "resultstr1",
            "ResultStr2": "resultstr2",
            "LSFE": "lsfe",
            "Winner": "winner",
            "TeamID1": "team_id1",
            "TeamID2": "team_id2",
        }
    )
    int_cols = [
        "competition_id",
        "session_id",
        "game_id",
        "group_id",
        "lsfe",
        "winner",
        "team_id1",
        "team_id2",
    ]
    games[int_cols] = games[int_cols].astype(int)
    games["match_id"] = (
        games["competition_id"].astype(str)
        + "_"
        + games["session_id"].astype(str)
        + "_"
        + games["game_id"].astype(str)
    )

    # LSFE
    # 1 means TeamID1 has hammer.  0 means TeamID2 has hammer.
    games["initial_hammer_team_id"] = np.where(
        games["lsfe"] == 1, games["team_id1"], games["team_id2"]
    )
    return games


def _load_ends() -> pd.DataFrame:
    # use preferred naming strecture
    ends = pd.read_csv(DATA_DIR / "Ends.csv").rename(
        columns={
            "CompetitionID": "competition_id",
            "SessionID": "session_id",
            "GameID": "game_id",
            "TeamID": "team_id",
            "EndID": "end_id",
            "Result": "result",
            "PowerPlay": "powerplay",
        }
    )
    int_cols = ["competition_id", "session_id", "game_id", "team_id", "end_id"]
    ends[int_cols] = ends[int_cols].astype(int)
    ends["match_id"] = (
        ends["competition_id"].astype(str)
        + "_"
        + ends["session_id"].astype(str)
        + "_"
        + ends["game_id"].astype(str)
    )
    # result 9 marks a conceded end; keep a flag but map to NA for score math
    ends["conceded_end"] = ends["result"] == 9
    ends["result"] = ends["result"].replace(9, np.nan)
    return ends


def _load_stones() -> pd.DataFrame:
    stones = pd.read_csv(DATA_DIR / "Stones.csv").rename(
        columns={
            "CompetitionID": "competition_id",
            "SessionID": "session_id",
            "GameID": "game_id",
            "EndID": "end_id",
            "ShotID": "shot_id",
            "TeamID": "team_id",
            "PlayerID": "player_id",
            "Task": "task",
            "Handle": "handle",
            "Points": "points",
            "TimeOut": "timeout",
        }
    )
    int_cols = [
        "competition_id",
        "session_id",
        "game_id",
        "end_id",
        "shot_id",
        "team_id",
        "player_id",
        "task",
    ]
    stones[int_cols] = stones[int_cols].astype(int)
    stones["match_id"] = (
        stones["competition_id"].astype(str)
        + "_"
        + stones["session_id"].astype(str)
        + "_"
        + stones["game_id"].astype(str)
    )
    return stones


def _load_teams() -> pd.DataFrame:
    teams = pd.read_csv(DATA_DIR / "Teams.csv").rename(
        columns={
            "CompetitionID": "competition_id",
            "TeamID": "team_id",
            "NOC": "noc",
            "Name": "name",
        }
    )
    return teams


# this runs thru the columns to count the number of shots per team.
def _aggregate_shot_counts(stones: pd.DataFrame) -> pd.DataFrame:
    counts = (
        stones.groupby(["match_id", "end_id", "team_id"])["task"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename_map = {
        col: _task_col_name(int(col))
        for col in counts.columns
        if isinstance(col, (int, float, np.integer, np.floating))
    }
    counts = counts.rename(columns=rename_map)
    task_cols = [c for c in counts.columns if c not in {"match_id", "end_id", "team_id"}]
    counts["total_shots"] = counts[task_cols].sum(axis=1)
    return counts


# this adds the other csv such that we can easily access NOC for both teams. you can
# safely remove this, because we still have teamID
def _merge_team_meta(
    ends_df: pd.DataFrame, teams: pd.DataFrame, side: str, team_col: str
) -> pd.DataFrame:
    """Attach team name/NOC for hammer or non-hammer columns."""
    suffix = f"{side}_"
    meta = teams.rename(
        columns={
            "team_id": team_col,
            "noc": f"{suffix}noc",
            "name": f"{suffix}name",
        }
    )[["competition_id", team_col, f"{suffix}noc", f"{suffix}name"]]
    return ends_df.merge(meta, on=["competition_id", team_col], how="left")


#just identifies the powerplay ends.
def _end_powerplay(end_df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    pp = end_df.loc[end_df["powerplay"].notna(), ["team_id", "powerplay"]]
    if pp.empty:
        return None, None
    rec = pp.iloc[0]
    return int(rec.team_id), float(rec.powerplay)


#this combines per team end rows into just one end row. remove this if you want both
def _summarize_match_ends(
    game_row: pd.Series, match_ends: pd.DataFrame
) -> List[Dict[str, object]]:

    rows: List[Dict[str, object]] = []
    teams: Tuple[int, int] = (int(game_row.team_id1), int(game_row.team_id2))
    hammer_team = int(game_row.initial_hammer_team_id)
    # Track running totals entering each end (i.e., before the end is played).
    cumulative_hammer = 0
    cumulative_nonhammer = 0

    grouped = match_ends.groupby("end_id")
    for end_id in sorted(grouped.groups):
        end_df = grouped.get_group(end_id)
        results = end_df.set_index("team_id")["result"].to_dict()
        score_hammer = results.get(hammer_team)
        non_hammer_team = teams[0] if hammer_team == teams[1] else teams[1]
        score_nonhammer = results.get(non_hammer_team)

        clean_scores = {k: v for k, v in results.items() if pd.notna(v)}
        total_points = sum(clean_scores.values()) if clean_scores else 0
        scoring_team = None
        if clean_scores:
            positives = {k: v for k, v in clean_scores.items() if v > 0}
            if positives:
                scoring_team = max(positives, key=positives.get)

        powerplay_team, powerplay_value = _end_powerplay(end_df)
        # score diff from hammer perpesctive 
        if pd.isna(score_hammer) and pd.isna(score_nonhammer):
            score_diff = np.nan
        else:
            score_diff = (score_hammer or 0) - (score_nonhammer or 0)

        conceded_end = bool(end_df["conceded_end"].any())

        rows.append(
            {
                "match_id": game_row.match_id,
                "competition_id": int(game_row.competition_id),
                "session_id": int(game_row.session_id),
                "game_id": int(game_row.game_id),
                "group_id": int(game_row.group_id),
                "sheet": game_row.sheet,
                "end_id": int(end_id),
                "hammer_team_id": hammer_team,
                "nonhammer_team_id": non_hammer_team,
                "score_hammer": score_hammer,
                "score_nonhammer": score_nonhammer,
                "score_diff": score_diff,
                "conceded_end": conceded_end,
                # cumulative scores entering this end (not including current end results)
                "cumulative_score_hammer": cumulative_hammer,
                "cumulative_score_nonhammer": cumulative_nonhammer,
                "cumulative_score_diff": cumulative_hammer - cumulative_nonhammer,
                "scoring_team_id": scoring_team,
                "scoring_team_role": (
                    "hammer"
                    if scoring_team is not None and scoring_team == hammer_team
                    else ("nonhammer" if scoring_team is not None else None)
                ),
                "powerplay_team_id": powerplay_team,
                "powerplay_value": powerplay_value,
                "powerplay_by_hammer": powerplay_team == hammer_team
                if powerplay_team is not None
                else False,
                # blank only when scores are recorded and sum to zero
                "blank_end": bool(clean_scores) and total_points == 0,
            }
        )

        ##hammer tracker. i couldnt find out a cleaner method.
        if scoring_team is None or total_points == 0:
            next_hammer = hammer_team
        elif scoring_team == hammer_team:
            next_hammer = non_hammer_team
        else:
            next_hammer = hammer_team
        hammer_team = next_hammer

        # running score after this end (used as entering score for the next end)
        cumulative_hammer += 0 if pd.isna(score_hammer) else score_hammer
        cumulative_nonhammer += 0 if pd.isna(score_nonhammer) else score_nonhammer

    return rows

# now data is pre processed. the next function cleanly mergers it all.

def build_end_level_dataframe() -> pd.DataFrame:
    games = _load_games()
    ends = _load_ends()
    stones = _load_stones()
    teams = _load_teams()

    shot_counts = _aggregate_shot_counts(stones)

    rows: List[Dict[str, object]] = []
    ends_by_match = {mid: df for mid, df in ends.groupby("match_id")}

    for _, game_row in games.iterrows():
        match_id = game_row.match_id
        if match_id not in ends_by_match:
            continue  # No end data found for this match.
        rows.extend(_summarize_match_ends(game_row, ends_by_match[match_id]))

    end_level = pd.DataFrame(rows)

    #attach team level data
    end_level = _merge_team_meta(end_level, teams, "hammer", "hammer_team_id")
    end_level = _merge_team_meta(end_level, teams, "nonhammer", "nonhammer_team_id")

    #merge shot counts
    hammer_counts = shot_counts.rename(columns={"team_id": "hammer_team_id"})
    hammer_task_cols = [
        c for c in hammer_counts.columns if c not in {"match_id", "end_id", "hammer_team_id"}
    ]
    hammer_counts = hammer_counts.rename(
        columns={c: f"hammer_{c}" for c in hammer_task_cols}
    )

    nonhammer_counts = shot_counts.rename(columns={"team_id": "nonhammer_team_id"})
    nonhammer_task_cols = [
        c for c in nonhammer_counts.columns if c not in {"match_id", "end_id", "nonhammer_team_id"}
    ]
    nonhammer_counts = nonhammer_counts.rename(
        columns={c: f"nonhammer_{c}" for c in nonhammer_task_cols}
    )

    end_level = end_level.merge(
        hammer_counts, on=["match_id", "end_id", "hammer_team_id"], how="left"
    ).merge(
        nonhammer_counts,
        on=["match_id", "end_id", "nonhammer_team_id"],
        how="left",
    )

    # missing shot counts filled with 0 because not technically missing
    count_cols = [
        c
        for c in end_level.columns
        if c.startswith(("hammer_task", "nonhammer_task", "hammer_total_shots", "nonhammer_total_shots"))
    ]
    end_level[count_cols] = end_level[count_cols].fillna(0).astype(int)

    end_level = end_level.sort_values(["match_id", "end_id"]).reset_index(drop=True)
    return end_level


def main(output_path: Path = OUTPUT_PATH) -> None:
    df = build_end_level_dataframe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} end-level rows to {output_path}")


if __name__ == "__main__":
    main()
