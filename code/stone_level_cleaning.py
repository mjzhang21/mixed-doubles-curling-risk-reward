# Import modules and data.

from pathlib import Path
import pandas as pd
import numpy as np

# change the data directory if the folder is called something other than "data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "stones_master_1.csv"


def load_cleaned_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)

    # games = pd.read_csv(data_dir / "Games.csv")

    ends_df = pd.read_csv(data_dir / "Ends.csv")

    ends_df["PowerPlay"] = ends_df["PowerPlay"].fillna(0)
    ends_df["PowerPlay"] = ends_df["PowerPlay"].astype("Int64")

    games_df = pd.read_csv(data_dir / "Games.csv")

    stones_df = pd.read_csv(data_dir / "Stones.csv")

    # cast stone coordinates to integer (they are float for some reason.)
    stone_cols = [c for c in stones_df.columns if c.startswith("stone_")]
    stones_df[stone_cols] = stones_df[stone_cols].astype("Int64")

    stones_df["TimeOut"] = stones_df["TimeOut"].fillna(0)
    stones_df["TimeOut"] = stones_df["TimeOut"].astype("Int64")

    # Make `MatchID` out of a combination of `CompetitionID`, `SessionID`,
    # and `GameID`. This is the unique key going forward.

    games_df["MatchID"] = (
        games_df["CompetitionID"].astype(str) + "_"
        + games_df["SessionID"].astype(str) + "_"
        + games_df["GameID"].astype(str)
    )

    ends_df["MatchID"] = (
        ends_df["CompetitionID"].astype(str) + "_"
        + ends_df["SessionID"].astype(str) + "_"
        + ends_df["GameID"].astype(str)
    )

    # Create `game_ends` via merger of `ends_df` and `games_df`.
    # This will attach game level information to each end.

    game_ends = (
        ends_df.merge(
            games_df,
            on="MatchID",
            how="left",
            validate="many_to_one",
        )
    )

    # That merger duplicates the redundant ID columns,
    # affixing them with `_y`, and affixes the existing ones with `_x`.

    # remove duplicate key columns from the right side
    game_ends = game_ends.drop(
        columns=["CompetitionID_y", "SessionID_y", "GameID_y"]
    )
    game_ends.rename(
        columns={
            "CompetitionID_x": "CompetitionID",
            "SessionID_x": "SessionID",
            "GameID_x": "GameID",
        },
        inplace=True,
    )

    # While a `PowerPlay` column exists, for any end which Power Play was not
    # used, there is no information on which team has the hammer. We need to
    # create a column `HasHammer` which gives us this useful information.

    # There are two ways to identify whether a team has the hammer in a given
    # end.
    # - **Iteratively:** Start with the first End in a game. The team with the
    # last stone first end (LSFE) has the hammer the first end. Depending on
    # which team scores, the hammer will be kept or transferred. This way, the
    # hammer can be followed through the length of the game.
    # - **Contextually:** The team which throws the first stone, i.e., the
    # `TeamID` for which `ShotID == 7` must *not* have the hammer.

    # Both of these approaches should correctly track the Hammer in theory. I
    # opt to implement both as a data validation measure. If there are
    # discrepancies, the particular End has bad data.

    # Iterative approach: HasHammer_I

    # Per the CSAS2026 data dictionary:
    # "`LSFE` – Last Stone First End. This column indicates which team threw
    # the last stone in the first end of the match. In curling parlance, this
    # is called starting with “the hammer”. A 0 value means that NOC2 threw the
    # last stone in the first end, a 1 means that NOC1 threw last."

    # helper: map LSFE (0 means NOC2, 1 means NOC1) to TeamID that starts with
    # hammer in End 1
    def initial_hammer_teamid(row):
        return row["TeamID1"] if row["LSFE"] == 1 else row["TeamID2"]

    cols = [
        "MatchID",
        "EndID",
        "TeamID",
        "Result",
        "TeamID1",
        "TeamID2",
        "LSFE",
    ]
    ge = game_ends[cols].copy()
    ge.sort_values(["MatchID", "EndID"], inplace=True)

    def assign_hammer_i(df):
        match_id = df["MatchID"].iloc[0] if "MatchID" in df.columns else df.name
        team1 = df["TeamID1"].iloc[0]
        team2 = df["TeamID2"].iloc[0]
        hammer = initial_hammer_teamid(df.iloc[0])
        out = []
        for _, end_rows in df.groupby("EndID", sort=True):
            end_rows = end_rows.copy()
            end_rows["MatchID"] = match_id
            end_rows["HammerTeamID_I"] = hammer
            out.append(end_rows)
            totals = end_rows.groupby("TeamID")["Result"].sum().to_dict()
            scored = (totals.get(team1, 0) or 0) + (totals.get(team2, 0) or 0)
            if scored > 0:
                scoring_team = (
                    team1 if totals.get(team1, 0) > totals.get(team2, 0)
                    else team2
                )
                hammer = team2 if scoring_team == team1 else team1
        return pd.concat(out, ignore_index=True)

    ge_hammer_i = (
        ge.groupby("MatchID", group_keys=False)
        .apply(assign_hammer_i, include_groups=False)
    )
    ge_hammer_i["HasHammer_I"] = (
        ge_hammer_i["TeamID"] == ge_hammer_i["HammerTeamID_I"]
    )
    game_ends = (
        game_ends.merge(
            ge_hammer_i[
                ["MatchID", "EndID", "TeamID", "HammerTeamID_I", "HasHammer_I"]
            ],
            on=["MatchID", "EndID", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    )

    # Contextual approach: HasHammer_C
    # The team that throws the first stone (ShotID == 7) does *not* have the
    # hammer.

    if "MatchID" not in stones_df.columns:
        stones_df["MatchID"] = (
            stones_df["CompetitionID"].astype(str) + "_"
            + stones_df["SessionID"].astype(str) + "_"
            + stones_df["GameID"].astype(str)
        )
    first_throw = (
        stones_df.loc[stones_df["ShotID"] == 7, ["MatchID", "EndID", "TeamID"]]
        .drop_duplicates()
        .rename(columns={"TeamID": "FirstStoneTeamID"})
    )
    hammer_c = (
        game_ends[["MatchID", "EndID", "TeamID"]]
        .merge(first_throw, on=["MatchID", "EndID"], how="left",
               validate="many_to_one")
    )
    hammer_c["HammerTeamID_C"] = np.where(
        hammer_c["TeamID"] != hammer_c["FirstStoneTeamID"],
        hammer_c["TeamID"],
        pd.NA,
    )
    hammer_c["HammerTeamID_C"] = (
        hammer_c.groupby(["MatchID", "EndID"])["HammerTeamID_C"]
        .transform("max")
    )
    hammer_c["HasHammer_C"] = (
        hammer_c["TeamID"] == hammer_c["HammerTeamID_C"]
    )
    game_ends = (
        game_ends.merge(
            hammer_c[["MatchID", "EndID", "TeamID", "HammerTeamID_C",
                      "HasHammer_C"]],
            on=["MatchID", "EndID", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    )

    # Quick checks for missing hammer info and disagreements between methods.

    # There are five cases where they do not match. Once stone-level information
    # is fully merged, I will defer to each case individually.

    # Stone-Level Data

    # Build a master stone-level table (one row per stone toss) by merging
    # stone, end, and game context.

    stones_master = (
        stones_df.merge(
            game_ends,
            on=["MatchID", "EndID", "TeamID"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_end"),
        )
    )

    stones_master = stones_master.drop(
        ["CompetitionID", "GameID", "SessionID", "CompetitionID_end",
         "GameID_end", "SessionID_end"],
        axis=1,
    )

    # Add pre-shot (pre-end) team score and score differential.

    score_progress = game_ends[["MatchID", "EndID", "TeamID", "Result"]].copy()
    score_progress = score_progress.sort_values(["MatchID", "TeamID", "EndID"])

    score_progress["PreEndScore"] = (
        score_progress.groupby(["MatchID", "TeamID"])["Result"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
        .astype(int)
    )

    score_progress = score_progress.merge(
        score_progress[["MatchID", "EndID", "TeamID", "PreEndScore"]]
        .rename(
            columns={"TeamID": "OppTeamID", "PreEndScore": "OppPreEndScore"}
        ),
        on=["MatchID", "EndID"],
        how="left",
        validate="many_to_many",
    )
    score_progress = (
        score_progress[score_progress["TeamID"] != score_progress["OppTeamID"]]
    )
    score_progress = score_progress.drop(columns=["OppTeamID"])

    score_progress["PreEndScoreDiff"] = (
        score_progress["PreEndScore"] - score_progress["OppPreEndScore"]
    )

    stones_master = stones_master.merge(
        score_progress[
            [
                "MatchID",
                "EndID",
                "TeamID",
                "PreEndScore",
                "OppPreEndScore",
                "PreEndScoreDiff",
            ]
        ],
        on=["MatchID", "EndID", "TeamID"],
        how="left",
        validate="many_to_one",
    )

    # Check that the team throwing first (ShotID == 7) does not have the hammer.

    # When `ShotID == 7`, the team which threw the stone cannot have the hammer.
    # `HasHammer_C` has no such instances but `HasHammer_I` has five, which are
    # the same from the earlier mismatch. I defer to `HasHammer_C` for a more
    # reliable tracker, then.

    # These exceptions are unlikely a data quality issue and probably stem from
    # an issue with the iterative method code, but this does not warrant a fix
    # as the contextual method seems completely fine.

    # The `stones_master` dataframe is finalized below.

    stones_master = stones_master.drop(
        ["HasHammer_I", "HammerTeamID_I"],
        axis=1,
    )
    stones_master["HammerTeamID_C"] = (
        stones_master["HammerTeamID_C"].astype("Int64")
    )

    new_order = [
        "MatchID",
        "GroupID",
        "Sheet",
        "NOC1",
        "NOC2",
        "TeamID1",
        "TeamID2",
        "ResultStr1",
        "ResultStr2",
        "LSFE",
        "Winner",
        "EndID",
        "Result",
        "PowerPlay",
        "PreEndScore",
        "OppPreEndScore",
        "PreEndScoreDiff",
        "HammerTeamID_C",
        "HasHammer_C",
        "ShotID",
        "TeamID",
        "PlayerID",
        "Task",
        "Handle",
        "TimeOut",
        "Points",
        "stone_1_x",
        "stone_1_y",
        "stone_2_x",
        "stone_2_y",
        "stone_3_x",
        "stone_3_y",
        "stone_4_x",
        "stone_4_y",
        "stone_5_x",
        "stone_5_y",
        "stone_6_x",
        "stone_6_y",
        "stone_7_x",
        "stone_7_y",
        "stone_8_x",
        "stone_8_y",
        "stone_9_x",
        "stone_9_y",
        "stone_10_x",
        "stone_10_y",
        "stone_11_x",
        "stone_11_y",
        "stone_12_x",
        "stone_12_y",
    ]
    stones_master = stones_master[new_order]

    stones_master = stones_master.rename(
        columns={
            "MatchID": "match_id",
            "GroupID": "group_id",
            "Sheet": "sheet",
            "NOC1": "noc1",
            "NOC2": "noc2",
            "TeamID1": "team_id1",
            "TeamID2": "team_id2",
            "ResultStr1": "resultstr1",
            "ResultStr2": "resultstr2",
            "LSFE": "lsfe",
            "Winner": "winner",
            "EndID": "end_id",
            "Result": "result",
            "PowerPlay": "powerplay",
            "PreEndScore": "pre_end_score",
            "OppPreEndScore": "opp_pre_end_score",
            "PreEndScoreDiff": "pre_end_score_diff",
            "HammerTeamID_C": "hammer_team_id",
            "HasHammer_C": "has_hammer",
            "ShotID": "shot_id",
            "TeamID": "team_id",
            "PlayerID": "player_id",
            "Task": "task",
            "Handle": "handle",
            "TimeOut": "timeout",
            "Points": "points",
            # stone_* already good!!!!
        }
    )

    stone_cols = [c for c in stones_master.columns if c.startswith("stone_")]
    stones_master[stone_cols] = stones_master[stone_cols].astype("Int64")

    return stones_master


def main(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stones_master = load_cleaned_data()
    stones_master.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
