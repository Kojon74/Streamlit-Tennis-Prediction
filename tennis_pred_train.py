"""
- Model input: Tourney ID, Surface, Draw Size, Tourney Level, Best Of, Round Player 1 ID, Player 1 Seed, Player 1 Entry, 
    Player 1 Hand, Player 1 Height, Player 1 IOC, Player 1 Age, Player 1 Rank, Player 1 Rank Points, Player 2 ID, Player 2 Seed, 
    Player 2 Entry, Player 2 Hand, Player 2 Height, Player 2 IOC, Player 2 Age, Player 2 Rank, Player 2 Rank Points
"""

import os
import json
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

ID_PATH = "ids"


def get_model():
    model = Sequential(
        [
            Dense(64, input_dim=24, activation="relu"),
            Dropout(0),
            Dense(64, activation="relu"),
            Dropout(0),
            Dense(64, activation="relu"),
            Dropout(0),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def get_data():
    tourn_id = {}
    player_id = {}
    ioc_id = {}
    surface_id = {0: 0, "Clay": 1, "Grass": 2, "Hard": 3, "Carpet": 4}
    level_id = {0: 0, "C": 1, "D": 2, "A": 3, "M": 4, "F": 5, "G": 6}
    round_id = {
        0: 0,
        "RR": 1,
        "BR": 2,
        "R128": 3,
        "R64": 4,
        "R32": 5,
        "R16": 6,
        "QF": 7,
        "SF": 8,
        "F": 9,
    }
    entry_id = {0: 0, "PR": 1, "LL": 2, "WC": 3, "Q": 4, "S": 5, "SE": 6}
    hand_id = {0: 0, "U": 1, "L": 2, "R": 3}
    headers = [
        "tourney_name",
        "surface",
        "draw_size",
        "tourney_level",
        "winner_seed",
        "winner_entry",
        "winner_name",
        "winner_hand",
        "winner_ht",
        "winner_ioc",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "loser_seed",
        "loser_entry",
        "loser_name",
        "loser_hand",
        "loser_ht",
        "loser_ioc",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
        "best_of",
        "round",
    ]
    for fn_csv in os.listdir("clean_data"):
        df = pd.read_csv(
            os.path.join("clean_data", fn_csv),
            usecols=headers,
        )
        df = df.dropna(
            subset=list(
                set(headers)
                - set(["winner_seed", "winner_entry", "loser_seed", "loser_entry"])
            )
        )
        df = df.fillna(0)
        df = df.reset_index()
        len_data = len(df["winner_name"])
        # Collect and clean player data (switch from winner/loser to player1/2)
        data = []
        labels = []  # 0: Player 0, 1: Player 1
        for i in range(len_data):
            # Get data from dataframe
            t_name, t_surface, t_size, t_level, t_best_of, t_round = [
                df[x][i] for x in headers[:4] + headers[-2:]
            ]
            wl_start = ["winner_", "loser_"]
            w_name = df["winner_name"][i]
            l_name = df["loser_name"][i]
            # Get Player 1 data
            (
                p1_seed,
                p1_entry,
                p1_name,
                p1_hand,
                p1_ht,
                p1_ioc,
                p1_age,
                p1_rank,
                p1_rank_pts,
            ) = [
                df[x][i]
                for x in headers
                if x.startswith(wl_start[not (w_name < l_name)])
            ]
            # Get Player 2 data
            (
                p2_seed,
                p2_entry,
                p2_name,
                p2_hand,
                p2_ht,
                p2_ioc,
                p2_age,
                p2_rank,
                p2_rank_pts,
            ) = [df[x][i] for x in headers if x.startswith(wl_start[w_name < l_name])]
            labels.append(int(not (w_name < l_name)))
            # Convert string data to ID's
            t_surface_id = surface_id[t_surface]
            t_level_id = level_id[t_level]
            t_round_id = round_id[t_round]
            p1_entry_id = entry_id[p1_entry]
            p2_entry_id = entry_id[p2_entry]
            p1_hand_id = hand_id[p1_hand]
            p2_hand_id = hand_id[p2_hand]
            if t_name in tourn_id.keys():  # Set tourney ID
                t_id = tourn_id[t_name][0]
            else:
                t_id = len(tourn_id)
                tourn_id[t_name] = list(
                    map(
                        int,
                        [
                            t_id,
                            t_surface_id,
                            t_size,
                            t_level_id,
                            t_best_of,
                        ],
                    )
                )
            if p1_ioc in ioc_id.keys():  # Set P1 IOC ID
                p1_ioc_id = ioc_id[p1_ioc][0]
            else:
                p1_ioc_id = len(ioc_id)
                ioc_id[p1_ioc] = [p1_ioc_id]
            if p2_ioc in ioc_id.keys():  # Set P2 IOC ID
                p2_ioc_id = ioc_id[p2_ioc][0]
            else:
                p2_ioc_id = len(ioc_id)
                ioc_id[p2_ioc] = [p2_ioc_id]
            if p1_name in player_id.keys():  # Set P1 ID
                p1_id = player_id[p1_name][0]
            else:
                p1_id = len(player_id)
                player_id[p1_name] = list(
                    map(int, [p1_id, p1_hand_id, p1_ht, p1_ioc_id])
                )
            if p2_name in player_id.keys():  # Set P2 ID
                p2_id = player_id[p2_name][0]
            else:
                p2_id = len(player_id)
                player_id[p2_name] = list(
                    map(int, [p2_id, p2_hand_id, p2_ht, p2_ioc_id])
                )
            data.append(
                [
                    t_id,
                    t_surface_id,
                    t_size,
                    t_level_id,
                    t_best_of,
                    t_round_id,
                    p1_id,
                    p1_seed,
                    p1_entry_id,
                    p1_hand_id,
                    p1_ht,
                    p1_ioc_id,
                    p1_age,
                    p1_rank,
                    p1_rank_pts,
                    p2_id,
                    p2_seed,
                    p2_entry_id,
                    p2_hand_id,
                    p2_ht,
                    p2_ioc_id,
                    p2_age,
                    p2_rank,
                    p2_rank_pts,
                ]
            )
    dump_json(
        [tourn_id, player_id],
        ["tourn_id.json", "player_id.json"],
    )
    return np.asarray(data), np.asarray(labels)


def dump_json(id_dict, json_names):
    for i, json_name in enumerate(json_names):
        with open(os.path.join(ID_PATH, json_name), "w") as f_json:
            json.dump(id_dict[i], f_json)


def main():
    model = get_model()
    X, y = get_data()
    model.fit(X, y, epochs=500)
    model.save("models")


if __name__ == "__main__":
    main()
