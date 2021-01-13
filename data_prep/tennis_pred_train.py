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
            Dense(64, input_dim=22, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(2, activation="softmax"),
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
    na_ok_cols = set(["winner_seed", "winner_entry", "loser_seed", "loser_entry"])
    for fn_csv in os.listdir("clean_data"):
        fp_csv = os.path.join("clean_data", fn_csv)  # File path
        # Preprocess data
        df = pd.read_csv(fp_csv, usecols=headers)
        df = df.dropna(subset=list(set(headers) - na_ok_cols))
        df = df.fillna(0)
        df = df.reset_index()
        len_data = len(df["winner_name"])
        # Collect and clean player data (switch from winner/loser to player1/2)
        data = []
        labels = []  # 0: Player 0, 1: Player 1
        for i in range(len_data):
            # Get data from dataframe
            c_data = (
                t_name,
                t_surface,
                t_size,
                t_level,
                w_seed,
                w_entry,
                w_name,
                w_hand,
                w_ht,
                w_ioc,
                w_age,
                w_rank,
                w_rank_pts,
                l_seed,
                l_entry,
                l_name,
                l_hand,
                l_ht,
                l_ioc,
                l_age,
                l_rank,
                l_rank_pts,
                t_best_of,
                t_round,
            ) = df.iloc[[i]].values[0][1:]
            # Convert string data to ID's
            t_surface_id = surface_id[t_surface]
            t_level_id = level_id[t_level]
            t_round_id = round_id[t_round]
            w_entry_id = entry_id[w_entry]
            l_entry_id = entry_id[l_entry]
            w_hand_id = hand_id[w_hand]
            l_hand_id = hand_id[l_hand]
            t_id = set_id(
                t_name, tourn_id, [t_surface_id, t_size, t_level_id, t_best_of]
            )
            w_ioc_id = set_id(w_ioc, ioc_id)
            l_ioc_id = set_id(l_ioc, ioc_id)
            w_id = set_id(w_name, player_id, [w_hand_id, w_ht, w_ioc_id])
            l_id = set_id(l_name, player_id, [l_hand_id, l_ht, l_ioc_id])
            data.append(
                [
                    t_id,
                    t_surface_id,
                    t_size,
                    t_level_id,
                    t_best_of,
                    t_round_id,
                    w_id,
                    w_seed,
                    w_entry_id,
                    w_hand_id,
                    w_ht,
                    w_ioc_id,
                    w_age,
                    w_rank,
                    # w_rank_pts,
                    l_id,
                    l_seed,
                    l_entry_id,
                    l_hand_id,
                    l_ht,
                    l_ioc_id,
                    l_age,
                    l_rank,
                    # l_rank_pts,
                ]
            )
            labels.append([1, 0])
            # Reverse winners and losers so model doesn't overfit
            data.append(
                [
                    t_id,
                    t_surface_id,
                    t_size,
                    t_level_id,
                    t_best_of,
                    t_round_id,
                    l_id,
                    l_seed,
                    l_entry_id,
                    l_hand_id,
                    l_ht,
                    l_ioc_id,
                    l_age,
                    l_rank,
                    # l_rank_pts,
                    w_id,
                    w_seed,
                    w_entry_id,
                    w_hand_id,
                    w_ht,
                    w_ioc_id,
                    w_age,
                    w_rank,
                    # w_rank_pts,
                ]
            )
            labels.append([0, 1])
    dump_json(
        [tourn_id, player_id],
        ["tourn_id.json", "player_id.json"],
    )
    return np.asarray(data), np.asarray(labels)


def set_id(str_key, id_dict, relevant_details=[]):
    if str_key in id_dict.keys():
        id_num = id_dict[str_key][0]
    else:
        id_num = len(id_dict)
        id_dict[str_key] = list(map(int, [id_num] + relevant_details))
    return id_num


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
