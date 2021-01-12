import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from tensorflow import keras


def main():
    player_id, tourn_id = load_ids()
    avg_ages, years = winner_age()
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
    entry_id = {"R": 0, "PR": 1, "LL": 2, "WC": 3, "Q": 4, "S": 5, "SE": 6}
    st.title("Tennis Match Prediction")

    t_name = st.multiselect("Tournament Name", options=list(tourn_id.keys()))
    t_round = st.multiselect(
        "Round", options=["RR", "BR", "R128", "R64", "R32", "R16", "QF", "SF", "F"]
    )

    # Player info
    col1, col2 = st.beta_columns(2)
    col1.header("Player 1")
    col2.header("Player 2")
    p1_name = col1.multiselect("Name", options=list(player_id.keys()), key=0)
    p1_seed_str = col1.text_input("Seed", value=0, key=1)
    p1_entry = col1.multiselect(
        "Entry", options=["R", "PR", "LL", "WC", "Q", "S", "SE"], key=2
    )
    p1_rank_str = col1.text_input("Rank", value=0, key=3)
    p1_rank_pts_str = col1.text_input("Rank Points", value=0, key=4)
    p1_age_str = col1.text_input("Age", value=0, key=5)
    p2_name = col2.multiselect("Name", options=list(player_id.keys()), key=6)
    p2_seed_str = col2.text_input("Seed", value=0, key=7)
    p2_entry = col2.multiselect(
        "Entry", options=["R", "PR", "LL", "WC", "Q", "S", "SE"], key=8
    )
    p2_rank_str = col2.text_input("Rank", value=0, key=9)
    p2_rank_pts_str = col2.text_input("Rank Points", value=0, key=10)
    p2_age_str = col2.text_input("Age", value=0, key=11)

    if st.button("Predict"):
        t_id, t_surface_id, t_size, t_level_id, t_best_of = tourn_id[t_name[0]]
        p1_id, p1_hand_id, p1_ht, p1_ioc_id = player_id[p1_name[0]]
        p2_id, p2_hand_id, p2_ht, p2_ioc_id = player_id[p2_name[0]]
        t_round_id = round_id[t_round[0]]
        p1_entry_id = entry_id[p1_entry[0]]
        p2_entry_id = entry_id[p2_entry[0]]
        p1_seed = 0 if p1_seed_str == "N/A" else int(p1_seed_str)
        p2_seed = 0 if p2_seed_str == "N/A" else int(p2_seed_str)
        p1_rank = int(p1_rank_str)
        p2_rank = int(p2_rank_str)
        p1_rank_pts = int(p1_rank_pts_str)
        p2_rank_pts = int(p2_rank_pts_str)
        p1_age = int(p1_age_str)
        p2_age = int(p2_age_str)
        data1 = [
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
                # p1_rank_pts,
                p2_id,
                p2_seed,
                p2_entry_id,
                p2_hand_id,
                p2_ht,
                p2_ioc_id,
                p2_age,
                p2_rank,
                # p2_rank_pts,
            ]
        ]
        data2 = [
            [
                t_id,
                t_surface_id,
                t_size,
                t_level_id,
                t_best_of,
                t_round_id,
                p2_id,
                p2_seed,
                p2_entry_id,
                p2_hand_id,
                p2_ht,
                p2_ioc_id,
                p2_age,
                p2_rank,
                # p2_rank_pts,
                p1_id,
                p1_seed,
                p1_entry_id,
                p1_hand_id,
                p1_ht,
                p1_ioc_id,
                p1_age,
                p1_rank,
                # p1_rank_pts,
            ]
        ]
        model = keras.models.load_model("models")
        pred1 = model.predict(data1)
        pred2 = model.predict(data2)
        prediction = np.asarray([(pred1[0][i] + pred2[0][1 - i]) / 2 for i in range(2)])
        players = [p1_name[0], p2_name[0]]
        winner = players[prediction.argmax()]
        probability = np.amax(prediction)
        st.write("Predicted winner is: {} ({}%)".format(winner, probability * 100))

    st.line_chart({"Average Age": avg_ages})


def winner_age():
    avg_ages = []
    headers = list(pd.read_csv("clean_data/atp_matches_2000.csv"))
    for fn_csv in sorted(os.listdir("clean_data")):
        fp_csv = os.path.join("clean_data", fn_csv)
        df = pd.read_csv(fp_csv, usecols=["winner_age", "round"])
        if fn_csv == "atp_matches_2016.csv" or fn_csv == "atp_matches_2017.csv":
            df = pd.read_csv(fp_csv, names=headers)
        df = df.dropna(subset=["winner_age", "round"])
        df = df[df["round"].map(lambda x: x == "F")]
        ages = df["winner_age"].values
        if fn_csv == "atp_matches_2016.csv" or fn_csv == "atp_matches_2017.csv":
            ages = [(float(str(age))) for age in ages]
        avg_age = sum(ages) / len(ages)
        avg_ages.append(avg_age)
    return avg_ages, range(2000, 2018)


def load_ids():
    id_jsons = sorted(os.listdir("ids"))
    id_dicts = []
    for id_json in id_jsons:
        with open(os.path.join("ids", id_json), "r") as f_json:
            id_dict = json.load(f_json)
            id_dicts.append(id_dict)
    return id_dicts


if __name__ == "__main__":
    main()