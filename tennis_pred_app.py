import json
import streamlit as st
import os

from tensorflow import keras


def main():
    player_id, tourn_id = load_ids()
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
    p1_seed = int(col1.text_input("Seed", value=0, key=1))
    p1_entry = col1.multiselect(
        "Entry", options=["R", "PR", "LL", "WC", "Q", "S", "SE"], key=2
    )
    p1_rank = int(col1.text_input("Rank", value=0, key=3))
    p1_rank_pts = int(col1.text_input("Rank Points", value=0, key=4))
    p1_age = int(col1.text_input("Age", value=0, key=5))
    p2_name = col2.multiselect("Name", options=list(player_id.keys()), key=6)
    p2_seed = int(col2.text_input("Seed", value=0, key=7))
    p2_entry = col2.multiselect(
        "Entry", options=["R", "PR", "LL", "WC", "Q", "S", "SE"], key=8
    )
    p2_rank = int(col2.text_input("Rank", value=0, key=9))
    p2_rank_pts = int(col2.text_input("Rank Points", value=0, key=10))
    p2_age = int(col2.text_input("Age", value=0, key=11))

    if st.button("Predict"):
        t_id, t_surface_id, t_size, t_level_id, t_best_of = tourn_id[t_name[0]]
        p1_id, p1_hand_id, p1_ht, p1_ioc_id = player_id[p1_name[0]]
        p2_id, p2_hand_id, p2_ht, p2_ioc_id = player_id[p2_name[0]]
        t_round_id = round_id[t_round[0]]
        p1_entry_id = entry_id[p1_entry[0]]
        p2_entry_id = entry_id[p2_entry[0]]
        p1_seed = 0 if p1_seed == "N/A" else p1_seed
        p2_seed = 0 if p2_seed == "N/A" else p2_seed
        data = [
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
        ]
        print(data)
        model = keras.models.load_model("models")
        prediction = model.predict(data)
        print(prediction)
        st.write(
            "Predicted winner is: {}".format(
                [p1_name[0], p2_name[0]][int(prediction[0][0])]
            )
        )
        print(prediction)


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