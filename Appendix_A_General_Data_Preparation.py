import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

columns = [
    "Groupnr",
    "Round",
    "Country",
    "childmortality",
    "childmortalitytime",
    *[f"foodsecurity{i}" for i in range(1, 10)],
    *[f"foodsecurity{i}freq" for i in range(1, 10)],
    "fuelcooking",
    "sourcelighting",
    "watersource",
    "timewatersource_1",
    "timewatersourceunit",
    "Toiletfacility",
    "materialroof",
    "materialfloor",
    "materialwallsext",
    "assetsmatrix2_7",
    "assetsmatrix2_14",
    "assetsmatrix2_16",
    "assetsmatrix1_23",
    "assetsmatrix3_14",
    "assetsmatrix3_16",
    "assetsmatrix2_12",
    "assetsmatrix3_22",
    *[f"HHMschool_{n}" for n in range(1, 6)],
    *[f"HHMschoolnow_{n}" for n in range(1, 6)],
    *[f"HHMschoolcompl_{n}" for n in range(1, 6)],
    "school",
    "schoolcompleted",
    "savinghowmuch_1",
    "savinghowmuch_2",
    "savinghowmuch_3",
    "savingstotal_1",
    "debt",
    "debtamount_1",
    "debtnote",
    *[f"anxiety{i}" for i in range(1, 8)],
    "psychwellbeing_1",
    "psychwellbeing_3",
    "psychwellbeing_5",
    "psychwellbeing2_5",
    "jealousy",
    "jealousywhat",
    *[f"livestocknumbers_{i}" for i in [1,13,3,4,5,6,11,8,9,7,2,10]],
    "assetsmatrix1_4",
    "assetsmatrix1_5",
    "assetsmatrix1_22",
    "assetsmatrix2_7",
    "assetsmatrix2_14",
    "assetsmatrix2_15",
    "assetsmatrix2_16",
    "assetsmatrix2_8",
    "assetsmatrix3_17",
    "assetsmatrix2_17",
    "assetsmatrix2_18",
    "assetsmatrix2_19",
    "assetsmatrix2_11",
    "assetsmatrix2_12",
    "assetsmatrix3_14",
    "assetsmatrix1_23",
    "assetsmatrix3_15",
    "assetsmatrix3_16",
    "assetsmatrix3_22",
    "assetsmatrix3_23",
    "occupationmain",
    "ownsland_scto",
    "meetings1",
    "moneywithdraw",
    "moneyproblems"
]

columns_available_in_data = [col for col in columns if col in df.columns]

df = df[columns_available_in_data]

numerical = [
    "savinghowmuch_1", "savinghowmuch_2", "savinghowmuch_3",
    "savingstotal_1", "debtamount_1", "timewatersource_1"
]

ordered_categorical = [
    *[f"foodsecurity{i}freq" for i in range(1, 10)],
    *[f"anxiety{i}" for i in range(1, 8)],
    "psychwellbeing_1", "psychwellbeing_3", "psychwellbeing_5", "psychwellbeing2_5"
]
categorical = [
    "fuelcooking", "sourcelighting", "watersource", "Toiletfacility",
    "materialroof", "materialfloor", "materialwallsext",
    *[f"HHMschoolcompl_{n}" for n in range(1, 6)],
    "schoolcompleted", "livestocknumbers_1",
    *[f"livestocknumbers_{i}" for i in [1, 13, 3, 4, 5, 6, 11, 8, 9, 7, 2, 10]],					
    "occupationmain"
]

binary = [
    "childmortality",
    *[f"foodsecurity{i}" for i in range(1, 10)],
    *[f"HHMschool_{n}" for n in range(1, 6)],
    *[f"HHMschoolnow_{n}" for n in range(1, 6)],
    "school", "debt", "jealousy",
    "assetsmatrix1_4", "assetsmatrix1_5", "assetsmatrix1_22",
    "assetsmatrix2_15", "assetsmatrix2_8", "assetsmatrix3_17",
    "assetsmatrix2_17", "assetsmatrix2_18", "assetsmatrix2_19",
    "assetsmatrix2_11", "assetsmatrix3_15", "assetsmatrix3_23",
    "meetings1", "moneywithdraw", "moneyproblems"
]

multiple_choice = [
    "debtnote", "jealousywhat"
]

information = [
    "Country", "Groupnr", "Round"
]


binary_neg = [
    "debt", "foodsecurity1","foodsecurity2", "foodsecurity3", "foodsecurity4", "foodsecurity5",
    "foodsecurity6", "foodsecurity7", "foodsecurity8", "foodsecurity9", "childmortality",
    "jealousy", "assetsmatrix1_4", "assetsmatrix1_5", "assetsmatrix1_22", "assetsmatrix2_15",
    "assetsmatrix2_8", "assetsmatrix3_17", "assetsmatrix2_17",
    "assetsmatrix2_18", "assetsmatrix2_19", "assetsmatrix2_11",
    "assetsmatrix3_15", "assetsmatrix3_23"
]

binary_pos = [
    "HHMschoolnow_1", "HHMschoolnow_2", "HHMschoolnow_3",
    "HHMschoolnow_4", "HHMschoolnow_5",
    "school", "meetings1", "moneywithdraw", "moneyproblems"
]

valid_rounds = ['0', '1', '2', '3', '100', '102']
df = df[df['Round'].notna() & ~df['Round'].isin(['Onboarding', '6', '6.0'])]
df['Round'] = pd.to_numeric(df['Round'], errors='coerce').dropna().astype(int).astype(str)
df = df[df['Round'].isin(valid_rounds)]
df = df.sort_values(by='Round', ascending=True)


countries = ['GHA', 'RWA', 'UGA', 'CIV', 'KEN']
df = df[df['Country'].isin(countries)]

df_gha = df[df['Country'] == 'GHA']
df_rwa = df[df['Country'] == 'RWA']
df_uga = df[df['Country'] == 'UGA']
df_civ = df[df['Country'] == 'CIV']
df_ken = df[df['Country'] == 'KEN']
