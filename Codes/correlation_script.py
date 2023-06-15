from joblib import Parallel, delayed
from itertools import combinations
from scipy.stats import spearmanr
from haversine import haversine
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import warnings

# Disable FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

years = ["2023"]
times = ["01","02","03","04"]
ay = ["January" , "February" , "March", "April"]
dates = []
realdates = []
for i in range(len(years)):
    for k in range(len(times)):
        dates.append("traffic_density_" + years[i] + times[k])
        realdates.append(ay[k] + " " + years[i])
        
def get_dist(row, distances):
    return distances[(row["GEOHASH_1"], row["GEOHASH_2"])]

def get_distance(row, distances):
    return distances[row["District Pair"]]

def process_date(date):
    print(date, "is processing.")

    df = pd.read_csv(date + ".csv")
    
    df.columns = ["DATE_TIME", "LONGITUDE", "LATITUDE", "GEOHASH", "MAXIMUM_SPEED", "MINIMUM_SPEED", "AVERAGE_SPEED", "NUMBER_OF_VEHICLES"]
    
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    df = df.set_index("DATE_TIME").groupby("GEOHASH").resample('1D').mean().reset_index()

    df.dropna(subset=['AVERAGE_SPEED'], inplace=True)
    
    geohash_counts = df.groupby("GEOHASH")["DATE_TIME"].nunique()

    max_count = geohash_counts.max()

    geohashes_to_drop = geohash_counts[geohash_counts < max_count].index

    df = df[~df["GEOHASH"].isin(geohashes_to_drop)]

    all_combs = np.array(list(combinations(list(np.unique(list(df["GEOHASH"]))), 2)))

    coordinates_df = df.groupby("GEOHASH")[["LONGITUDE", "LATITUDE"]].agg(lambda x: np.unique(x).tolist())
    coordinates_dict = coordinates_df.to_dict('index')
    
    distances = {}

    for comb in all_combs:
        c1 = (coordinates_dict[comb[0]]["LATITUDE"][0], coordinates_dict[comb[0]]["LONGITUDE"][0])
        c2 = (coordinates_dict[comb[1]]["LATITUDE"][0], coordinates_dict[comb[1]]["LONGITUDE"][0])
        d = haversine(c1, c2)
        distances[(comb[0], comb[1])] = d
    
    s_corr_dict = {}

    for comb in tqdm(all_combs):
        s_corr_dict[(comb[0], comb[1])] = tuple(spearmanr(df[df["GEOHASH"] == comb[0]]["AVERAGE_SPEED"], df[df["GEOHASH"] == comb[1]]["AVERAGE_SPEED"]))
        
    s_corr_dict_avg = pd.DataFrame.from_dict(s_corr_dict, orient="index").rename(columns={0: "Spearman Correlation", 1: "P-value"})
    s_corr_dict_avg = s_corr_dict_avg.sort_values("Spearman Correlation", ascending=False).reset_index().rename(columns={"index": "District Pair"})

    s_corr_dict_avg = s_corr_dict_avg[s_corr_dict_avg["P-value"] < 0.01]

    s_corr_dict_avg["Distance"] = s_corr_dict_avg.apply(get_distance, args=(distances,), axis=1)

    s_corr_dict_avg.to_csv(date + "_full_corr.csv", index=False)

    network_df = s_corr_dict_avg[(s_corr_dict_avg["Spearman Correlation"] > 0.7) | (s_corr_dict_avg["Spearman Correlation"] < -0.7)].reset_index(drop = True)

    if network_df.empty == False:
        network_df["GEOHASH_1"] = [network_df["District Pair"].iloc[i][0] for i in range(len(network_df))]
        network_df["GEOHASH_2"] = [network_df["District Pair"].iloc[i][1] for i in range(len(network_df))]
        network_df["Distance"] = network_df.apply(get_dist, args=(distances,), axis=1)
        g = nx.Graph()
        for _ , row in network_df.iterrows():
            if row["Distance"] < 10:
                g.add_edge(row["GEOHASH_1"], row["GEOHASH_2"], dist=row["Distance"], weight=abs(row["Spearman Correlation"]), isNegative=row["Spearman Correlation"] < 0)
        nx.write_gexf(g, date+".gexf")
        completion_message = f"{date} completed"
        return completion_message

num_cores = 4

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore the FutureWarning
    results = Parallel(n_jobs=num_cores)(delayed(process_date)(date) for date in tqdm(dates, desc="Processing dates", postfix=""))