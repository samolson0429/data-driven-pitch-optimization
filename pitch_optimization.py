#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kneed import KneeLocator


# ## Load in Files
# Note that CSV files are not included in this repository due to their size.
# See data/data_instructions.txt for download and setup instructions.

# In[2]:


names = pl.read_csv("data/player_names.csv")
atbats = pl.read_csv("data/atbats.csv")
pitches = pl.read_csv("data/pitches.csv", ignore_errors = True)


# ## Wrangling

# In[3]:


## Select columns from pitches dataset
pitches = pitches[["px", "pz", "start_speed", "end_speed", "spin_rate", "spin_dir", "break_angle", "break_length", 
                   "break_y", "zone", "pitch_type", "ab_id", "pitch_num"]]


# In[4]:


## Create Full Name column in the names dataset
names = names.with_columns((pl.col("first_name") + " " + pl.col("last_name")).alias("Full Name"))


# In[5]:


## Join atbats column with the names column to get batter name and create main dataset
baseball = atbats[["ab_id", "batter_id", "event", "p_throws", "pitcher_id", "stand"]].join(
    names[["id", "Full Name"]].rename({"Full Name":"Batter Name"}), left_on = "batter_id", right_on = "id")


# In[6]:


## Join baseball dataset to names dataset again to get pitcher name
baseball = baseball.join(
    names[["id", "Full Name"]].rename({"Full Name":"Pitcher Name"}), left_on = "pitcher_id", right_on = "id")


# In[7]:


## Filter pitches to be the final pitch of the at-bat to get the outcome
pitches = pitches.with_columns(
    pl.col("pitch_num").max().over("ab_id").alias("max_pitch_num"), pl.col("ab_id").cast(pl.Int64)
).filter(pl.col("pitch_num") == pl.col("max_pitch_num")).drop("max_pitch_num")


# In[8]:


## Join pitches with main dataset to get result of at-bat
baseball = baseball.join(pitches, on = "ab_id")


# In[9]:


## Filter to only players that have had over 1000 at-bats
baseball = baseball.with_columns(
    pl.col("event").count().over("Batter Name").alias("atbats_total")
).filter(pl.col("atbats_total") >= 1000).drop("atbats_total")


# In[10]:


## Filter only to events that factor into batter average
baseball = baseball.with_columns(
    pl.when(pl.col("event").is_in([
        "Batter Interference", "Catcher Interference", "Field Error", "Hit By Pitch",
        "Intent Walk", "Sac Bunt", "Sac Fly", "Sac Fly DP", "Sacrifice Bunt DP", "Walk"
    ]))
    .then(pl.lit("Other"))
    .otherwise(
        pl.when(pl.col("event").is_in(["Home Run", "Triple", "Single", "Double"]))
        .then(pl.lit("Hit"))
        .otherwise(pl.lit("Out"))
    )
    .alias("Hit_Category")
).filter(pl.col("Hit_Category") != "Other")


# In[11]:


## Create numeric column for getting a hit
baseball = baseball.with_columns(pl.when(pl.col("Hit_Category") == "Hit").then(1).otherwise(0).alias("Hit_Numeric"))


# In[12]:


## Create pitch name column
baseball = baseball.with_columns(
    pl.when(pl.col("pitch_type") == "CH").then(pl.lit("Changeup"))
    .when(pl.col("pitch_type") == "CU").then(pl.lit("Curveball"))
    .when(pl.col("pitch_type") == "EP").then(pl.lit("Eephus"))
    .when(pl.col("pitch_type") == "FC").then(pl.lit("Cutter"))
    .when(pl.col("pitch_type") == "FF").then(pl.lit("Four-seam Fastball"))
    .when(pl.col("pitch_type") == "FO").then(pl.lit("Pitchout"))
    .when(pl.col("pitch_type") == "FS").then(pl.lit("Splitter"))
    .when(pl.col("pitch_type") == "FT").then(pl.lit("Two-seam Fastball"))
    .when(pl.col("pitch_type") == "IN").then(pl.lit("Intentional Ball"))
    .when(pl.col("pitch_type") == "KC").then(pl.lit("Knuckle Curve"))
    .when(pl.col("pitch_type") == "KN").then(pl.lit("Knuckleball"))
    .when(pl.col("pitch_type") == "PO").then(pl.lit("Pitchout"))
    .when(pl.col("pitch_type") == "SC").then(pl.lit("Screwball"))
    .when(pl.col("pitch_type") == "SI").then(pl.lit("Sinker"))
    .when(pl.col("pitch_type") == "SL").then(pl.lit("Slider"))
    .when(pl.col("pitch_type") == "UN").then(pl.lit("Unknown"))
    .otherwise(pl.lit("Unknown"))
    .alias("pitch_name")
)


# ## Create K-Means Algorithm

# In[13]:


## Create Algorithm that runs through each player and clusters the pitches thrown to them

## Create cluster dataset
cluster_df = pl.DataFrame()

## Loop through each batter and pitcher dominant hand
for player_name in set(baseball["Batter Name"]):
    
    ## Filter to player
    player = baseball.filter(pl.col("Batter Name") == player_name)
    
    for dom_hand in set(baseball["p_throws"]):

        ## Filter to dominant hand and drop nulls
        player2 = player.filter(pl.col("p_throws") == dom_hand).drop_nulls(
            subset = ["px", "pz", "end_speed", "spin_rate", "spin_dir", "break_angle", "break_length"])

        ## Create array with columns
        player_numpy = player2[["px", "pz", "end_speed", "spin_rate", 
                             "spin_dir", "break_angle", "break_length", "break_y"]].to_numpy()

        ## Scale columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(player_numpy)

        ## Loop from 10 to 30 clusters and find optimal clusters for each player
        wcss = []
        k_values = range(10,31)

        for k in k_values:

            ## K-means algorithm
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            kmeans.fit(X_scaled)
            
            wcss.append(kmeans.inertia_)

        ## Find optimized cluster
        kn = KneeLocator(k_values, wcss, curve = "convex", direction = "decreasing")

        ## Create Alogrithm with optimized cluster
        kmeans = KMeans(n_clusters=kn.elbow, n_init='auto', random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        ## Create cluster column 
        player2 = player2.with_columns(pl.Series("cluster_label", cluster_labels))

        ## Append to cluster dataset
        cluster_df = pl.concat([cluster_df, player2])


# ## Write to csv

# In[14]:


cluster_df.write_csv("baseball.csv")

