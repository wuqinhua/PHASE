#!/usr/bin/env python3

import math
import os
import pathlib
import numpy as np
import pandas
import scipy.sparse
import tqdm
import scanpy

os.chdir("/data/wuqinhua/phase/covid19")

if __name__ == "__main__":
    data = scanpy.read_h5ad("./Alldata_hvg.h5ad")
    meta = pandas.read_csv("./metadata.csv", sep=",",index_col=0)
    data.obs["predicted_labels"] = meta["predicted_labels"]
    print(data.obs)
  
    for state in set(data.obs["group"]):
        pathlib.Path(os.path.join("./Compare/data", state)).mkdir(parents=True, exist_ok=True)
    print("group_over")
    
    for ind in tqdm.tqdm(set(data.obs["sample_id"])):
        state = list(set(data.obs[data.obs["sample_id"] == ind]["group"]))
        assert(len(state) == 1)
        state = state[0]
        
        X = data.X[(data.obs["sample_id"] == ind).values, :].transpose()
        print("X_over")
        scipy.sparse.save_npz(os.path.join("./Compare/data", state, "{}.npz".format(ind)), X)
        np.save(os.path.join("./Compare/data", state, "ct_{}.npy".format(ind)),
                data.obs[data.obs["sample_id"] == ind]["predicted_labels"].values)
        print("save_over")
