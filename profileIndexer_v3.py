# -*- coding: utf-8 -*-

# import packages

import argparse
import csv
from datetime import datetime, timedelta
from dateutil import parser
import glob
from loguru import logger
import numpy as np
import os
import pandas as pd
import re
import s3fs
import subprocess
import xarray as xr

paramsDict = pd.read_csv("paramsDictionary.csv").set_index("profiler").T.to_dict("series")


def data_resample(ds, timeSpan):
    df = ds.to_dataframe().resample(timeSpan).mean()
    vals = [
        xr.DataArray(data=df[c], dims=["time"], coords={"time": df.index}, attrs=ds[c].attrs)
        for c in df.columns
    ]
    ds_resampled = xr.Dataset(dict(zip(df.columns, vals)), attrs=ds.attrs)

    return ds_resampled


def getLastIndex(indexFile):
    lastLine = subprocess.check_output(["tail", "-n", "1", indexFile], encoding="UTF-8")
    bits = lastLine.split(",")
    profileIndex = int(bits[0]) + 1
    startDate = parser.parse(bits[2])

    return profileIndex, startDate


def loadData_zarr(zarrDir):
    fs = s3fs.S3FileSystem(anon=True)
    zarr_store = fs.get_mapper(zarrDir)
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds


def loadData_local(localData):
    if ".nc" in localData:
        ds = xr.open_dataset(localData)
        ds = ds.swap_dims({"obs": "time"})
    else:
        fileList = glob.glob(localData + "/*.nc")
        if fileList:
            frames = [xr.open_dataset(file) for file in fileList]
            ds = frames[0]
            for idx, frame in enumerate(frames[1:], start=2):
                ds = xr.concat([ds, frame], dim="obs")
                ds = ds.swap_dims({"obs": "time"})
        else:
            raise ValueError("unable to open data in local directory")

    return ds


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Profile Indexer")
    arg_parser.add_argument("--profiler", type=str, default="RS01SBPS")
    arg_parser.add_argument("--dataSource", type=str, default="zarr")
    arg_parser.add_argument("--fileCreation", type=str, default="append")
    arg_parser.add_argument("--startDate", type=str, default="")
    arg_parser.add_argument("--endDate", type=str, default="")

    return arg_parser.parse_args()


def profileIndexer(ds, pressureVariable, profileIndex, threshold):
    logger.info("entering main profile Index loop")
    diffArray = np.diff(ds[pressureVariable])
    diffArray_times = ds.time[1:]
    diffArray_pressure = ds[pressureVariable][1:]

    diffArray_cast = diffArray[np.where(abs(diffArray) > threshold)]
    diffArray_cast_times = diffArray_times[np.where(abs(diffArray) > threshold)]

    idx = np.where(np.sign(diffArray_cast[:-1]) != np.sign(diffArray_cast[1:]))[0] + 1
    castState_change = diffArray_cast[idx]
    castState_change_times = diffArray_cast_times[idx]

    stopTimes = diffArray_times[np.where(abs(diffArray) < threshold)]
    stopPressure = diffArray_pressure[np.where(abs(diffArray) < threshold)]
    parkTimes = stopTimes[np.where(stopPressure > 180)].values

    profileList = []
    indexer = profileIndex
    castLength = 5
    iterationLength = len(castState_change)
    logger.info("entering profile index iteration loop with length of {}", iterationLength)
    for i in range(len(castState_change) - 1):
        if i % 100 == 0:
            logger.info("iteration: {}", i)
        if (castState_change[i] < 0) & (castState_change[i + 1] > 0):
            profile = True
            profileNumber = indexer
            profileStart = castState_change_times[i].values
            profilePeak = castState_change_times[i + 1].values
            parkTest = list(filter(lambda d: d > profileStart, parkTimes))
            if parkTest:
                if parkTest[0] < profilePeak:
                    ### false profile start...use end of parktime instead
                    parkEnd = list(filter(lambda d: d < profilePeak, parkTimes))
                    profileStart = min(parkEnd, key=lambda x: abs(x - profilePeak))
            ### peak no more than 5 hours after upcast start, else skip incomplete profile
            if (pd.Timestamp(profilePeak) - pd.Timestamp(profileStart)) < timedelta(
                hours=castLength
            ):
                parkTimes_filtered = list(filter(lambda d: d > profilePeak, parkTimes))
                if parkTimes_filtered:
                    profileEnd = min(parkTimes_filtered, key=lambda x: abs(x - profilePeak))
                    ### downcast end no more than 5 hours after peak, else skip incomplete profile
                    if (pd.Timestamp(profileEnd) - pd.Timestamp(profilePeak)) > timedelta(
                        hours=castLength
                    ):
                        profile = False
                else:
                    profile = False
            else:
                profile = False
            if profile:
                profileList.append(
                    [
                        profileNumber,
                        pd.Timestamp(profileStart),
                        pd.Timestamp(profilePeak),
                        pd.Timestamp(profileEnd),
                    ]
                )
                indexer = indexer + 1

    return profileList


def retrieveLatestFile(site):
    year = 2014
    for x in os.listdir():
        if x.endswith(".csv") and site in x and re.search("([0-9]{4})", x):
            yearNew = int(re.search("([0-9]{4})", x).group(1))
            if yearNew > year:
                year = yearNew
    currentFile = site + "_profiles_" + str(year) + ".csv"

    return currentFile


def main():
    args = parse_args()
    logger.add("logfile_profileIndexer_{time}.log")
    logger.info("profile Indexer initiated for {}", args.profiler)

    profilerDict = paramsDict[args.profiler]

    ### Load data file
    if "zarr" in args.dataSource:
        logger.info("loading zarr file")
        ds = loadData_zarr(profilerDict["zarrFile"])
    elif "gc_thredds" in args.dataSource:
        logger.info("loading gold copy")
        ds = load_gc_thredds(profilerDict["gc_thredds_dir"])
    else:
        logger.info("loading local data")
        ds = loadData_local(args.dataSource)

    ### drop unused variables
    vars = [profilerDict["pressureVariable"]]
    allVar = list(ds.keys())
    dropList = [item for item in allVar if item not in vars]
    ds = ds.drop(dropList)

    if "append" in args.fileCreation:
        ### if appending to file, load last current index
        logger.info("appending new indices for : {}", args.profiler)
        currentFile = retrieveLatestFile(args.profiler)
        (profileIndex, startDate) = getLastIndex(currentFile)
        logger.info("starting with profile {}, {}", profileIndex, startDate)
        ds = ds.sel(time=slice(startDate, datetime.utcnow()))
        append = True
    elif "create" in args.fileCreation:
        logger.info("creating new profile index file for {}", args.profiler)
        profileIndex = 1
        append = False
    elif "test" in args.fileCreation:
        profileIndex = 1
        indexFile = args.profiler + "_test.csv"
        logger.info(
            "creating test profile index file for {} for {} - {}",
            args.profiler,
            args.startDate,
            args.endDate,
        )
        ds = ds.sel(time=slice(parser.parse(args.startDate), parser.parse(args.endDate)))
        append = False

    ### resample and extract profile indices
    if ds:
        if "PS" in args.profiler:
            logger.info("resampling shallow profiler dataset to 1 minute bins")
            ds_resampled = data_resample(ds, "1Min")
            logger.info("determining profile indices")
            now = datetime.utcnow()
            profileList = profileIndexer(
                ds_resampled, profilerDict["pressureVariable"], profileIndex, 0.5
            )
            end = datetime.utcnow()
            logger.info("profile indexing completed, time elapse: {}", end - now)

        elif "PD" in args.profiler:
            if "CE04" in args.profiler:
                threshold = 1
            else:
                threshold = 2
            logger.info("processing deep profiler dataset")
            ds_resampled = data_resample(ds, "10s")
            ds_cleaned = ds_resampled.dropna(dim="time", how="any")
            logger.info("determining profile indices")
            now = datetime.utcnow()
            profileList = profileIndexer(
                ds_cleaned, profilerDict["pressureVariable"], profileIndex, threshold
            )
            end = datetime.utcnow()
            logger.info("profile indexing completed, time elapse: {}", end - now)

        if profileList:
            profile_dataframe = pd.DataFrame(
                profileList, columns=["profile", "start", "peak", "end"]
            )
            profile_dataframe["start"] = pd.to_datetime(profile_dataframe["start"])
            profileYears = set(profile_dataframe["start"].dt.year)
            if "test" in args.fileCreation:
                profile_dataframe.to_csv(
                    "{}_profiles_{}.csv".format(args.profiler, "test"), index=False
                )
            elif "append" in args.fileCreation:
                currentFileYear = int(re.search("([0-9]{4})", currentFile).group(1))
                for subYear in profileYears:
                    profiles_sub = profile_dataframe[
                        profile_dataframe["start"].dt.year == subYear
                    ]
                    profiles_sub_list = profiles_sub.values.tolist()
                    if subYear == currentFileYear:
                        with open(currentFile, "a") as csvfile:
                            write = csv.writer(csvfile)
                            write.writerows(profiles_sub_list)
                    else:
                        profiles_sub.to_csv(
                            "{}_profiles_{}.csv".format(args.profiler, str(subYear)),
                            index=False,
                        )
            elif "create" in args.fileCreation:
                for subYear in profileYears:
                    profiles_sub = profile_dataframe[
                        profile_dataframe["start"].dt.year == subYear
                    ]
                    profiles_sub.to_csv(
                        "{}_profiles_{}.csv".format(args.profiler, str(subYear)), index=False
                    )
        else:
            logger.info("no profiles detected")

    else:
        logger.info("no data to index...")


if __name__ == "__main__":
    main()
