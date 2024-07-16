import planetary_computer
from netCDF4 import Dataset
import fsspec
import pystac_client
import numpy as np
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve
from bitstring import Bits, pack
from datetime import date, timedelta
import sys
from OpenVisus import *
base_date = date(1950, 1, 1)
present_date = date(2015, 1, 1)
last_date = date(2100, 12, 31)
cache_base='.azrcache'

try:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("nasa-nex-gddp-cmip6")
    variable_list=collection.summaries.get_list("cmip6:variable")
    model_list=collection.summaries.get_list("cmip6:model")[:10]
    scenario_list=collection.summaries.get_list("cmip6:scenario")
    have_pc = True
except pystac_client.exceptions.APIError:
    print("don't have planetary computer api access")
    have_pc = False

def _get_gddp_time_ranges(version):
    bits = Bits(uint=version, length=32)
    start_day, span_days = bits.unpack('uint:16, uint:16')
    start_date = base_date + timedelta(days = start_day)
    end_date = start_date + timedelta(days = span_days)
    if start_date > last_date:
        raise ValueError(f'WARNING: start_date of {start_date} is too far in the future')
    if end_date > last_date:
        print(f"WARNING: truncating end_date of {end_date} to {last_date}")
        end_date = last_date
    return start_date, end_date

def _get_gddp_params(name):
    model = 'CESM2'
    scenario = 'ssp585'
    variable = None
    var_name = name.split('\\')[-1]
    name_parts = var_name.split(',')
    for part in name_parts:
        if part[0] == 'm':
            model = part[2:]
            if have_pc and model not in model_list:
                raise ValueError(f"model {model} not available.") 
        if part[0] == 's':
            scenario = part[2:]
            if have_pc and scenario not in scenario_list:
                raise ValueError(f"scenario {scenario} not available.")
        if part[0] == 'v':
            variable = part[2:]
            if have_pc and variable not in variable_list:
                raise ValueError(f"variable {variable} not available.")
    if variable == None:
        raise ValueError('No variable name specified')
    return model, scenario, variable

def _get_dataset(url):
    path = urlparse(url).path
    cache_entry = f'{cache_base}/{path}'
    if not os.path.exists(cache_entry):
        cache_dir = os.path.dirname(cache_entry)
        if not os.path.exists(cache_dir):
            os.makedirs(os.path.dirname(cache_entry))
        urlretrieve(url, filename=cache_entry)
    return(Dataset(cache_entry))

def _get_cmip6_data(model, scenario, variable, start_date, end_date, lb, ub):
    model = "ACCESS-CM2"
    variable  = "tas" 

    year = 2020 
    # 2015 is the year whne the data switches from historical to simulated
    scenario = "historical" if year < 2015 else "ssp585"

    # Open (connect to) dataset
    dataset_name = f"{variable}_day_{model}_{scenario}_r1i1p1f1_gn"
    print(dataset_name)
    db = LoadDataset(f"http://atlantis.sci.utah.edu/mod_visus?dataset={dataset_name}&cached=arco")

    day_of_the_year = 202 
    timestep =year*365 + day_of_the_year
    quality = 0 
    data=db.read(time=timestep,quality=quality)
    result = data[0:256,0:256]

    return (result)

def query(name, version, lb, ub):
    start_date, end_date = _get_gddp_time_ranges(version)
    model, scenario, variable = _get_gddp_params(name)
    result = _get_cmip6_data(model, scenario, variable, start_date, end_date, lb, ub)
    return(result)

if __name__ == '__main__':
    s = date(2013, 5, 2)
    e = date(2013, 5, 2)
    start = (s - base_date).days
    span = (e - s).days
    lb = (0,0)
    ub = (599,1399)
    version = pack('uint:16, uint:16', start, span).uint
    res = query(name='cmip6-planetary\\m:ACCESS-ESM1-5,v:tas', version=1, lb=lb, ub=ub)
    print(res.shape)