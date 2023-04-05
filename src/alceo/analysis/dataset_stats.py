# %%

import geopandas as gpd
from pathlib import Path

dataset_path = Path("/HDD1/gsech/source/alceo/dataset/pits/EBLA")
vectorial_gdf = gpd.read_file(dataset_path / "vectorial.geojson")
vectorial_gdf.groupby(["change", "change_kind"]).OBJECTID.count()

# %%
"""
DURA
2013-2014
    appeared: 3328
    disappeared: 507

ASWAN 
2005-2015
    appeared: 303
    disappeared: 158
2005-2018
    appeared: 479
    disappeared: 158
2015-2018
    appeared: 333
    disappeared 158
    
EBLA
2004-2014
    appeared: 22
    disappeared: 4
2014-2018
    appeared: 9
    disappered 22
2018-2022
    appeared: 1
    disappeared 9
"""