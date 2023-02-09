# %%
from pathlib import Path
import geopandas
from alceo.utils import in_notebook
from argparse import ArgumentParser

if not in_notebook():
    parser = ArgumentParser("change_from_annotations.py", description="""
Given a GeoJSON containing geometrical features describing pits location for various dates (in a field called Day_Month_Year with dd/mm/YYYY format).
Compute the pits that appeared, disappeared and persisted between two dates.
"""
    )
    parser.add_argument(
        "-i",
        "--input_geojson_path",
        type=Path,
        help="Input GeoJSON containing the geometries of the annotations done on the various images.",
        required=True,
    )
    parser.add_argument(
        "-f", "--first_image_date", type=str, help="Date (in dd/mm/YYYY format) of the first image.", required=True,
    )
    parser.add_argument(
        "-s", "--second_image_date", type=str, help="Date (in dd/mm/YYYY format) of the second image.", required=True,
    )
    parser.add_argument(
        "-o",
        "--output_directory_path",
        type=Path,
        help="The output directory where the `.appeared`, `.disappeared` and `.persisted` features will be saved. Defaults to current directory.",
        default='.',
    )
    parser.add_argument(
        "--crs",
        type=str,
        help="The output GeoJSON crs. Defaults to \"EPSG:32637\"",
        default="EPSG:32637",
    )

    args = parser.parse_args()
    input_geojson_path = args.input_geojson_path
    first_image_date = args.first_image_date
    second_image_date = args.second_image_date
    output_directory_path = args.output_directory_path
    crs = args.crs
else:
    input_geojson_path = Path("/home/gsech/Source/alceo/data/annotation/DURA/complete/DURA.geojson")
    first_image_date = "26/5/2013"
    second_image_date = "19/09/2014"
    output_directory_path = Path("/home/gsech/Source/alceo/data/annotation/DURA/complete")
    crs = "EPSG:32637"

# %%
_gdf = geopandas.read_file(input_geojson_path).to_crs(crs)
# %%
first_gpd: geopandas.GeoDataFrame = _gdf[_gdf.Day_Month_Year == first_image_date].reindex()
second_gpd: geopandas.GeoDataFrame = _gdf[_gdf.Day_Month_Year == second_image_date].reindex()

# # %% Visualizing the two GeoDataFrames
if in_notebook():
    _map = first_gpd.explore(
        style_kwds=dict(color="blue", opacity=0.3, weight=0.5),
        max_zoom=20,
    )
    second_gpd.explore(m=_map, style_kwds=dict(color="orange", opacity=0.3, weight=0.5))
    display(_map)

# %% Find intersections between pits of the first and second date!

intersecates = first_gpd.sjoin(
    second_gpd,
    predicate="intersects",
    how="inner",
)

intersecates_second = second_gpd.loc[intersecates.index_right]
intersecates_second.index = intersecates.index
intersecates_union = intersecates.union(intersecates_second)

# %%

# Compute Intersection over Union
intersecates["iou"] = (
    intersecates.intersection(intersecates_second).area / intersecates_union.area
)

# Relative intersection size wrt first area
intersecates["rel_intersection_first"] = (
    intersecates.intersection(intersecates_second).area / intersecates.area
)
# Relative intersection size wrt second area
intersecates["rel_intersection_second"] = (
    intersecates.intersection(intersecates_second).area / intersecates_second.area
)

# Compute pits that are considered "persisted"
persisted = intersecates[
    (intersecates.iou > 0.66)
    | (intersecates.rel_intersection_first > 0.68)
    | (intersecates.rel_intersection_second > 0.68)
]
intersecates = intersecates.drop(index=persisted.index)


# %% Dropping the pits that are considered "persisted"
disappeared = first_gpd.drop(persisted.index)
appeared = second_gpd.drop(persisted.index_right)

# %% Visualization
if in_notebook():
    _map = disappeared.explore(
        style_kwds=dict(color="red", opacity=0.3, weight=0.5, fill_opacity=0.2),
        max_zoom=20,
    )
    appeared.explore(
        m=_map, style_kwds=dict(color="green", opacity=0.3, weight=0.5, fill_opacity=0.2)
    )
    first_gpd.loc[persisted.index].explore(
        m=_map, style_kwds=dict(color="yellow", opacity=0.3, weight=0.5, fill_opacity=0.1)
    )
    second_gpd.loc[persisted.index_right].explore(
        m=_map, style_kwds=dict(color="purple", opacity=0.3, weight=0.5, fill_opacity=0.1)
    )
    display(_map)

# %% Save data
appeared_path = str(output_directory_path / input_geojson_path.with_suffix(".appeared"+input_geojson_path.suffix).name)
disappeared_path = str(output_directory_path / input_geojson_path.with_suffix(".disappeared"+input_geojson_path.suffix))
persisted_path = str(output_directory_path / input_geojson_path.with_suffix(".persisted"+input_geojson_path.suffix))
appeared.to_file(appeared_path, na="keep")
disappeared.to_file(disappeared_path, na="keep")
persisted.to_file(persisted_path, na="keep")
# %%
