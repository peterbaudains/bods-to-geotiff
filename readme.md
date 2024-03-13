# BODS to GeoTIFF generator

This repository contains a script for generating GeoTIFF raster files from bus open data, provided by the UK DfT [Bus Open Data Service](https://www.bus-data.dft.gov.uk/).

The bus open data is accessed via a Neo4j database instance containing nodes of type `Observation`. Links between nodes on the same bus journey are provided as edge type `SAME_JOURNEY`. `Observation` nodes contain properties relating to the bus journey, observation time and location. The `SAME_JOURNEY` edges contain speed estimates between consecutive points.

The [rasterio](https://rasterio.readthedocs.io/en/stable/#) package is used to write the GeoTIFF files.

Example output data files from using this script can be found in the CUSP London [bus-open-data-rasters](https://github.com/cusp-london/bus-open-data-rasters) repository.
