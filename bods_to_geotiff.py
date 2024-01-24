from osgeo import gdal
import math
import rasterio
import rasterio.features
import rasterio.plot
import numpy as np
import datetime as dt
import logging
import geopandas as gpd
import time
import os
from neo4j import GraphDatabase
from dotenv import dotenv_values
config = dotenv_values(".env")
logging.basicConfig(filename='bods_to_geotiff.log', 
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
log = logging.getLogger()

def get_driver():
    uri=config['NEO4J_SERVER']
    auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD'])
    driver=GraphDatabase.driver(uri, auth=auth)
    return driver


def get_point_data(tx, recordedDateInt, recordedTimeIntMin, recordedTimeIntMax):
    if recordedTimeIntMax > recordedTimeIntMin:    
        CIPHER_QUERY = """
        MATCH (o1:Observation)-[s:SAME_JOURNEY]->(o2:Observation)
        WHERE o2.recordedDateInt = $recordedDateInt
        AND o2.recordedTimeInt >= $recordedTimeIntMin
        AND o2.recordedTimeInt < $recordedTimeIntMax
        RETURN o2.vehicleJourneyRef, o2.vehicleRef, o2.directionRef, o2.lineRef, o2.recordedTimeInt, o2.geometry, s.speed_ms
        """
        log.info(CIPHER_QUERY)
        log.info("%s - %s - %s" % (recordedDateInt, recordedTimeIntMin, recordedTimeIntMax))
        return tx.run(CIPHER_QUERY, 
                      recordedDateInt=recordedDateInt, 
                      recordedTimeIntMin=recordedTimeIntMin, 
                      recordedTimeIntMax=recordedTimeIntMax).to_df()
    elif recordedTimeIntMax == 0:
        CIPHER_QUERY = """
        MATCH (o1:Observation)-[s:SAME_JOURNEY]->(o2:Observation)
        WHERE o2.recordedDateInt = $recordedDateInt
        AND o2.recordedTimeInt >= $recordedTimeIntMin
        RETURN o2.vehicleJourneyRef, o2.vehicleRef, o2.directionRef, o2.lineRef, o2.recordedTimeInt, o2.geometry, s.speed_ms
        """
        return tx.run(CIPHER_QUERY, 
                      recordedDateInt=recordedDateInt, 
                      recordedTimeIntMin=recordedTimeIntMin).to_df()
    else:
        raise Exception("Check recordedTimeInt values going into neo4j query")


def write_raster(gdf, bounds, resolution, filename):
    
    transform = rasterio.transform.from_origin(
        west=bounds[0], 
        north=bounds[3], 
        xsize=resolution, 
        ysize=resolution)

    rows = math.ceil((bounds[3] - bounds[1]) / resolution)
    cols = math.ceil((bounds[2] - bounds[0]) / resolution)
    shape = (rows, cols)
    
    stacked_raster = np.zeros(shape, dtype=np.uint8)
    for ind, grp in gdf.groupby(['o2.vehicleJourneyRef', 'o2.vehicleRef', 'o2.directionRef', 'o2.lineRef']):
        route_raster = rasterio.features.rasterize(shapes=[(g,1) for g in grp.geometry], out_shape=shape, transform=transform)
        stacked_raster += route_raster
    
    with rasterio.open(config['OUTDIR'] + "/distinctJourneyCounts/"+filename, 'w', 
                    driver = 'GTiff',
                    height = stacked_raster.shape[0],
                    width = stacked_raster.shape[1],
                    count = 1,
                    dtype = stacked_raster.dtype,
                    crs = 27700,
                    transform = transform
    ) as dst:
        dst.write(stacked_raster, 1)

    g1 = [(g, 1) for g in gdf.geometry]
    raster_obs_count = rasterio.features.rasterize(
        shapes=g1, 
        out_shape=shape, 
        transform=transform, 
        merge_alg=rasterio.enums.MergeAlg.add, 
        dtype='uint16')

    g2 = [(g, v) for g, v in gdf[['geometry', 's.speed_ms']].values]
    raster_speed_agg = rasterio.features.rasterize(
        shapes=g2, 
        out_shape=shape, 
        transform=transform, 
        merge_alg=rasterio.enums.MergeAlg.add, 
        dtype='float32')

    raster_speeds = np.divide(raster_speed_agg.astype('float32'), 
                              raster_obs_count, 
                              out=-1 * np.ones(shape).astype('float32'), 
                              where=raster_obs_count!=0)

    with rasterio.open(config['OUTDIR'] + "/averageSpeeds/"+filename, 'w', 
                        driver = 'GTiff',
                        height = raster_speeds.shape[0],
                        width = raster_speeds.shape[1],
                        count = 1,
                        dtype = raster_speeds.dtype,
                        crs = 27700,
                        transform = transform
    ) as dst:
        dst.write(raster_speeds, 1)

    return

if __name__=="__main__":

    res = 50
    current_datetime = dt.datetime(2023, 12, 17, 0, 0)
    time_delta = dt.timedelta(0, 3600)
    end_date = dt.datetime(2024, 1, 9, 0, 0)
    bounds = [521055,169648, 547612,  188327]

    while current_datetime < end_date:
        t1 = time.time()
        driver = get_driver()
        with driver.session(database='busopendata') as session:
            log.info("Querying Neo4j for %s-%s"  % (current_datetime.strftime('%Y%m%d'), current_datetime.strftime('%H%M%S')))
            df = session.execute_read(get_point_data, 
                                      recordedDateInt=int(current_datetime.strftime('%Y%m%d')),
                                      recordedTimeIntMin=int(current_datetime.strftime('%H%M%S')), 
                                      recordedTimeIntMax=int((current_datetime + time_delta).strftime('%H%M%S')))
            log.info("%s records returned" % df.shape[0])
        
        if df.shape[0] > 0:
            gdf = gpd.GeoDataFrame(df, 
                                geometry=gpd.points_from_xy(
                                df['o2.geometry'].apply(lambda x: x[0]), 
                                df['o2.geometry'].apply(lambda x: x[1])), 
                                crs=4326)
            gdf = gdf.to_crs(27700)
            
            os.makedirs(config['OUTDIR'] + "distinctJourneyCounts/%d/%d/%d" % (current_datetime.year, current_datetime.month, current_datetime.day), exist_ok=True)
            os.makedirs(config['OUTDIR'] + "averageSpeeds/%d/%d/%d" % (current_datetime.year, current_datetime.month, current_datetime.day), exist_ok=True)
            filename = "%d/%d/%d/%s_%s_%s.gtiff" % (current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.strftime('%Y%m%d%H'), int(time_delta.total_seconds()), res)
            log.info("Writing raster data")
            write_raster(gdf,bounds, res, filename)
            
            log.info('Time taken for this time interval: %s' % (time.time() - t1))
        else:
            log.warning('No file found for %s' % current_datetime.strftime('%Y%m%d %H%M%S'))

        current_datetime += time_delta