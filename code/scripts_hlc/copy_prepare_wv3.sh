#!/usr/bin/env bash

# copy and prepare world view 3 images

org_dir=~/Data/super_resolution/SpaceNet/AOI_5_Khartoum_Merge_Train_Test/RGB-PanSharpen

for tif in $(ls ${org_dir}/*.tif); do
    #reproject to "WGS 84 / UTM zone 36N", same as the Planet images
    # resample the spatial resolution as 1.5 meters

    filename=$(basename -- "$tif")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo $filename

    #gdalwarp -overwrite -s_srs EPSG:4326 -t_srs EPSG:32636 -r cubic -tr 1.5 1.5 $tif  ${filename}_1.5m.tif
    gdal_contrast_stretch -percentile-range 0.01 0.99 ${filename}_1.5m.tif ${filename}_1.5m_8bit.tif

    rm ${filename}_1.5m.tif
    #exit
done