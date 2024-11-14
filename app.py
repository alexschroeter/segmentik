from arkitekt_next import register, startup, context
from mikro_next.api.schema import from_parquet_like, Image, Table
import time
import tifffile
import pandas as pd

import SemSeg
import matlab
from dataclasses import dataclass

@context
@dataclass
class MatlabContext:
    app: any

@startup
async def startup(instance_id) -> MatlabContext:
    app = SemSeg.initialize()

    return MatlabContext(app=app)


@register
def run_segmantik_segmentation(context: MatlabContext, n: Image) -> Table:
    
    # TODO: Select the correct dimension of the image (dims: c, t, z, y, x)
    image_data = n.data.sel(c=0,t=0)

    # Convert the image data to a tiff file
    tifffile.imwrite('image.tif', image_data)

    # Run the segmentation algorithm
    context.app.SemanticSegmentation(
        "/app/image.tif",
        "/app/results/csvFiles",
        "Kraemer2024A-DS0008TP0005DR_D1_CH0001PL_NS",
        matlab.double(1),
        matlab.double(100),
        matlab.double(0.5),
        matlab.double(245),
        matlab.double(1000),
        "yes",nargout=0
        )

    # Load the csv file
    df = pd.read_csv("/app/results/csvFiles/points.csv", header=None)
    df.columns =["X", "Y", "Z"] 

    # Convert the csv file to a Table
    table = from_parquet_like(df, name="sem_seg_points")

    return table
