from arkitekt_next import register, startup, context
from mikro_next.api.schema import from_parquet_like, Image, Table
import time
import tifffile
import pandas as pd

@context
class MatlabContext:
    app: any

@startup
async def startup() -> MatlabContext:
    app = matlab.initialize()

    return MatlabContext(app=app)


@register
def run_segmantik_segmentation(context: MatlabContext, n: Image) -> Table:
    
    # TODO: Select the correct dimension of the image (dims: c, t, z, y, x)
    image_data = n.data.sel(t=0)

    # Convert the image data to a tiff file
    tifffile.imwrite('image.tif', image_data)

    # Run the segmentation algorithm
    file_path_of_csv = context.app.run_segmentation('image.tif')

    # Load the csv file
    df = pd.read_csv(file_path_of_csv)

    # Convert the csv file to a Table
    table = from_parquet_like(df)

    # TODO: Assign meaning to the columns of the table

    return table


