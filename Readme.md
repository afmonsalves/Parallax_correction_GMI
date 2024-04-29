# GMI Parallax Correction Project

## Overview

This project performs a height-based parallax correction for the GPM Microwave Imager (GMI) Precipitation retrievals. The correction is achieved by utilizing various datasets including ERA5 temperature profiles and geopotential height data, MRMS instantaneous data, and GPROF level 2 data.

## Requirements

To run the code, you'll need the following data:

- ERA5 hourly temperature profiles and geopotential height data
- MRMS instantaneous data
- GPROF level 2 data

## Usage

To run the code, follow these steps:

1. Open a command line terminal.
2. Navigate to the project directory.
3. Run the following command:

```bash
python create_df.py --REGION# parallax_correction_GMI
# parallax_correction_GMI
