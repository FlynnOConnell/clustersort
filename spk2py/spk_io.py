import h5py
import numpy as np
from collections import namedtuple

Segment = namedtuple("Segment", ["segment_number", "data"])
UnitData = namedtuple("UnitData", ["slices", "times"])

def load_from_h5(filename):
    # Dictionary to hold the loaded data
    data_dict = {}
    with h5py.File(filename, "r") as f:
        # Load metadata
        metadata_grp = f["metadata"]
        data_dict["metadata"] = {
            attr: metadata_grp.attrs[attr] for attr in metadata_grp.attrs
        }

        # Load unit data
        unit_grp = f["unit"]
        data_dict["unit"] = {}

        for title in unit_grp.keys():
            channel_grp = unit_grp[title]
            segments = []

            for segment_name in channel_grp.keys():
                segment_grp = channel_grp[segment_name]

                slices = np.array(segment_grp["slices"])
                times = np.array(segment_grp["times"])

                # Assuming Segment and UnitData are namedtuples
                segment = Segment(
                    segment_number=int(segment_name.split("_")[1]),
                    data=UnitData(slices=slices, times=times),
                )
                segments.append(segment)

            data_dict["unit"][title] = segments

    return data_dict
