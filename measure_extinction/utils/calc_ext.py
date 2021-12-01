#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData, AverageExtData


def calc_extinction(redstarname, compstarname, path, savepath="./", deredden=False):
    # read in the observed data for both stars
    redstarobs = StarData("%s.dat" % redstarname.lower(), path=path)
    compstarobs = StarData(
        "%s.dat" % compstarname.lower(), path=path, deredden=deredden
    )

    # calculate the extinction curve
    extdata = ExtData()
    extdata.calc_elx(redstarobs, compstarobs)
    extdata.save(
        savepath + "%s_%s_ext.fits" % (redstarname.lower(), compstarname.lower())
    )


def calc_ave_ext(
    starpair_list, path, outname="average_ext.fits", min_number=3, mask=[]
):
    """
    Calculate the average extinction curve

    Parameters
    ----------
    starpair_list : list of strings
        List of star pairs for which to calculate the average extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    outname : string [default="average_ext.fits"]
        Name of the output fits file with the average extinction curve

    min_number : int [default=3]
        Minimum number of extinction curves that are required to measure the average extinction; if less than min_number of curves are available at certain wavelengths, the average extinction will still be calculated, but the number of points (npts) at those wavelengths will be set to zero (e.g. used in the plotting)

     mask : list of tuples [default=[]]
        List of tuples with wavelength regions (in micron) that need to be masked, e.g. [(2.55,2.61),(3.01,3.10)]

    Returns
    -------
    Average extinction curve
    """
    extdatas = []
    for starpair in starpair_list:
        extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
        extdatas.append(extdata)
    average = AverageExtData(extdatas, min_number=min_number, mask=mask)
    average.save(path + outname)


def main():
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("redstarname", help="name of reddened star")
    parser.add_argument("compstarname", help="name of comparison star")
    parser.add_argument(
        "--path",
        help="path to data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    parser.add_argument(
        "--deredden",
        help="deredden standard based on DAT file dered parameters",
        action="store_true",
    )
    args = parser.parse_args()

    calc_extinction(
        args.redstarname, args.compstarname, args.path, deredden=args.deredden
    )


if __name__ == "__main__":
    main()
