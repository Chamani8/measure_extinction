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


def calc_ave_ext(starpair_list, path, min_number=1, mask=[]):
    extdatas = []
    for starpair in starpair_list:
        extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
        extdatas.append(extdata)
    average = AverageExtData(extdatas, min_number=min_number, mask=mask)
    average.save(path + "average_ext.fits")


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
