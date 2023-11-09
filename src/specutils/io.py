__all__ = ["read_spectrum_pyraf", "fits2df", "read_spectrum_pyird", "read_flux_pyird"]

import numpy as np
import pandas as pd
from astropy.io import fits


def read_spectrum_pyraf(filename, df=True):
    """ read fits using pyraf.iraf.listpix

        Args:
            filename: fits file
            df: if True, return the dataframe; else wavelength, flux, header

        Returns:
            pandas dataframe containg the spectrum data, header (df=True)
            otherwise wavelength, flux, header

    """
    from pyraf import iraf
    header = fits.open(filename)[0].header
    spec = iraf.listpix(filename, wcs='world', Stdout=1)
    lam = np.array([float(spec[i].split()[0]) for i in range(len(spec))])
    flux = np.array([float(spec[i].split()[2]) for i in range(len(spec))])
    lam, flux = lam.reshape(-1, header['NAXIS1']), flux.reshape(-1, header['NAXIS1'])
    hdr = dict(header.items())
    print ("# input file:", filename)
    print ("# object name in header:", header["OBJECT"])
    if not df:
        return lam, flux, hdr
    return specdf(lam, flux), hdr
    #return header['MJD'], lam, flux, header["OBJECT"], header["RA"], header["DEC"]


def specdf(lam, flux):
    """ dataframe for spectrum data

        Args:
            lam: wavelength data, array of (# of orders, # of pixels)
            flux: flux data, array of (# of orders, # of pixels)

        Returns:
            pandas dataframe

    """
    orders, lams, fluxes = [], [], []
    for i in range(len(lam)):
        lams += list(lam[i])
        fluxes += list(flux[i])
        orders += [i]*len(flux[i])
    return pd.DataFrame(data={"order": orders, "lam": lams, "flux": fluxes})


def fits2df(filename, key):
    """ read fits spectrum

        Args:
            filename: fits file
            key: column keys

        Return:
            pandas dataframe

    """
    data = fits.open(filename)[0].data
    orders, vals = [], []
    for i in range(len(data.T)):
        vals += list(data[:,i])
        orders += [i] * len(data[:,i])
    return pd.DataFrame(data={"order": orders, key: vals}).fillna(0)


def read_spectrum_pyird(objfile, wavfile):
    header = fits.open(objfile)[0].header
    hdr = dict(header.items())
    data = read_flux_pyird(objfile)
    wavdata = fits2df(wavfile, "lam")
    data['lam'] = wavdata.lam
    return data, hdr


def read_flux_pyird(objfile):
    return fits2df(objfile, "flux")
