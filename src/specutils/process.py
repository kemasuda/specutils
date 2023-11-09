__all__ = ["median_filtered_flux", "compute_ccf", "iterative_fit",
           "barycentric_correction", "barycentric_correction_pyasl"]

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.signal import correlate


def median_filtered_flux(df, norder, filter_size=31):
    """ flux smoothing using median filter

        Args:
            df: spectrum dataframe
            norder: number of orders
            filter_size: window size for median filter

        Returns:
            1d array for smooethed flux

    """
    smoothed_flux = []
    for i in range(norder):
        order_idx = df.order == i
        smoothed_flux += list(median_filter(
            df.flux[order_idx], size=filter_size, mode='constant'))
    return smoothed_flux


c_in_kms = 2.99792458e5


def compute_ccf(x, y, xmodel, ymodel, mask=None, oversample_factor=5):
    """ compute cross-correlation function with model (copied from jaxspec.utils)

        Args:
            x: data wavelength
            y: data flux
            xmodel: model wavelength
            ymodel: model flux
            mask: data to be masked
            oversample_factor: ovesampling factor for the data

        Returns:
            velgrid: velocity grid (km/s)
            ccf: CCF values as a function of velocity

    """
    if mask is None:
        mask = np.zeros_like(y).astype(bool)
    yy = np.array(y)
    yy[mask] = np.nan

    ndata = len(x)
    xgrid = np.logspace(np.log10(x.min())+1e-4,
                        np.log10(x.max())-1e-4, ndata*oversample_factor)
    ygrid = interp1d(x, yy)(xgrid) - np.nanmean(yy)  # 1.
    ymgrid = interp1d(xmodel, ymodel)(xgrid) - np.nanmean(ymodel)
    ygrid[ygrid != ygrid] = 0

    ccf = correlate(ygrid, ymgrid)
    logxgrid = np.log(xgrid)
    dlogx = np.diff(logxgrid)[0]
    velgrid = (np.arange(len(ccf))*dlogx -
               (logxgrid[-1]-logxgrid[0])) * c_in_kms

    return velgrid, ccf


def iterative_fit(x, y, order, nsigma=[1., 3.], maxniter=10):
    """ polynomial fitting with iterative & asymmetric sigma clipping
    """
    A = np.vander(x, order+1)
    idx = np.ones_like(x) > 0
    for i in range(maxniter):
        w = np.linalg.solve(np.dot(A[idx].T, A[idx]), np.dot(A[idx].T, y[idx]))
        mu = np.dot(A, w)
        res = y - mu
        sigma = np.sqrt(np.median(res**2))
        idx_new = (res > -nsigma[0]*sigma) & (res < nsigma[1]*sigma)
        if np.sum(idx) == np.sum(idx_new):
            idx = idx_new
            break
        idx = idx_new
    return mu, w


def barycentric_correction_pyasl(times, ra, dec, location='maunakea', raunit='ha', decunit='deg'):
    """ barycentric correction using Pyastronomy;
    different from astorpy version by: a few sec (bjd) and a few m/s (bcc)

        Args:
            times: time (MJD)
            ra: right ascension (hour angle)
            dec: declination (deg)

        Returns:
            arrays of heliocentric julian days, velocity corrections

    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import PyAstronomy.pyasl as pyasl

    # Coordinates of Mauna Kea Observatory (Coordinates of UT1)
    if location == 'maunakea':
        longitude, latitude, altitude = -155.47333, 19.82444, 4205
    elif location == 'calaralto':
        longitude, latitude, altitude = 37.220791, -2.546847, 2168
    elif location == 'okayama':
        longitude, latitude, altitude = 133.59464318378573, 34.576238359684375, 372
    print("assumed location for barycentric correction:", location)

    c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
    """
    if raunit=='ha' and decunit=='deg':
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
    elif raunit=='ha' and decunit=='ha':
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.hourangle))
    elif raunit=='deg' and decunit=='deg':
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
    else:
        print ("This RA/DEC format is not supported.")
    """
    ra2000, dec2000 = c.ra.deg, c.dec.deg
    print("coordinates (J2000):", ra2000, dec2000)

    corrs, hjds = [], []
    for i in range(len(times)):
        jd = times[i] + 2400000.5  # obs.time: MJD to JD

        # Calculate barycentric correction (debug=True show various intermediate results)
        corr, hjd = pyasl.helcorr(
            longitude, latitude, altitude, ra2000, dec2000, jd, debug=False)

        corrs.append(corr)
        hjds.append(hjd)

    return np.array(hjds), None, np.array(corrs)


def barycentric_correction(times, ra, dec, location='maunakea', epoch='J2000', time_format='mjd'):
    """ barycentric correction using astropy https://docs.astropy.org/en/stable/time/

        Args:
            times: time (MJD)
            ra: right ascension (hour angle)
            dec: declination (deg)

        Returns:
            heliocentric julian days (UTC)
            barycentric julian days (TDB)
            radial velocity corrections (km/s)

    """
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    if location == 'maunakea':
        longitude, latitude, altitude = -155.47333, 19.82444, 4205
    elif location == 'calaralto':
        longitude, latitude, altitude = 37.220791, -2.546847, 2168
    elif location == 'okayama':
        longitude, latitude, altitude = 133.59464318378573, 34.576238359684375, 372
    else:
        raise Exception("location %s not supported." % location)
    print("assumed location for barycentric correction:", location)

    loc = EarthLocation.from_geodetic(
        longitude*u.deg, latitude*u.deg, height=altitude*u.m)
    t = Time(times, format=time_format)
    c = SkyCoord(ra, dec, frame='icrs', unit=(
        u.hourangle, u.deg), obstime=Time(epoch))
    print("coordinates (J2000):", c.ra.deg, c.dec.deg)

    rv_correction = c.radial_velocity_correction(
        kind='barycentric', obstime=t, location=loc).value * 1e-3

    t_corr = Time(times, format=time_format, location=loc)
    ltt_helio = t_corr.light_travel_time(c, 'heliocentric')
    ltt_bary = t_corr.light_travel_time(c, 'barycentric')

    times_hjd = t_corr.utc.jd + ltt_helio
    times_bjd = t_corr.tdb.jd + ltt_bary

    return np.array(times_hjd.value), np.array(times_bjd.value), np.array(rv_correction)
