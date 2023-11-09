__all__ = ["Spectrum"]


import numpy as np
from .io import *
from .process import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 15


class Spectrum:
    def __init__(self):
        self.data = None
        self.header = None
        self.norder = None
        self.blaze = None
        self.telluric = None
        self.sky = None
        self.model = None
        self.mask = None
        self.read_func = read_spectrum_pyraf

    def add_data(self, filename, data_filter_size=31, wavfile=None, pyird=False, location='maunakea'):
        if not pyird:
            data, hdr = self.read_func(filename)
            order_key = 'NAXIS2'
        else:
            if wavfile is None:
                raise Exception(
                    "# a separate wavelength file is need for pyird data.")
            else:
                data, hdr = read_spectrumm_pyird(filename, wavfile)
                order_key = 'NAXIS1'

        try:
            self.norder = hdr[order_key]
        except:
            print("# %s: no %s in header." % (filename, order_key))
            self.norder = data.order.max() + 1

        try:
            #hjd, bjd, bcc = barycentric_correction([hdr['MJD']], hdr['RA'], hdr['DEC'], location=location, raunit=raunit, decunit=decunit)
            hjd, bjd, bcc = barycentric_correction(
                hdr['MJD'], hdr['RA'], hdr['DEC'], location=location)
        except:
            print("# %s: barycentric correction failed." % filename)
            hjd, bjd, bcc = np.nan, np.nan, np.nan

        data['smoothed_flux'] = median_filtered_flux(
            data, self.norder, filter_size=data_filter_size)
        data['smoothed_flux_neg'] = median_filtered_flux(
            data, self.norder, filter_size=5)
        tmp_flux = np.where(data.flux > 0, data.flux, 1)
        data['flux_error'] = np.where(
            data.flux > 0, 1. / np.sqrt(tmp_flux), np.inf)
        data['hjd'] = np.ones_like(data.flux) * hjd
        data['bjd'] = np.ones_like(data.flux) * bjd
        data['bcc'] = np.ones_like(data.flux) * bcc
        self.data = data
        self.header = hdr

    def add_telluric(self, df):
        self.telluric = df

    def add_sky(self, filename, pyird=False):
        if not pyird:
            sky, _ = self.read_func(filename)
        else:
            sky = read_flux_pyird(filename, "flux")
            sky["lam"] = self.data.lam
        sky['smoothed_flux'] = median_filtered_flux(
            sky, self.norder, filter_size=301)
        sky['normed_flux'] = sky['flux'] / sky['smoothed_flux']
        self.sky = sky

    def add_blaze(self, filename, pyird=False):
        if not pyird:
            blaze, _ = self.read_func(filename)
        else:
            blaze = read_flux_pyird(filename)
            blaze["lam"] = self.data.lam
        blaze['smoothed_flux'] = median_filtered_flux(blaze, self.norder)
        self.blaze = blaze
    
    def blaze_normalize(self, max_iteration=5, nsigma_low=1., nsigma_high=3.):
        if self.blaze is None:
            print("Blaze is not loaded. Just normalizing data.")
            normed_flux = []
            for i in range(self.norder):
                idx = self.data.order == i
                original_data, smoothed_data = self.data.flux[idx], self.data.smoothed_flux[idx]
                normed_flux += list(original_data /
                                    np.nanmedian(smoothed_data))
            self.data['normed_flux'] = normed_flux
        else:
            normed_flux = []
            for i in range(self.norder):
                idx = self.data.order == i
                original_data, smoothed_data, smoothed_blaze \
                    = np.array(self.data.flux[idx]), np.array(self.data.smoothed_flux[idx]), np.array(self.blaze.smoothed_flux[idx])
                idx_itr = np.ones_like(original_data) > 0.
                for j in range(max_iteration):
                    coeff = np.dot(smoothed_data, smoothed_blaze) / \
                        np.dot(smoothed_blaze, smoothed_blaze)
                    res = smoothed_data - coeff * smoothed_blaze
                    res_sigma = np.std(res)#1.4826 * np.median(np.abs(res))
                    idx_sigma = (-nsigma_low * res_sigma < res) & (res < nsigma_high * res_sigma)
                    if np.sum(idx_itr) == np.sum(idx_sigma):
                        break
                    idx_itr = idx_sigma
                normed_flux += list(original_data / (coeff * smoothed_blaze))
            self.data['normed_flux'] = normed_flux

    def add_masks(self, sky_threshold=2., telluric_threshold=0.98, pct_threshold=5., out_threshold=5., manual_masks=None, mask_negative_outliers=False):
        from scipy.interpolate import interp1d
        # mask sky
        if self.sky is None:
            print("no sky data.")
            self.data['sky_mask'] = np.zeros(
                len(self.data['lam'])).astype(bool)
        else:
            #self.data['sky_mask'] = self.sky['flux'] / self.sky['smoothed_flux'] > sky_threshold
            skymask = []
            for i in range(self.norder):
                idx = self.data.order == i
                x = np.r_[-np.inf, self.sky['lam'][idx], np.inf]
                y = np.r_[
                    1., (self.sky['flux']/self.sky['smoothed_flux'])[idx], 1.]
                func_normed_sky = interp1d(x, y)
                skymask += list(func_normed_sky(
                    self.data['lam'][idx]) > sky_threshold)
            self.data['sky_mask'] = skymask

        # mask telluric
        if self.telluric is None:
            print("no telluric data.")
            self.data['telluric_mask'] = np.zeros(
                len(self.data['lam'])).astype(bool)
        elif self.telluric['lam'].min() > self.data['lam'].min() or self.telluric['lam'].max() < self.data['lam'].max():
            print("check wavelengths of telluric data.")
            self.data['telluric_mask'] = np.zeros(
                len(self.data['lam'])).astype(bool)
        else:
            func_telluric = interp1d(
                self.telluric['lam'], self.telluric['flux'])
            self.data['telluric_mask'] = func_telluric(
                self.data['lam']) < telluric_threshold

        # mask edges of each order
        if self.blaze is None:
            mask = np.zeros(len(self.data['lam']))
            nmask = len(self.data['lam']) // self.norder // 400
            print("no blaze data: masking %d edge pixels in each order." % nmask)
            pct_mask = []
            for i in range(self.norder):
                idx = self.data.order == i
                mask = np.zeros(np.sum(idx))
                mask[:nmask] = 1.
                mask[-nmask:] = 1.
                pct_mask += list((mask > 0).astype(bool))
                #original_data, smoothed_data = self.data.flux[idx], self.data.smoothed_flux[idx]
                #pct_mask += list(smoothed_data < np.percentile(smoothed_data, pct_threshold))
            self.data['pct_mask'] = pct_mask
        else:
            pct_mask = []
            for i in range(self.norder):
                idx = self.blaze.order == i
                smoothed_blaze = self.blaze.smoothed_flux[idx]
                pct_mask += list(smoothed_blaze <
                                 np.percentile(smoothed_blaze, pct_threshold))
            self.data['pct_mask'] = pct_mask

        # mask positive outliers
        out_mask = []
        for i in range(self.norder):
            df = self.data
            idx = df.order == i
            flux, smoothed_flux = df.flux[idx], df.smoothed_flux[idx]
            mad = np.median(np.abs(flux - smoothed_flux))
            posmask = flux-smoothed_flux > out_threshold * mad
            if not mask_negative_outliers:
                out_mask += list(posmask)
            else:
                smoothed_flux_neg = df.smoothed_flux_neg[idx]
                negmask = smoothed_flux_neg-flux > out_threshold * mad
                out_mask += list(posmask+negmask)
        self.data['out_mask'] = out_mask
        #self.data['out_mask2'] = self.data['flux'] / self.data['smoothed_flux'] > 2

        if manual_masks is None:
            print("no manual masks.")
            self.data['man_mask'] = np.zeros(
                len(self.data['lam'])).astype(bool)
        else:
            print("manually masked ranges:", manual_masks)
            man_mask = []
            for i in range(self.norder):
                df = self.data
                idx = df.order == i
                wavs = np.array(df.lam[idx])
                man_mask_order = np.zeros_like(wavs)
                for mask in manual_masks:
                    wmin, wmax = mask
                    man_mask_order += ((wmin < wavs) & (wavs < wmax))
                man_mask += list(man_mask_order > 0.)
            self.data['man_mask'] = man_mask

        self.data['all_mask'] = self.data.sky_mask | self.data.telluric_mask | self.data.pct_mask | self.data.out_mask | self.data.man_mask
        #self.data['all_mask2'] = self.data.sky_mask | self.data.telluric_mask | self.data.pct_mask | self.data.out_mask2

    def check_masks(self, savedir=None):
        df = self.data

        if 'all_mask' not in df.keys():
            print("# masks not defined.")
            return None

        for order in range(self.norder):
            idx = df.order == order
            x, y, ys, yn, smask, tmask, pmask, omask, allmask = np.array(
                df[idx][['lam', 'flux', 'smoothed_flux', 'normed_flux', 'sky_mask', 'telluric_mask', 'pct_mask', 'out_mask', 'all_mask']]).T
            smask, tmask, pmask, omask, allmask \
                = smask.astype(bool), tmask.astype(bool), pmask.astype(bool), omask.astype(bool), allmask.astype(bool)

            if self.telluric is not None:
                idxt = (x[0] < self.telluric.lam) & (x[-1] > self.telluric.lam)
                xt, yt = np.array(self.telluric[idxt][['lam', 'flux']]).T
            else:
                xt, yt = None, None

            if self.sky is not None:
                xsky, ysky = np.array(self.sky[idx][['lam', 'normed_flux']]).T
            else:
                xsky, ysky = None, None

            frac_removed = np.sum(allmask) / len(x)
            if frac_removed > 0.4:
                color = 'crimson'
            else:
                color = 'k'

            plt.figure(figsize=(16*0.8, 7*0.8))
            plt.xlabel('wavelength (nm)')
            plt.ylabel('normalized flux')
            plt.title("order %d, removed: %d%s" %
                      (order, frac_removed*100, "%"), fontdict={'color': color})
            plt.ylim(0., 2)
            plt.plot(x, yn, alpha=0.2, color='gray')
            plt.plot(x[~allmask], yn[~allmask], '.', color='gray')
            if xt is not None:
                plt.plot(xt, yt*0.5, color='darkred', lw=0.2)
            if xsky is not None:
                plt.plot(xsky, 1e-2*(ysky-1.)+1, color='darkblue', lw=0.2)
            plt.plot(x[smask], yn[smask], 'o', mfc='none',
                     color='darkblue', label='sky')
            plt.plot(x[omask], yn[omask], 's', mfc='none',
                     color='C0', label='outlier')
            plt.plot(x[tmask], yn[tmask], '*', mfc='none',
                     color='darkred', label='telluric')
            plt.plot(x[pmask], yn[pmask], 'v',
                     mfc='none', label='edge', color='C2')
            plt.legend(loc='best', bbox_to_anchor=(1, 1))
            if savedir is not None:
                plt.savefig(savedir+"order%02d.png" %
                            order, dpi=200, bbox_inches="tight")
                plt.close()

    def ccf_telluric(self, plot=False, vlim=None, oversample_factor=5):
        ccfvels, ccfs = [], []
        for i in range(self.norder):
            oidx = self.data.order == i
            x, y = self.data.lam[oidx], self.data.normed_flux[oidx]
            mask = (self.data.sky_mask | self.data.pct_mask |
                    self.data.out_mask | self.data.man_mask)[oidx]
            moidx = (self.telluric.lam > x.min()) & (
                self.telluric.lam < x.max())
            xmodel, ymodel = self.telluric.lam[moidx], self.telluric.flux[moidx]
            velgrid, ccf = compute_ccf(
                x, y, xmodel, ymodel, mask=mask, oversample_factor=oversample_factor)

            if vlim is not None:
                vidx = np.abs(velgrid) < vlim
                velgrid, ccf = velgrid[vidx], ccf[vidx]

            ccfvel = velgrid[np.argmax(ccf)]

            if plot:
                plt.figure()
                plt.xlabel('velocity (km/s)')
                plt.ylabel('normalized CCF')
                plt.title('order %d' % i)
                plt.xlim(velgrid.min(), velgrid.max())
                plt.plot(velgrid, ccf/np.max(ccf), lw=1)
                plt.axvline(x=velgrid[np.argmax(ccf)], color='gray',
                            lw=3, alpha=0.2, label='%.1fkm/s' % ccfvel)

            ccfs.append(ccfs)
            ccfvels.append(ccfvel)

        return np.array(ccfvels), ccfs

    def check_telluric_rvs(self, telluric_orders=[1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 15, 16], plot=False, vlim=None, oversample_factor=10, savedir=None):
        rvs, _ = self.ccf_telluric(
            plot=plot, vlim=vlim, oversample_factor=oversample_factor)

        vmean, vstd = np.mean(rvs[telluric_orders]), np.std(
            rvs[telluric_orders])
        """
        vpixs = []
        for n in range(spec.norder):
            s = spec.data[spec.data.order==n]
            vpix = np.mean(3e5*np.diff(s.lam)/s.lam[1:])
            vpixs.append(vpix)
        vpixs = np.array(vpixs)
        """
        plt.figure()
        plt.xlabel("order")
        plt.ylabel("telluric RV (km/s)")
        plt.plot(telluric_orders, rvs[telluric_orders], '*',
                 label='mean: %.2fkm/s, SD: %.2fkm/s' % (vmean, vstd), markersize=10)
        #plt.title("data: %s, thar: %s"%(objfile.split("/")[-1], wavfile.split("/")[-1]))
        #plt.ylim(-5, 5)
        #plt.plot(telluric_orders, ccfvels[telluric_orders], '*', label='telluric', markersize=10)
        #plt.plot(telluric_orders, -vpixs[telluric_orders], '-', label='1 pixel')
        #plt.plot(telluric_orders, 0*vpixs[telluric_orders], '-', label='', color='gray')
        plt.legend()
        if savedir is not None:
            plt.savefig(savedir+"telluric_rvs.png",
                        dpi=200, bbox_inches="tight")
            plt.close()

        return rvs


"""
def clip_varr(varr, sigma_clip=3):
    idx = varr==varr
    for i in range(10):
        vmed = np.median(varr[idx])
        sigma = np.median(np.abs(varr-vmed))*1.4826
        idx_new = np.abs(varr-vmed) < sigma_clip * sigma
        if np.sum(idx) == np.sum(idx_new):
            break
        idx = idx_new
    return varr[idx]
"""
