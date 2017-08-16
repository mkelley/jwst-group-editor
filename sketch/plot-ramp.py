import itertools
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import mskpy
from mskpy import ksplot

pfx = 'appulse-1'
hdu = fits.open('{}.fits'.format(pfx))
groups = hdu[0].data
frames = hdu[2].data
tg = hdu[3].data

def diff(a):
    return a[1:] - a[:-1]

# add the reset frame
groups = np.concatenate((np.zeros((1,) + groups.shape[1:]), groups))
frames = np.concatenate((np.zeros((1,) + frames.shape[1:]), frames))
tg = np.r_[0, tg]
tf = np.arange(len(frames))
dfdt = (diff(frames).T / diff(tf)).T
dgdt = (diff(groups).T / diff(tg)).T

cyx = hdu[0].header['cy'], hdu[0].header['cx']
rap = 10
n, f = mskpy.apphot(frames, cyx, rap)
g = mskpy.apphot(groups, cyx, rap)[1]

yx = ((77, 68), (98, 65), (65, 71))


########################################################################
plt.figure(1)
plt.clf()

markers = itertools.cycle(['s', 'o', '^', 'v'])

plt.plot([0, len(frames) - 1], [0, 1], 'k-', lw=1, label='Linear ramp')

for y, x in yx:
    norm = frames[-1, y, x]
    plt.plot(tf, frames[:, y, x] / norm, marker='.', ls='none', color='k')
    plt.plot(tg, groups[:, y, x] / norm, marker=next(markers), ls='none',
             label='Pixel {}, {}'.format(y, x))

norm = f[-1]
plt.plot(tf, f / norm, marker='.', ls='none', color='k')
plt.plot(tg, g / norm, marker=next(markers), ls='none',
         label='{} pix aperture'.format(rap))

plt.setp(plt.gca(), xlabel='Frame', ylabel='Cumulative fraction')

mskpy.niceplot()
mskpy.nicelegend()
plt.draw()
plt.savefig('{}-cumulative.png'.format(pfx))

########################################################################
plt.figure(2)
plt.clf()

marker_opts = dict(marker='.', color='k', ls='none', zorder=-98)
line_opts = dict(ls='-', lw=1, zorder=-99)
for y, x in yx:
    norm = frames[-1, y, x]
    m = plt.plot(tg[1:], diff(groups[:, y, x] / norm) / diff(tg),
                 label='Pixel {}, {}'.format(y, x),
                 marker=next(markers), ls='none')
    line_opts['color'] = m[0].get_color()
    for opts in (line_opts, marker_opts):
        plt.plot(tf[1:], diff(frames[:, y, x] / norm) / diff(tf), **opts)

norm = f[-1]
m = plt.plot(tg[1:], diff(g / norm) / diff(tg), marker=next(markers),
             label='{} pix aperture'.format(rap),
             ls='none')
for opts in (line_opts, marker_opts):
    line_opts['color'] = m[0].get_color()
    plt.plot(tf[1:], diff(f / norm) / diff(tf), **opts)

plt.setp(plt.gca(), xlabel='Frame', ylabel=r'd$f$/d$t$ (s$^{-1}$)')

mskpy.niceplot()
mskpy.nicelegend(loc='upper left')
plt.draw()
plt.savefig('{}-dfdt.png'.format(pfx))


########################################################################
fig = plt.figure(3, figsize=(2, 4))
fig.clear()
tax, bax = [fig.add_subplot(gs) for gs in
            plt.GridSpec(2, 1, wspace=0, hspace=0, bottom=0, left=0, top=1,
                         right=1)]

clipped = sigma_clip(dgdt, sigma_lower=5, sigma_upper=1.5, axis=0)
cleaned = np.ma.sum(clipped, 0).data / np.sum(~clipped.mask, 0)

im = np.log(groups[-1])
#im[~np.isfinite(im)] = im[np.isfinite()].min()
tax.imshow(im)

im = np.log(cleaned)
#im[~np.isfinite(im)] = im[np.isfinite()].min()
bax.imshow(im)

mskpy.remaxes((tax, bax))
plt.draw()
plt.savefig('{}-cleaned.png'.format(pfx))
