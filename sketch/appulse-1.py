import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve
import mskpy

class config:
    filt = 'F430M'
    subframe = 'FULL'
    readpat = 'SHALLOW2'
    exptime_request = 300 * u.s
    mu = 5. * u.mas / u.s
    pa = 10 * u.deg
    impact = 0.1 * u.arcsec

# simulate JWST stellar appulse of an asteroid
nircam_readouts = {
    # nsamples, nframes
    'RAPID': (1, 1),
    'BRIGHT1': (2, 1),
    'BRIGHT2': (2, 2),
    'SHALLOW2': (5, 2),
    'SHALLOW4': (5, 4),
    'MEDIUM2': (10, 2),
    'MEDIUM8': (10, 8),
    'DEEP2': (20, 2),
    'DEEP8': (20, 8),
}
nircam_tframes = {
    'FULL': 10.73676 * u.s,
}

def dist_to_line(yx1, yx0, pa):
    QP = np.array(yx1) - np.array(yx0)

    # parallel:
    n = np.r_[np.cos(pa).value, -np.sin(pa).value]
    para = np.dot(QP, n) # / np.sqrt(np.dot(n, n))

    # perpendicular:
    n = np.r_[np.cos(pa + 90 * u.deg).value, -np.sin(pa + 90 * u.deg).value]
    perp = np.abs(np.dot(QP, n)) # / np.sqrt(np.dot(n, n))

    return para, perp

def trail(t, exptime, shape, cyx, b, ps, mu, pa):
    """
    t : time offset, the line intercepts cyx + (0, b) at t = 0
    exptime : exposure time
    shape : shape of image array
    cyx : the trail intercept point
    b : impact parameter
    ps : pixel scale of image array
    mu : proper motion magnitude of target
    pa : proper motion position angle of target, E of N

    """
    
    import scipy.ndimage as nd

    K = np.zeros(shape)
    r = (mu / ps).decompose().value
    
    n = r * exptime.to('s').value  # number of pixels to trail
    offset = r * t.to('s').value

    yx0 = (cyx[0], cyx[1] + (b / ps).decompose().value)
    yx = np.rollaxis(np.rollaxis(np.indices(shape), 2), 2)
    para, perp = dist_to_line(yx, yx0, pa)
    i = perp <= 1
    K[i] = 1 - perp[i]
    i = (para < offset) + (para > offset + n)
    K[i] = 0

    # odd dimensions:
    K = K[:(K.shape[0] % 2 - 1), :(K.shape[1] % 2 - 1)]
    
    K = K / K.sum() * exptime.to('s').value
    return K[::-1, ::-1]  # flip for convolution

def group_read(ramp, readpat, noise=False):
    t0 = np.arange(len(ramp)) + 1
    t1 = []
    groups = []
    if noise:
        noise = np.random.poisson(ramp, ramp.shape)
    else:
        noise = 0

    readout = ramp + noise

    # first frame is always saved
    t1.append(t0[0])
    groups.append(readout[0])
    
    for i in range(0, len(ramp), readpat[0]):
        t1.append(t0[i:i+readpat[1]].mean())
        groups.append(readout[i:i+readpat[1]].mean(0))

    return np.array(t1), np.array(groups)

def calc_psf(filt, source=None):
    # NIRCam default source is 5700 K star
    import webbpsf
    webbpsf.setup_logging()
    nc = webbpsf.NIRCam()
    nc.filter = filt
    psf = nc.calc_psf(source=source, nlambda=5, fov_arcsec=2)
    return psf

fn = 'star-{}.fits'.format(config.filt)
if not os.path.exists(fn):
    star = calc_psf(config.filt)
    star.writeto(fn, overwrite=True)
else:
    star = fits.open(fn)

fn = 'ast-{}.fits'.format(config.filt)
if not os.path.exists(fn):
    import pysynphot as S
    sp = S.BlackBody(170)
    ast = calc_psf(config.filt, source=sp)
    ast.writeto(fn, overwrite=True)
else:
    ast = fits.open(fn)

tframe = nircam_tframes[config.subframe]
readpat = nircam_readouts[config.readpat]
ngroups = int(np.floor(
    (config.exptime_request / tframe + readpat[0] - readpat[1]) / readpat[0]))
nframes = ngroups * readpat[0] - (readpat[0] - readpat[1])
exptime = nframes * tframe

shape = ast[0].data.shape
cyx = mskpy.gcentroid(ast[0].data, np.array(shape) / 2, box=5)
ps = ast[0].header['PIXELSCL'] * u.arcsec / u.pix
ast[0].header['CY'] = cyx[0], 'centroid'
ast[0].header['CX'] = cyx[1], 'centroid'

stack = []
for i in range(nframes):
    t = (i - nframes / 2) * tframe
    K = trail(t, tframe, shape, cyx, config.impact, ps, config.mu, config.pa)
    star_trail = convolve(star[0].data, K)
    im = star_trail + ast[0].data * tframe.to('s').value
    stack.append(im)

ramp = np.cumsum(stack, 0)

t0 = np.array(len(ramp))
t1, groups = group_read(ramp, readpat)
ast[0].data = groups
ast[1].data = np.array([mskpy.rebin(g, -4) for g in groups])
ast.append(fits.ImageHDU(ramp, name='FRAMES'))
ast.append(fits.ImageHDU(t1, name='GRPTIME'))

for i in range(2):
    ast[i].header.add_history('Asteroid with star trail')
    ast[i].header['SUBFRAME'] = config.subframe
    ast[i].header['READPAT'] = config.readpat
    ast[i].header['NGROUPS'] = ngroups
    ast[i].header['NFRAMES'] = nframes
    ast[i].header['EXPTIME'] = (tframe * nframes).value, tframe.unit
    ast[i].header['MU'] = config.mu.value, config.mu.unit
    ast[i].header['PA'] = config.pa.value, config.pa.unit
    ast[i].header['IMPACT'] = config.impact.value, config.impact.unit

ast.writeto('appulse-1.fits', overwrite=True)
