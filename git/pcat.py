import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
#import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
#import astropy.wcs
#import astropy.io.fits
import sys
import os
import warnings
import overlap_integrals

from image_eval import psf_poly_fit, image_model_eval
from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

# ix, iy = 0. to 3.999
def testpsf(nc, cf, psf, ix, iy, lib=None):
    psf0 = image_model_eval(np.array([12.-ix/5.], dtype=np.float32), np.array([12.-iy/5.], dtype=np.float32), np.array([1.], dtype=np.float32), 0., (25,25), nc, cf, lib=lib)
    plt.subplot(2,2,1)
    plt.imshow(psf0, interpolation='none', origin='lower')
    plt.title('matrix multiply PSF')
    plt.subplot(2,2,2)
    iix = int(np.floor(ix))
    iiy = int(np.floor(iy))
    dix = ix - iix
    diy = iy - iiy
    f00 = psf[iiy:125:5,  iix:125:5]
    f01 = psf[iiy+1:125:5,iix:125:5]
    f10 = psf[iiy:125:5,  iix+1:125:5]
    f11 = psf[iiy+1:125:5,iix+1:125:5]
    realpsf = f00*(1.-dix)*(1.-diy) + f10*dix*(1.-diy) + f01*(1.-dix)*diy + f11*dix*diy
    plt.imshow(realpsf, interpolation='none', origin='lower')
    plt.title('bilinear interpolate PSF')
    invrealpsf = np.zeros((25,25))
    mask = realpsf > 1e-3
    invrealpsf[mask] = 1./realpsf[mask]
    plt.subplot(2,2,3)
    plt.title('absolute difference')
    plt.imshow(psf0-realpsf, interpolation='none', origin='lower')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow((psf0-realpsf)*invrealpsf, interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('fractional difference')
    plt.show()

def neighbours(x,y,neigh,i,generate=False):
    neighx = np.abs(x - x[i])
    neighy = np.abs(y - y[i])
    adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
    oldadj = adjacency.copy()
    adjacency[i] = 0.
    neighbours = np.sum(adjacency)
    if generate:
        if neighbours:
            j = np.random.choice(adjacency.size, p=adjacency.flatten()/float(neighbours))
        else:
            j = -1
        return neighbours, j
    else:
        return neighbours


def get_region(x, offsetx, regsize):
    return np.floor(x + offsetx).astype(np.int) / regsize

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero(np.logical_and(match_x, match_y))

# script arguments
dataname = sys.argv[1]
visual = int(sys.argv[2]) > 0
# 1 to test, 0 not to test
testpsfn = int(sys.argv[3]) > 0
# 'star' for star only, 'stargalx' for star and galaxy
strgmode = sys.argv[4]
# 'mock' for simulated
datatype = sys.argv[5]


f = open('Data/'+dataname+'_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/'+dataname+'_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
    warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)

array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
libmmult = npct.load_library('pcat-lion', '.')
libmmult.pcat_model_eval.restype = None
libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
libmmult.pcat_imag_acpt.restype = None
libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
#
libmmult.pcat_like_eval.restype = None
libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

if visual and testpsfn:
    testpsf(nc, cf, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

f = open('Data/'+dataname+'_pix.txt')
w, h, nband = [np.int32(i) for i in f.readline().split()]
imsz = (w, h)
print imsz
assert nband == 1
bias, gain = [np.float32(i) for i in f.readline().split()]
f.close()
data = np.loadtxt('Data/'+dataname+'_cts.txt').astype(np.float32)
data -= bias
trueback = np.float32(445*250)#179.)
variance = data / gain
weight = 1. / variance # inverse variance

print 'Lion mode:', strgmode
print 'datatype:', datatype

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'_tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2]
    if strgmode == 'galx':
        truth_s = np.loadtxt('Data/'+dataname+'_str.txt')
        truex = truth_s[:,0]
        truey = truth_s[:,1]
        truef = truth_s[:,2]
        truth_g = np.loadtxt('Data/'+dataname+'_gal.txt')
        truexg = truth_g[:,0]
        trueyg = truth_g[:,1]
        truefg = truth_g[:,2]
        truexxg= truth_g[:,3]
        truexyg= truth_g[:,4]
        trueyyg= truth_g[:,5]
        truerng, theta, phi = from_moments(truexxg, truexyg, trueyyg)
    if strgmode == 'stargalx':
        pathliondata = os.environ["LION_DATA_PATH"] + '/data/'
        truth = np.loadtxt(pathliondata + 'truecnts.txt')
        filetrue = h5py.File(pathliondata + 'true.h5', 'r')
        dictglob = dict()
        for attr in filetrue:
            dictglob[attr] = filetrue[attr][()]
        filetrue.close()
    
    labldata = 'True'
else:
    labldata = 'HST 606W'

# number of stars to use in fit
nstar = 100
trueminf = np.float32(250.*136)
truealpha = np.float32(2.00)
back = trueback
ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)

n_all = []
x_all = []
y_all = []
f_all = []
for i in xrange(ntemps):
    n = np.random.randint(nstar)+1
    x = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
    y = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
    f = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nstar)).astype(np.float32)
    x[n:] = 0.
    y[n:] = 0.
    f[n:] = 0.
    n_all.append(n)
    x_all.append(x)
    y_all.append(y)
    f_all.append(f)

def moments_from_prior(truermin_g, ngalx, slope=np.float32(4)):
    rg = truermin_g*np.exp(np.random.exponential(scale=1./(slope-1.),size=ngalx)).astype(np.float32)
    ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
    thetag = np.arccos(ug).astype(np.float32)
    phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
    return to_moments(rg, thetag, phig)


def log_prior_moments(xx, xy, yy):
    slope = 4.
    a, theta, phi = from_moments(xx, xy, yy)
    u = np.cos(theta)
    return np.log(slope-1) + (slope-1)*np.log(truermin_g) - slope*np.log(a) - 5*np.log(a) - np.log(u*u) - np.log(1-u*u)

ngalx = 100
trueminf_g = np.float32(250.*136)
truealpha_g = np.float32(2.00)
ng_all = []
xg_all = []
yg_all = []
fg_all = []
xxg_all= []
xyg_all= []
yyg_all= []

for i in xrange(ntemps):
    ng = np.random.randint(ngalx)+1
    xg = (np.random.uniform(size=ngalx)*(imsz[0]-1)).astype(np.float32)
    yg = (np.random.uniform(size=ngalx)*(imsz[1]-1)).astype(np.float32)
    fg = trueminf_g * np.exp(np.random.exponential(scale=1./(truealpha_g-1.),size=ngalx)).astype(np.float32)
    truermin_g = np.float32(1.00)
    xxg, xyg, yyg = moments_from_prior(truermin_g, ngalx)
    xg[ng:] = 0
    yg[ng:] = 0
    fg[ng:] = 0
    xxg[ng:]= 0
    xyg[ng:]= 0
    yyg[ng:]= 0
    ng_all.append(ng)
    xg_all.append(xg)
    yg_all.append(yg)
    fg_all.append(fg)
    xxg_all.append(xxg)
    xyg_all.append(xyg)
    yyg_all.append(yyg)

gridphon, amplphon = retr_sers(sersindx=2.)

nsamp = 1000
nloop = 1000
temp_init = 100
cooling_rate = 2.*temp_init/float(nsamp) # for simulated annealing
nsample = np.zeros(nsamp, dtype=np.int32)
chi2sample = np.zeros(nsamp,dtype=float)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
fsample = np.zeros((nsamp, nstar), dtype=np.float32)
ngsample = np.zeros(nsamp, dtype=np.int32)
xgsample = np.zeros((nsamp, nstar), dtype=np.float32)
ygsample = np.zeros((nsamp, nstar), dtype=np.float32)
fgsample = np.zeros((nsamp, nstar), dtype=np.float32)
xxgsample = np.zeros((nsamp, nstar), dtype=np.float32)
xygsample = np.zeros((nsamp, nstar), dtype=np.float32)
yygsample = np.zeros((nsamp, nstar), dtype=np.float32)


penalty = 1.5
penalty_g = 3.0
regsize = 20
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10
kickrange = 1.
kickrange_g = 1.

def run_sampler(x, y, f, n, xg, yg, fg, xxg, xyg, yyg, ng, temperature, nloop=1000, visual=False):
    t0 = time.clock()
    nmov = np.zeros(nloop)
    movetype = np.zeros(nloop)
    accept = np.zeros(nloop)
    outbounds = np.zeros(nloop)
    dt1 = np.zeros(nloop)
    dt2 = np.zeros(nloop)
    dt3 = np.zeros(nloop)

    offsetx = np.random.randint(regsize)
    offsety = np.random.randint(regsize)
    nregx = imsz[0] / regsize + 1
    nregy = imsz[1] / regsize + 1

    resid = data.copy() # residual for zero image is data
    if strgmode == 'star':
        evalx = x[0:n]
        evaly = y[0:n]
        evalf = f[0:n]
    else:
        xposphon, yposphon, specphon = retr_tranphon(gridphon, amplphon, xg[0:ng], yg[0:ng], fg[0:ng], xxg[0:ng], xyg[0:ng], yyg[0:ng])
        evalx = np.concatenate([x[0:n], xposphon]).astype(np.float32)
        evaly = np.concatenate([y[0:n], yposphon]).astype(np.float32)
        evalf = np.concatenate([f[0:n], specphon]).astype(np.float32)
    n_phon = evalx.size
    model, diff2 = image_model_eval_flux(evalx, evaly, back, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, \
        regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
    logL = -0.5*diff2
    resid -= model

    moveweights = np.array([80., 0., 40., 40.])
    if strgmode == 'galx':
        moveweights = np.array([80., 0., 40., 40., 80., 40., 40., 40., 40., 40.])
    moveweights /= np.sum(moveweights)

    for i in xrange(nloop):
        t1 = time.clock()
        rtype = np.random.choice(moveweights.size, p=moveweights)
        movetype[i] = rtype
        # defaults
        nw = 0
        dback = np.float32(0.)
        pn = n
        factor = None # best way to incorporate acceptance ratio factors?
        goodmove = False

	# should regions be perturbed randomly or systematically?
	parity_x = np.random.randint(2)
	parity_y = np.random.randint(2)

	idx_move = None
        idx_move_g = None
	do_birth = False
        do_birth_g = False
	idx_kill = None
        idx_kill_g = None
        # mover
        if rtype == 0:
            idx_move = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            nw = idx_move.size
            f0 = f.take(idx_move)

            lindf = np.float32(60.*134/np.sqrt(25.))
            logdf = np.float32(0.01/np.sqrt(25.))
            ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
            ffmin = np.log(logdf*logdf*trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf*trueminf)) / logdf
            dff = np.random.normal(size=nw).astype(np.float32)
            aboveffmin = ff - ffmin
            oob_flux = (-dff > aboveffmin)
            dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
            pff = ff + dff
            pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
            # calculate flux distribution prior factor
            dlogf = np.log(pf/f0)
            factor = -truealpha*dlogf

            dpos_rms = np.float32(60.*134/np.sqrt(25.))/(np.maximum(f0, pf))
            dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            x0 = x.take(idx_move)
            y0 = y.take(idx_move)
            px = x0 + dx
            py = y0 + dy
            # bounce off of edges of image
            mask = px < 0
            px[mask] *= -1
            mask = px > (imsz[0] - 1)
            px[mask] *= -1
            px[mask] += 2*(imsz[0] - 1)
            mask = py < 0
            py[mask] *= -1
            mask = py > (imsz[1] - 1)
            py[mask] *= -1
            py[mask] += 2*(imsz[1] - 1)

            goodmove = True # always True because we bounce off the edges of the image and fmin
        # background change
        elif rtype == 1:
            dback = np.float32(np.random.normal())
            goodmove = True 
        # birth and death
        elif rtype == 2:
            lifeordeath = np.random.randint(2)
            nbd = (nregx * nregy) / 4
            # birth
            if lifeordeath and n < nstar: # need room for at least one source
                nbd = min(nbd, nstar-n) # add nbd sources, or just as many as will fit
                                        # mildly violates detailed balance when n close to nstar
                # want number of regions in each direction, divided by two, rounded up
                mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
                mregy = ((imsz[1] / regsize + 1) + 1) / 2
                bx = ((np.random.randint(mregx, size=nbd)*2 + parity_x + np.random.uniform(size=nbd))*regsize - offsetx).astype(np.float32)
                by = ((np.random.randint(mregy, size=nbd)*2 + parity_y + np.random.uniform(size=nbd))*regsize - offsety).astype(np.float32)
                bf = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nbd)).astype(np.float32)

		# some sources might be generated outside image
		inbounds = (bx > 0) * (bx < (imsz[0] -1)) * (by > 0) * (by < imsz[1] - 1)
		idx_in = np.flatnonzero(inbounds)
                nw = idx_in.size
		bx = bx.take(idx_in)
                by = by.take(idx_in)
                bf = bf.take(idx_in)
                do_birth = True
                factor = np.full(nw, -penalty)
                goodmove = True
            # death
            # does region based death obey detailed balance?
            elif not lifeordeath and n > 0: # need something to kill
                idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)

		nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
                nw = nbd
                if nbd > 0:
                    idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
		    xk = x.take(idx_kill)
                    yk = y.take(idx_kill)
                    fk = f.take(idx_kill)
                    factor = np.full(nbd, penalty)
                    goodmove = True
                else:
                    goodmove = False
        # merges and splits
        elif rtype == 3:
            splitsville = np.random.randint(2)
            idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg.take(np.flatnonzero(f.take(idx_reg) > 2*trueminf)) # in region!
            bright_n = idx_bright.size

            nms = (nregx * nregy) / 4
            # split
            if splitsville and n > 0 and n < nstar and bright_n > 0: # need something to split, but don't exceed nstar
                nms = min(nms, bright_n, nstar-n) # need bright source AND room for split source
                dx = (np.random.normal(size=nms)*kickrange).astype(np.float32)
                dy = (np.random.normal(size=nms)*kickrange).astype(np.float32)
		idx_move = np.random.choice(idx_bright, size=nms, replace=False)
                x0 = x.take(idx_move)
                y0 = y.take(idx_move)
		f0 = f.take(idx_move)
                fminratio = f0 / trueminf
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                px = x0 + ((1-frac)*dx)
                py = y0 + ((1-frac)*dy)
                pf = f0 * frac
                do_birth = True
                bx = x0 - frac*dx
                by = y0 - frac*dy
                bf = f0 * (1-frac)

                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (px > 0) * (px < imsz[0] - 1) * (py > 0) * (py < imsz[1] - 1) * \
                           (bx > 0) * (bx < imsz[0] - 1) * (by > 0) * (by < imsz[1] - 1)
                idx_in = np.flatnonzero(inbounds)
                x0 = x0.take(idx_in)
                y0 = y0.take(idx_in)
                f0 = f0.take(idx_in)
                px = px.take(idx_in)
                py = py.take(idx_in)
                pf = pf.take(idx_in)
                bx = bx.take(idx_in)
                by = by.take(idx_in)
                bf = bf.take(idx_in)
                idx_move = idx_move.take(idx_in)
                fminratio = fminratio.take(idx_in)
                frac = frac.take(idx_in)
                goodmove = idx_in.size > 0

                # need to calculate factor
                sum_f = f0
                nms = idx_in.size
                nw = nms
                invpairs = np.zeros(nms)
                for k in xrange(nms):
                    xtemp = x[0:n].copy()
                    ytemp = y[0:n].copy()
                    xtemp[idx_move[k]] = px[k]
                    ytemp[idx_move[k]] = py[k]
                    xtemp = np.concatenate([xtemp, bx[k:k+1]])
                    ytemp = np.concatenate([ytemp, by[k:k+1]])

                    invpairs[k] =  1./neighbours(xtemp, ytemp, kickrange, idx_move[k])
                    invpairs[k] += 1./neighbours(xtemp, ytemp, kickrange, n)
                invpairs *= 0.5
            # merge
            elif not splitsville and idx_reg.size > 1: # need two things to merge!
                nms = min(nms, idx_reg.size/2)
                idx_move = np.zeros(nms, dtype=np.int)
                idx_kill = np.zeros(nms, dtype=np.int)
                choosable = np.zeros(nstar, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(np.count_nonzero(choosable))
                invpairs = np.zeros(nms)

                for k in xrange(nms):
                    idx_move[k] = np.random.choice(nstar, p=choosable/nchoosable)
                    invpairs[k], idx_kill[k] = neighbours(x[0:n], y[0:n], kickrange, idx_move[k], generate=True)
                    if invpairs[k] > 0:
                        invpairs[k] = 1./invpairs[k]
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k]]:
                        idx_kill[k] = -1
                    if idx_kill[k] != -1:
                        invpairs[k] += 1./neighbours(x[0:n], y[0:n], kickrange, idx_kill[k])
                        choosable[idx_move[k]] = False
                        choosable[idx_kill[k]] = False
                        nchoosable -= 2
                invpairs *= 0.5

                inbounds = (idx_kill != -1)
                idx_in = np.flatnonzero(inbounds)
                nms = idx_in.size
                nw = nms
                idx_move = idx_move.take(idx_in)
                idx_kill = idx_kill.take(idx_in)
                invpairs = invpairs.take(idx_in)
                goodmove = idx_in.size > 0

                x0 = x.take(idx_move)
                y0 = y.take(idx_move)
                f0 = f.take(idx_move)
                xk = x.take(idx_kill)
                yk = y.take(idx_kill)
                fk = f.take(idx_kill)
                sum_f = f0 + fk
                fminratio = sum_f / trueminf
                frac = f0 / sum_f
                px = frac*x0 + (1-frac)*xk
                py = frac*y0 + (1-frac)*yk
                pf = f0 + fk
                # turn bright_n into an array
                bright_n = bright_n - (f0 > 2*trueminf) - (fk > 2*trueminf) + (pf > 2*trueminf)
            if goodmove:
                factor = np.log(truealpha-1) + (truealpha-1)*np.log(trueminf) - truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*kickrange*kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
                if not splitsville:
                    factor *= -1
                    factor += penalty
                else:
                    factor -= penalty
        # move galaxies
        elif rtype == 4:
            idx_move_g = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize)
            nw = idx_move_g.size
            f0g = fg.take(idx_move_g)

            lindf = np.float32(60.*134/np.sqrt(25.))
            logdf = np.float32(0.01/np.sqrt(25.))
            ff = np.log(logdf*logdf*f0g + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0g*f0g)) / logdf
            ffmin = np.log(logdf*logdf*trueminf_g + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf_g*trueminf_g)) / logdf
            dff = np.random.normal(size=nw).astype(np.float32)
            aboveffmin = ff - ffmin
            oob_flux = (-dff > aboveffmin)
            dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
            pff = ff + dff
            pfg = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)

            dlogfg = np.log(pfg/f0g)
            factor = -truealpha_g*dlogfg

            dpos_rms = np.float32(60.*134/np.sqrt(25.))/(np.maximum(f0g, pfg))
            dxg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            dyg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            dxxg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
            dxyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
            dyyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
            x0g = xg.take(idx_move_g)
            y0g = yg.take(idx_move_g)
            xx0g= xxg.take(idx_move_g)
            xy0g= xyg.take(idx_move_g)
            yy0g= yyg.take(idx_move_g)
            pxg = x0g + dxg
            pyg = y0g + dyg
            pxxg=xx0g + dxxg
            pxyg=xy0g + dxyg
            pyyg=yy0g + dyyg
            # bounce xx, yy off of zero TODO min radius of galaxies?
            mask = pxxg < 0
            pxxg[mask] *= -1
            mask = pyyg < 0
            pyyg[mask] *= -1
            # put in -inf for illegal ellipses
            mask = (pxxg*pyyg - pxyg*pxyg) <= 0
            pxxg[mask] = xx0g[mask] # make illegal ellipses legal so that prior factor can be calculated, even though not used
            pxyg[mask] = xy0g[mask]
            pyyg[mask] = yy0g[mask]
            factor[mask] = -float('inf')

            # calculate prior factor 
            factor += -log_prior_moments(xx0g, xy0g, yy0g) + log_prior_moments(pxxg, pxyg, pyyg)
            
            # bounce off of edges of image
            mask = pxg < 0
            pxg[mask] *= -1
            mask = pxg > (imsz[0] - 1)
            pxg[mask] *= -1
            pxg[mask] += 2*(imsz[0] - 1)
            mask = pyg < 0
            pyg[mask] *= -1
            mask = pyg > (imsz[1] - 1)
            pyg[mask] *= -1
            pyg[mask] += 2*(imsz[1] - 1)
            goodmove = True
        # galaxy birth and death
        elif rtype == 5:
            lifeordeath = np.random.randint(2)
            nbd = (nregx * nregy) / 4
            # birth
            if lifeordeath and ng < ngalx: # need room for at least one source
                nbd = min(nbd, ngalx-ng) # add nbd sources, or just as many as will fit
                                        # mildly violates detailed balance when n close to nstar
                # want number of regions in each direction, divided by two, rounded up
                mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
                mregy = ((imsz[1] / regsize + 1) + 1) / 2
                bxg = ((np.random.randint(mregx, size=nbd)*2 + parity_x + np.random.uniform(size=nbd))*regsize - offsetx).astype(np.float32)
                byg = ((np.random.randint(mregy, size=nbd)*2 + parity_y + np.random.uniform(size=nbd))*regsize - offsety).astype(np.float32)
                bfg = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nbd)).astype(np.float32)
                # put in function?
                bxxg, bxyg, byyg = moments_from_prior(truermin_g, nbd)

                # some sources might be generated outside image
                inbounds = (bxg > 0) * (bxg < (imsz[0] - 1)) * (byg > 0) * (byg < (imsz[1] - 1))
                idx_in = np.flatnonzero(inbounds)
                nw = idx_in.size
                bxg = bxg.take(idx_in)
                byg = byg.take(idx_in)
                bfg = bfg.take(idx_in)
                bxxg=bxxg.take(idx_in)
                bxyg=bxyg.take(idx_in)
                byyg=byyg.take(idx_in)
                do_birth_g = True
                factor = np.full(nw, -penalty_g)
                goodmove = True
            # death
            # does region based death obey detailed balance?
            elif not lifeordeath and ng > 0: # need something to kill
                idx_reg = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize)

                nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
                nw = nbd
                if nbd > 0:
                    idx_kill_g = np.random.choice(idx_reg, size=nbd, replace=False)
                    xkg = xg.take(idx_kill_g)
                    ykg = yg.take(idx_kill_g)
                    fkg = fg.take(idx_kill_g)
                    xxkg=xxg.take(idx_kill_g)
                    xykg=xyg.take(idx_kill_g)
                    yykg=yyg.take(idx_kill_g)
                    factor = np.full(nbd, penalty_g)
                    goodmove = True
                else:
                    goodmove = False
        # one star <-> one galaxy
        elif rtype == 6:
                starorgalx = np.random.randint(2)
                nsg = (nregx * nregy) / 4
                # star -> galaxy
                if starorgalx and n > 0 and ng < ngalx:
                    idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
                    nsg = min(nsg, min(idx_reg.size, ngalx-ng))
                    nw = nsg
                    if nsg > 0:
                        idx_kill = np.random.choice(idx_reg, size=nsg, replace=False)
                        xk = x.take(idx_kill)
                        yk = y.take(idx_kill)
                        fk = f.take(idx_kill)
                        do_birth_g = True
                        bxg = xk.copy()
                        byg = yk.copy()
                        bfg = fk.copy()

                        bxxg, bxyg, byyg = moments_from_prior(truermin_g, nsg)
                        factor = np.full(nsg, penalty-penalty_g) # TODO factor if star and galaxy flux distributions different
                        goodmove = True
                    else:
                        goodmove = False
                # galaxy -> star
                elif not starorgalx and ng > 1 and n < nstar:
                    idx_reg = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize)
                    nsg = min(nsg, min(idx_reg.size, nstar-n))
                    nw = nsg
                    if nsg > 0:
                        idx_kill_g = np.random.choice(idx_reg, size=nsg, replace=False)
                        xkg = xg.take(idx_kill_g)
                        ykg = yg.take(idx_kill_g)
                        fkg = fg.take(idx_kill_g)
                        xxkg=xxg.take(idx_kill_g)
                        xykg=xyg.take(idx_kill_g)
                        yykg=yyg.take(idx_kill_g)
                        do_birth = True
                        bx = xkg.copy()
                        by = ykg.copy()
                        bf = fkg.copy()
                        factor = np.full(nsg, penalty_g-penalty) # TODO factor if star and galaxy flux distributions different
                        goodmove = True
                    else:
                        goodmove = False
        # one galaxy <-> two stars
        elif rtype == 7:
            splitsville = np.random.randint(2)
            idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize) # stars
            idx_reg_g = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize) # galaxies
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg_g.take(np.flatnonzero(fg.take(idx_reg_g) > 2*trueminf)) # in region and bright enough to make two stars
            bright_n = idx_bright.size # can only split bright galaxies

            nms = (nregx * nregy) / 4
            # split
            if splitsville and ng > 0 and n < nstar-2 and bright_n > 0: # need something to split, but don't exceed nstar
                nms = min(nms, bright_n, (nstar-n)/2) # need bright galaxy AND room for split stars
                idx_kill_g = np.random.choice(idx_bright, size=nms, replace=False)
                xkg = xg.take(idx_kill_g)
                ykg = yg.take(idx_kill_g)
                fkg = fg.take(idx_kill_g)
                xxkg=xxg.take(idx_kill_g)
                xykg=xyg.take(idx_kill_g)
                yykg=yyg.take(idx_kill_g)
                fminratio = fkg / trueminf # again, care about fmin for stars
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                f1Mf = frac * (1. - frac) # frac(1 - frac)
                agalx, theta, phi = from_moments(xxkg, xykg, yykg)
                dx = agalx * np.cos(phi) / np.sqrt(2 * f1Mf)
                dy = agalx * np.sin(phi) / np.sqrt(2 * f1Mf)
                dr2 = dx*dx + dy*dy
                do_birth = True
                bx = np.column_stack([xkg + ((1-frac)*dx), xkg - frac*dx])
                by = np.column_stack([ykg + ((1-frac)*dy), ykg - frac*dy])
                bf = np.column_stack([fkg * frac, fkg * (1-frac)])

                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (bx[:,0] > 0) * (bx[:,0] < imsz[0] - 1) * (by[:,0] > 0) * (by[:,0] < imsz[1] - 1) * \
                           (bx[:,1] > 0) * (bx[:,1] < imsz[0] - 1) * (by[:,1] > 0) * (by[:,1] < imsz[1] - 1)
                idx_in = np.flatnonzero(inbounds)
                xkg = xkg.take(idx_in)
                ykg = ykg.take(idx_in)
                fkg = fkg.take(idx_in)
                xxkg=xxkg.take(idx_in)
                xykg=xykg.take(idx_in)
                yykg=yykg.take(idx_in)
                bx = bx.take(idx_in, axis=0) # birth arrays are 2D
                by = by.take(idx_in, axis=0)
                bf = bf.take(idx_in, axis=0)
                idx_kill_g = idx_kill_g.take(idx_in)
                fminratio = fminratio.take(idx_in)
                frac = frac.take(idx_in)
                dr2 = dr2.take(idx_in)
                f1Mf = f1Mf.take(idx_in)
                theta = theta.take(idx_in)
                goodmove = idx_in.size > 0

                # need star pairs to calculate factor
                sum_f = fkg
                nms = idx_in.size
                nw = nms
                weightoverpairs = np.zeros(nms) # w (1/sum w_1 + 1/sum w_2) / 2
                for k in xrange(nms):
                    xtemp = x[0:n].copy()
                    ytemp = y[0:n].copy()
                    xtemp = np.concatenate([xtemp, bx[k:k+1,0], bx[k:k+1,1]])
                    ytemp = np.concatenate([ytemp, by[k:k+1,0], by[k:k+1,1]])

                    neighi = neighbours(xtemp, ytemp, kickrange_g, n)
                    neighj = neighbours(xtemp, ytemp, kickrange_g, n+1)
                    if neighi > 0 and neighj > 0:
                        weightoverpairs[k] = 1./neighi + 1./neighj
                    # else keep zero
                weightoverpairs *= 0.5 * np.exp(-dr2/(2.*kickrange_g*kickrange_g))
                weightoverpairs[weightoverpairs == 0] = 1
            # merge
            elif not splitsville and idx_reg.size > 1: # need two things to merge!
                nms = min(nms, idx_reg.size/2, ngalx-ng)
                idx_kill = np.zeros((nms, 2), dtype=np.int)
                choosable = np.zeros(nstar, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(np.count_nonzero(choosable))
                invpairs = np.zeros(nms)
                weightoverpairs = np.zeros(nms)

                for k in xrange(nms):
                    idx_kill[k,0] = np.random.choice(nstar, p=choosable/nchoosable)
                    invpairs[k], idx_kill[k,1] = neighbours(x[0:n], y[0:n], kickrange_g, idx_kill[k,0], generate=True)
                    invpairs[k] = 1./invpairs[k]
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k,1]]:
                        idx_kill[k,1] = -1
                    if idx_kill[k,1] != -1:
                        invpairs[k] += 1./neighbours(x[0:n], y[0:n], kickrange_g, idx_kill[k,1])
                        choosable[idx_kill[k,:]] = False
                        nchoosable -= 2
                invpairs *= 0.5

                inbounds = (idx_kill[:,1] != -1)
                idx_in = np.flatnonzero(inbounds)
                nms = idx_in.size
                nw = nms
                idx_kill = idx_kill.take(idx_in, axis=0)
                invpairs = invpairs.take(idx_in)
                goodmove = idx_in.size > 0

                xk = x.take(idx_kill) # because idx_kill is 2D so are these
                yk = y.take(idx_kill)
                fk = f.take(idx_kill)
                dx = xk[:,1] - xk[:,0]
                dy = yk[:,1] - yk[:,0]
                dr2 = dx*dx + dy*dy
                weightoverpairs = np.exp(-dr2/(2.*kickrange_g*kickrange_g)) * invpairs
                weightoverpairs[weightoverpairs == 0] = 1
                sum_f = np.sum(fk, axis=1)
                fminratio = sum_f / trueminf
                frac = fk[:,0] / sum_f
                f1Mf = frac * (1. - frac)
                do_birth_g = True
                bxg = frac*xk[:,0] + (1-frac)*xk[:,1]
                byg = frac*yk[:,0] + (1-frac)*yk[:,1]
                bfg = sum_f
                u = np.random.uniform(low=3e-4, high=1., size=idx_in.size).astype(np.float32) #3e-4 for numerics
                theta = np.arccos(u).astype(np.float32)
                bxxg = f1Mf*(dx*dx+u*u*dy*dy)
                bxyg = f1Mf*(1-u*u)*dx*dy
                byyg = f1Mf*(dy*dy+u*u*dx*dx)
                # this move proposes a splittable galaxy
                bright_n += 1 
            if goodmove:
                factor = 2*np.log(truealpha-1) - np.log(truealpha_g-1) + 2*(truealpha-1)*np.log(trueminf) - (truealpha_g-1)*np.log(trueminf_g) - \
                    truealpha*np.log(f1Mf) - (2*truealpha - truealpha_g)*np.log(sum_f) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) - \
                    np.log(2*np.pi*kickrange_g*kickrange_g) + np.log(bright_n/(ng+1.-splitsville)) + np.log((n-1+2*splitsville)*weightoverpairs) + \
                    np.log(sum_f) - np.log(4.) - 2*np.log(dr2) - 3*np.log(f1Mf) - np.log(np.cos(theta)) - 3*np.log(np.sin(theta))
                if not splitsville:
                    factor *= -1
                    factor += 2*penalty - penalty_g
                    factor += log_prior_moments(bxxg, bxyg, byyg)
                else:
                    factor -= 2*penalty - penalty_g
                    factor -= log_prior_moments(xxkg, xykg, yykg)
        # galaxy <-> galaxy + star
        elif rtype == 8:
            splitsville = np.random.randint(2)
            idx_reg_g = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize)
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg_g.take(np.flatnonzero(fg.take(idx_reg_g) > trueminf + trueminf_g)) # in region and bright enough to make s+g
            bright_n = idx_bright.size

            nms = (nregx * nregy) / 4
            # split off star
            if splitsville and ng > 0 and n < nstar and bright_n > 0: # need something to split, but don't exceed nstar
                nms = min(nms, bright_n, nstar-n) # need bright source AND room for split off star
                dx = (np.random.normal(size=nms)*kickrange_g).astype(np.float32)
                dy = (np.random.normal(size=nms)*kickrange_g).astype(np.float32)
                idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
                x0g = xg.take(idx_move_g)
                y0g = yg.take(idx_move_g)
                f0g = fg.take(idx_move_g)
                xx0g = xxg.take(idx_move_g)
                xy0g = xyg.take(idx_move_g)
                yy0g = yyg.take(idx_move_g)
                frac = (trueminf_g/f0g + np.random.uniform(size=nms)*(1. - (trueminf_g + trueminf)/f0g)).astype(np.float32)
                pxg = x0g + ((1-frac)*dx)
                pyg = y0g + ((1-frac)*dy)
                pfg = f0g * frac
                pxxg = (xx0g - frac*(1-frac)*dx*dx)/frac
                pxyg = (xy0g - frac*(1-frac)*dx*dy)/frac
                pyyg = (yy0g - frac*(1-frac)*dy*dy)/frac
                do_birth = True
                bx = x0g - frac*dx
                by = y0g - frac*dy
                bf = f0g * (1-frac)

                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (pxg > 0) * (pxg < imsz[0] - 1) * (pyg > 0) * (pyg < imsz[1] - 1) * \
                           (bx > 0) * (bx < imsz[0] - 1) * (by > 0) * (by < imsz[1] - 1) * \
                           (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # TODO minimum galaxy radius?
                idx_in = np.flatnonzero(inbounds)
                x0g = x0g.take(idx_in)
                y0g = y0g.take(idx_in)
                f0g = f0g.take(idx_in)
                xx0g = xx0g.take(idx_in)
                xy0g = xy0g.take(idx_in)
                yy0g = yy0g.take(idx_in)
                pxg = pxg.take(idx_in)
                pyg = pyg.take(idx_in)
                pfg = pfg.take(idx_in)
                pxxg = pxxg.take(idx_in)
                pxyg = pxyg.take(idx_in)
                pyyg = pyyg.take(idx_in)
                bx = bx.take(idx_in)
                by = by.take(idx_in)
                bf = bf.take(idx_in)
                idx_move_g = idx_move_g.take(idx_in)
                frac = frac.take(idx_in)
                goodmove = idx_in.size > 0

                # need to calculate factor
                sum_f = f0g
                nms = idx_in.size
                nw = nms
                invpairs = np.zeros(nms)
                for k in xrange(nms):
                    xtemp = x[0:n].copy()
                    ytemp = y[0:n].copy()
                    xtemp = np.concatenate([xtemp, pxg[k:k+1], bx[k:k+1]])
                    ytemp = np.concatenate([ytemp, pyg[k:k+1], by[k:k+1]])

                    invpairs[k] =  1./neighbours(xtemp, ytemp, kickrange_g, n)
            # merge star into galaxy
            elif not splitsville and idx_reg_g.size > 1: # need two things to merge!
                nms = min(nms, idx_reg_g.size)
                idx_move_g = np.random.choice(idx_reg_g, size=nms, replace=False) # choose galaxies and then see if they have neighbours
                idx_kill = np.zeros(nms, dtype=np.int)
                choosable = np.full(nstar, True, dtype=np.bool)
                nchoosable = float(np.count_nonzero(choosable))
                invpairs = np.zeros(nms)

                for k in xrange(nms):
                    l = idx_move_g[k]
                    invpairs[k], idx_kill[k] = neighbours(np.concatenate([x[0:n], xg[l:l+1]]), np.concatenate([y[0:n], yg[l:l+1]]), kickrange_g, n, generate=True)
                    if invpairs[k] > 0:
                        invpairs[k] = 1./invpairs[k]
                    else:
                        assert idx_kill[k] == -1
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k]]:
                        idx_kill[k] = -1
                    if idx_kill[k] != -1:
                        choosable[idx_kill[k]] = False

                inbounds = (idx_kill != -1)
                idx_in = np.flatnonzero(inbounds)
                nms = idx_in.size
                nw = nms
                idx_move_g = idx_move_g.take(idx_in)
                idx_kill = idx_kill.take(idx_in)
                invpairs = invpairs.take(idx_in)
                goodmove = idx_in.size > 0

                x0g = xg.take(idx_move_g)
                y0g = yg.take(idx_move_g)
                f0g = fg.take(idx_move_g)
                xx0g=xxg.take(idx_move_g)
                xy0g=xyg.take(idx_move_g)
                yy0g=yyg.take(idx_move_g)
                xk = x.take(idx_kill)
                yk = y.take(idx_kill)
                fk = f.take(idx_kill)
                sum_f = f0g + fk
                frac = f0g / sum_f
                dx = x0g - xk
                dy = y0g - yk
                pxxg = frac*xx0g + frac*(1-frac)*dx*dx
                pxyg = frac*xy0g + frac*(1-frac)*dx*dy
                pyyg = frac*yy0g + frac*(1-frac)*dy*dy
                inbounds = (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # make sure ellipse is legal TODO min galaxy radius
                x0g = x0g.compress(inbounds)
                y0g = y0g.compress(inbounds)
                f0g = f0g.compress(inbounds)
                xx0g=xx0g.compress(inbounds)
                xy0g=xy0g.compress(inbounds)
                yy0g=yy0g.compress(inbounds)
                xk =   xk.compress(inbounds)
                yk =   yk.compress(inbounds)
                fk =   fk.compress(inbounds)
                sum_f=sum_f.compress(inbounds)
                frac=frac.compress(inbounds)
                pxxg=pxxg.compress(inbounds)
                pxyg=pxyg.compress(inbounds)
                pyyg=pyyg.compress(inbounds)
                goodmove = np.logical_and(goodmove, inbounds.any())

                pxg = frac*x0g + (1-frac)*xk
                pyg = frac*y0g + (1-frac)*yk
                pfg = f0g + fk
                # this proposal makes a galaxy that is bright enough to split
                bright_n = bright_n + 1
            if goodmove:
                factor = np.log(truealpha-1) - (truealpha-1)*np.log(sum_f/trueminf) - truealpha_g*np.log(frac) - truealpha*np.log(1-frac) + np.log(2*np.pi*kickrange_g*kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - (trueminf+trueminf_g)/sum_f) + np.log(bright_n/float(ng)) + np.log((n+1.-splitsville)*invpairs) - 2*np.log(frac)
                if not splitsville:
                    factor *= -1
                    factor += penalty
                else:
                    factor -= penalty
                factor += log_prior_moments(pxxg, pxyg, pyyg) - log_prior_moments(xx0g, xy0g, yy0g) # galaxy prior
        # galaxy merges and splits
        elif rtype == 9:
            splitsville = np.random.randint(2)
            idx_reg = idx_parity(xg, yg, ng, offsetx, offsety, parity_x, parity_y, regsize)
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg.take(np.flatnonzero(fg.take(idx_reg) > 2*trueminf_g)) # in region!
            bright_n = idx_bright.size

            nms = (nregx * nregy) / 4
            # split
            if splitsville and ng > 0 and ng < ngalx and bright_n > 0: # need something to split, but don't exceed nstar
                nms = min(nms, bright_n, ngalx-ng) # need bright source AND room for split source
                dx = (np.random.normal(size=nms)*kickrange_g).astype(np.float32)
                dy = (np.random.normal(size=nms)*kickrange_g).astype(np.float32)
                idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
                x0g = xg.take(idx_move_g)
                y0g = yg.take(idx_move_g)
                f0g = fg.take(idx_move_g)
                xx0g=xxg.take(idx_move_g)
                xy0g=xyg.take(idx_move_g)
                yy0g=yyg.take(idx_move_g)
                fminratio = f0g / trueminf_g
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                frac_xx = np.random.uniform(size=nms).astype(np.float32)
                frac_xy = np.random.uniform(size=nms).astype(np.float32)
                frac_yy = np.random.uniform(size=nms).astype(np.float32)
                xx_p = xx0g - frac*(1-frac)*dx*dx# moments of just galaxy pair
                xy_p = xy0g - frac*(1-frac)*dx*dy
                yy_p = yy0g - frac*(1-frac)*dy*dy
                pxg = x0g + ((1-frac)*dx)
                pyg = y0g + ((1-frac)*dy)
                pfg = f0g * frac
                pxxg = xx_p * frac_xx / frac
                pxyg = xy_p * frac_xy / frac
                pyyg = yy_p * frac_yy / frac
                do_birth_g = True
                bxg = x0g - frac*dx
                byg = y0g - frac*dy
                bfg = f0g * (1-frac)
                bxxg = xx_p * (1-frac_xx) / (1-frac) # FIXME is this right?
                bxyg = xy_p * (1-frac_xy) / (1-frac)
                byyg = yy_p * (1-frac_yy) / (1-frac)
                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (pxg > 0) * (pxg < imsz[0] - 1) * (pyg > 0) * (pyg < imsz[1] - 1) * \
                           (bxg > 0) * (bxg < imsz[0] - 1) * (byg > 0) * (byg < imsz[1] - 1) * \
                           (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) * \
                           (bxxg > 0) * (byyg > 0) * (bxxg*byyg > bxyg*bxyg) # TODO min galaxy rad
                idx_in = np.flatnonzero(inbounds)
                x0g = x0g.take(idx_in)
                y0g = y0g.take(idx_in)
                f0g = f0g.take(idx_in)
                xx0g=xx0g.take(idx_in)
                xy0g=xy0g.take(idx_in)
                yy0g=yy0g.take(idx_in)
                pxg = pxg.take(idx_in)
                pyg = pyg.take(idx_in)
                pfg = pfg.take(idx_in)
                pxxg=pxxg.take(idx_in)
                pxyg=pxyg.take(idx_in)
                pyyg=pyyg.take(idx_in)
                bxg = bxg.take(idx_in)
                byg = byg.take(idx_in)
                bfg = bfg.take(idx_in)
                bxxg=bxxg.take(idx_in)
                bxyg=bxyg.take(idx_in)
                byyg=byyg.take(idx_in)
                idx_move_g = idx_move_g.take(idx_in)
                fminratio = fminratio.take(idx_in)
                frac = frac.take(idx_in)
                xx_p = xx_p.take(idx_in)
                xy_p = xy_p.take(idx_in)
                yy_p = yy_p.take(idx_in)
                goodmove = idx_in.size > 0

                # need to calculate factor
                sum_f = f0g
                nms = idx_in.size
                nw = nms
                invpairs = np.zeros(nms)
                for k in xrange(nms):
                    xtemp = xg[0:ng].copy()
                    ytemp = yg[0:ng].copy()
                    xtemp[idx_move_g[k]] = pxg[k]
                    ytemp[idx_move_g[k]] = pyg[k]
                    xtemp = np.concatenate([xtemp, bxg[k:k+1]])
                    ytemp = np.concatenate([ytemp, byg[k:k+1]])

                    invpairs[k] =  1./neighbours(xtemp, ytemp, kickrange_g, idx_move_g[k])
                    invpairs[k] += 1./neighbours(xtemp, ytemp, kickrange_g, ng)
                invpairs *= 0.5
            # merge
            elif not splitsville and idx_reg.size > 1: # need two things to merge!
                nms = min(nms, idx_reg.size/2)
                idx_move_g = np.zeros(nms, dtype=np.int)
                idx_kill_g = np.zeros(nms, dtype=np.int)
                choosable = np.zeros(ngalx, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(np.count_nonzero(choosable))
                invpairs = np.zeros(nms)

                for k in xrange(nms):
                    idx_move_g[k] = np.random.choice(ngalx, p=choosable/nchoosable)
                    invpairs[k], idx_kill_g[k] = neighbours(xg[0:ng], yg[0:ng], kickrange_g, idx_move_g[k], generate=True)
                    if invpairs[k] > 0:
                        invpairs[k] = 1./invpairs[k]
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill_g[k]]:
                        idx_kill_g[k] = -1
                    if idx_kill_g[k] != -1:
                        invpairs[k] += 1./neighbours(xg[0:ng], yg[0:ng], kickrange_g, idx_kill_g[k])
                        choosable[idx_move_g[k]] = False
                        choosable[idx_kill_g[k]] = False
                        nchoosable -= 2
                invpairs *= 0.5

                inbounds = (idx_kill_g != -1)
                idx_in = np.flatnonzero(inbounds)
                nms = idx_in.size
                nw = nms
                idx_move_g = idx_move_g.take(idx_in)
                idx_kill_g = idx_kill_g.take(idx_in)
                invpairs = invpairs.take(idx_in)
                goodmove = idx_in.size > 0

                x0g = xg.take(idx_move_g)
                y0g = yg.take(idx_move_g)
                f0g = fg.take(idx_move_g)
                xx0g=xxg.take(idx_move_g)
                xy0g=xyg.take(idx_move_g)
                yy0g=yyg.take(idx_move_g)
                xkg = xg.take(idx_kill_g)
                ykg = yg.take(idx_kill_g)
                fkg = fg.take(idx_kill_g)
                xxkg=xxg.take(idx_kill_g)
                xykg=xyg.take(idx_kill_g)
                yykg=yyg.take(idx_kill_g)
                sum_f = f0g + fkg
                fminratio = sum_f / trueminf_g
                frac = f0g / sum_f

                pxg = frac*x0g + (1-frac)*xkg
                pyg = frac*y0g + (1-frac)*ykg
                pfg = f0g + fkg
                dx = x0g - xkg
                dy = y0g - ykg
                xx_p = frac*xx0g + (1-frac)*xxkg
                xy_p = frac*xy0g + (1-frac)*xykg
                yy_p = frac*yy0g + (1-frac)*yykg
                pxxg = xx_p + frac*(1-frac)*dx*dx
                pxyg = xy_p + frac*(1-frac)*dx*dy
                pyyg = yy_p + frac*(1-frac)*dy*dy

                idx_in = np.flatnonzero((pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg)) # ellipse legal TODO minimum radius
                x0g = x0g.take(idx_in)
                y0g = y0g.take(idx_in)
                f0g = f0g.take(idx_in)
                xx0g=xx0g.take(idx_in)
                xy0g=xy0g.take(idx_in)
                yy0g=yy0g.take(idx_in)
                xkg = xkg.take(idx_in)
                ykg = ykg.take(idx_in)
                fkg = fkg.take(idx_in)
                xxkg=xxkg.take(idx_in)
                xykg=xykg.take(idx_in)
                yykg=yykg.take(idx_in)
                pxg = pxg.take(idx_in)
                pyg = pyg.take(idx_in)
                pfg = pfg.take(idx_in)
                pxxg=pxxg.take(idx_in)
                pxyg=pxyg.take(idx_in)
                pyyg=pyyg.take(idx_in)
                
                idx_move_g = idx_move_g.take(idx_in)
                idx_kill_g = idx_kill_g.take(idx_in)
                invpairs = invpairs.take(idx_in)
                sum_f = sum_f.take(idx_in)
                fminratio = fminratio.take(idx_in)
                frac = frac.take(idx_in)
                xx_p = xx_p.take(idx_in)
                xy_p = xy_p.take(idx_in)
                yy_p = yy_p.take(idx_in)
                
                nms = idx_in.size
                nw = nms
                goodmove = nms > 0
                # turn bright_n into an array
                bright_n = bright_n - (f0g > 2*trueminf_g) - (fkg > 2*trueminf_g) + (pfg > 2*trueminf_g)
            if goodmove:
                factor = np.log(truealpha_g-1) + (truealpha_g-1)*np.log(trueminf) - truealpha_g*np.log(frac*(1-frac)*sum_f) + \
                    np.log(2*np.pi*kickrange_g*kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + \
                    np.log(sum_f) + np.log(xx_p) + np.log(np.abs(xy_p)) + np.log(yy_p) - 3*np.log(frac) - 3*np.log(1-frac) # last line is Jacobian #TODO galaxy prior
                if not splitsville:
                    factor *= -1
                    factor += penalty_g
                    factor += log_prior_moments(pxxg, pxyg, pyyg) - log_prior_moments(xx0g, xy0g, yy0g) - log_prior_moments(xxkg, xykg, yykg)
                else:
                    factor -= penalty_g
                    factor += log_prior_moments(pxxg, pxyg, pyyg) + log_prior_moments(bxxg, bxyg, byyg) - log_prior_moments(xx0g, xy0g, yy0g)
        # endif rtype   
        nmov[i] = nw
        dt1[i] = time.clock() - t1

        if goodmove:
            t2 = time.clock()
            xtemp = []
            ytemp = []
            ftemp = []
            if idx_move is not None:
                xtemp.extend([ x0, px])
                ytemp.extend([ y0, py])
                ftemp.extend([-f0, pf])
            if idx_move_g is not None:
                xposphon, yposphon, fluxphon = retr_tranphon(gridphon, amplphon, np.concatenate([x0g, pxg]), np.concatenate([y0g, pyg]), np.concatenate([-f0g, pfg]), \
                    np.concatenate([xx0g, pxxg]), np.concatenate([xy0g, pxyg]), np.concatenate([yy0g, pyyg]))
                xtemp.append(xposphon)
                ytemp.append(yposphon)
                ftemp.append(fluxphon)
            if do_birth:
                xtemp.append(bx.flatten())
                ytemp.append(by.flatten())
                ftemp.append(bf.flatten())
            if idx_kill is not None:
                xtemp.append(xk.flatten())
                ytemp.append(yk.flatten())
                ftemp.append(-fk.flatten())
            if do_birth_g:
                xposphon, yposphon, fluxphon = retr_tranphon(gridphon, amplphon, bxg, byg, bfg, bxxg, bxyg, byyg)
                xtemp.append(xposphon)
                ytemp.append(yposphon)
                ftemp.append(fluxphon)
            if idx_kill_g is not None:
                xposphon, yposphon, fluxphon = retr_tranphon(gridphon, amplphon, xkg, ykg, -fkg, xxkg, xykg, yykg)
                xtemp.append(xposphon)
                ytemp.append(yposphon)
                ftemp.append(fluxphon)

            dmodel, diff2 = image_model_eval(np.concatenate(xtemp), np.concatenate(ytemp), np.concatenate(ftemp), dback, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
            plogL = -0.5*diff2
            dt2[i] = time.clock() - t2

            t3 = time.clock()
            nregx = imsz[0] / regsize + 1
            nregy = imsz[1] / regsize + 1
            refx = None
            refy = None
            if idx_move is not None:
                refx = x0
                refy = y0
            else: # merges and splits evaluated in idx_move region
                if idx_move_g is not None:
                    refx = x0g
                    refy = y0g
                elif do_birth:
                    refx = bx if bx.ndim == 1 else bx[:,0]
                    refy = by if by.ndim == 1 else by[:,0]
                elif idx_kill is not None:
                    refx = xk if xk.ndim == 1 else xk[:,0]
                    refy = yk if yk.ndim == 1 else yk[:,0]
                elif do_birth_g:
                    refx = bxg
                    refy = byg
                elif idx_kill_g is not None:
                    refx = xkg
                    refy = ykg
            regionx = get_region(refx, offsetx, regsize)
            regiony = get_region(refy, offsety, regsize)

            plogL[(1-parity_y)::2,:] = float('-inf') # don't accept off-parity regions
            plogL[:,(1-parity_x)::2] = float('-inf')
            dlogP = (plogL - logL) / temperature
            if factor is not None:
                dlogP[regiony, regionx] += factor
            acceptreg = (np.log(np.random.uniform(size=(nregy, nregx))) < dlogP).astype(np.int32)
            acceptprop = acceptreg[regiony, regionx]
            numaccept = np.count_nonzero(acceptprop)

            # only keep dmodel in accepted regions+margins
            dmodel_acpt = np.zeros_like(dmodel)
            libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodel, dmodel_acpt, acceptreg, regsize, margin, offsetx, offsety)
            # using this dmodel containing only accepted moves, update logL
            diff2.fill(0)
            libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resid, weight, diff2, regsize, margin, offsetx, offsety)
            logL = -0.5*diff2
            resid -= dmodel_acpt # has to occur after pcat_like_eval, because resid is used as ref
            model += dmodel_acpt
            # implement accepted moves
            if idx_move is not None:
                px_a = px.compress(acceptprop)
                py_a = py.compress(acceptprop)
                pf_a = pf.compress(acceptprop)
                idx_move_a = idx_move.compress(acceptprop)
                x[idx_move_a] = px_a
                y[idx_move_a] = py_a
                f[idx_move_a] = pf_a
            if idx_move_g is not None:
                pxg_a = pxg.compress(acceptprop)
                pyg_a = pyg.compress(acceptprop)
                pfg_a = pfg.compress(acceptprop)
                pxxg_a=pxxg.compress(acceptprop)
                pxyg_a=pxyg.compress(acceptprop)
                pyyg_a=pyyg.compress(acceptprop)
                idx_move_a = idx_move_g.compress(acceptprop)
                xg[idx_move_a] = pxg_a
                yg[idx_move_a] = pyg_a
                fg[idx_move_a] = pfg_a
                xxg[idx_move_a]=pxxg_a
                xyg[idx_move_a]=pxyg_a
                yyg[idx_move_a]=pyyg_a
            if do_birth:
                bx_a = bx.compress(acceptprop, axis=0).flatten()
                by_a = by.compress(acceptprop, axis=0).flatten()
                bf_a = bf.compress(acceptprop, axis=0).flatten()
                num_born = bf_a.size # works for 1D or 2D
                x[n:n+num_born] = bx_a
                y[n:n+num_born] = by_a
                f[n:n+num_born] = bf_a
                n += num_born
            if do_birth_g:
                bxg_a = bxg.compress(acceptprop)
                byg_a = byg.compress(acceptprop)
                bfg_a = bfg.compress(acceptprop)
                bxxg_a=bxxg.compress(acceptprop)
                bxyg_a=bxyg.compress(acceptprop)
                byyg_a=byyg.compress(acceptprop)
                num_born = np.count_nonzero(acceptprop)
                xg[ng:ng+num_born] = bxg_a
                yg[ng:ng+num_born] = byg_a
                fg[ng:ng+num_born] = bfg_a
                xxg[ng:ng+num_born]=bxxg_a
                xyg[ng:ng+num_born]=bxyg_a
                yyg[ng:ng+num_born]=byyg_a
                ng += num_born
            if idx_kill is not None:
                idx_kill_a = idx_kill.compress(acceptprop, axis=0).flatten()
                num_kill = idx_kill_a.size
                # nstar is correct, not n, because x,y,f are full nstar arrays
                x[0:nstar-num_kill] = np.delete(x, idx_kill_a)
                y[0:nstar-num_kill] = np.delete(y, idx_kill_a)
                f[0:nstar-num_kill] = np.delete(f, idx_kill_a)
                x[nstar-num_kill:] = 0
                y[nstar-num_kill:] = 0
                f[nstar-num_kill:] = 0
                n -= num_kill
            if idx_kill_g is not None:
                idx_kill_a = idx_kill_g.compress(acceptprop)
                num_kill = idx_kill_a.size
                # like above, ngalx is correct
                xg[0:ngalx-num_kill] = np.delete(xg, idx_kill_a)
                yg[0:ngalx-num_kill] = np.delete(yg, idx_kill_a)
                fg[0:ngalx-num_kill] = np.delete(fg, idx_kill_a)
                xxg[0:ngalx-num_kill]=np.delete(xxg, idx_kill_a)
                xyg[0:ngalx-num_kill]=np.delete(xyg, idx_kill_a)
                yyg[0:ngalx-num_kill]=np.delete(yyg, idx_kill_a)
                xg[ngalx-num_kill:] = 0
                yg[ngalx-num_kill:] = 0
                fg[ngalx-num_kill:] = 0
                xxg[ngalx-num_kill:]= 0
                xyg[ngalx-num_kill:]= 0
                yyg[ngalx-num_kill:]= 0
                ng -= num_kill
            dt3[i] = time.clock() - t3

            # hmm...
            #back += dback
            if acceptprop.size > 0: 
                accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
            else:
                accept[i] = 0
        else:
            outbounds[i] = 1

    chi2 = np.sum(weight*(data-model)*(data-model))
    fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
    print 'Temperature', temperature, 'background', back, 'N_star', n, 'N_gal', ng, 'N_phon', n_phon, 'chi^2', chi2
    '''print 'Acceptance'+fmtstr % (np.mean(accept), np.mean(accept[movetype == 0]), np.mean(accept[movetype == 2]), np.mean(accept[movetype == 3]), np.mean(accept[movetype==4]), np.mean(accept[movetype==5]), np.mean(accept[movetype==6]), np.mean(accept[movetype==8]), np.mean(accept[movetype==9]))
    print 'Out of bounds'+fmtstr % (np.mean(outbounds), np.mean(outbounds[movetype == 0]), np.mean(outbounds[movetype == 2]), np.mean(outbounds[movetype == 3]), np.mean(outbounds[movetype==4]), np.mean(outbounds[movetype==5]), np.mean(outbounds[movetype==6]), np.mean(outbounds[movetype==8]), np.mean(outbounds[movetype==9]))
    print '# src pert\t(all) %0.1f (P) %0.1f (B-D) %0.1f (M-S) %0.1f' % (np.mean(nmov), np.mean(nmov[movetype == 0]), np.mean(nmov[movetype == 2]), np.mean(nmov[movetype == 3]))
    print '-'*16
    dt1 *= 1000
    dt2 *= 1000
    dt3 *= 1000
    print 'Proposal (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt1), np.mean(dt1[movetype == 0]) , np.mean(dt1[movetype == 2]), np.mean(dt1[movetype == 3]))
    print 'Likelihood (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt2), np.mean(dt2[movetype == 0]) , np.mean(dt2[movetype == 2]), np.mean(dt2[movetype == 3]))
    print 'Implement (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt3), np.mean(dt3[movetype == 0]) , np.mean(dt3[movetype == 2]), np.mean(dt3[movetype == 3]))
    print '='*16'''

    if visual:
            plt.figure(1)
            plt.clf()
            plt.subplot(1,3,1)
            plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            sizefac = 10.*136
            if datatype == 'mock':
                if strgmode == 'star' or strgmode == 'galx':
                    mask = truef > 250 # will have to change this for other data sets
                    plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='lime')
                    mask = np.logical_not(mask)
                    plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='g')
                if strgmode == 'galx':
                    plt.scatter(truexg, trueyg, marker='1', s=truefg / sizefac, color='lime')
                    plt.scatter(truexg, trueyg, marker='o', s=truerng*truerng*4, edgecolors='lime', facecolors='none')
                if strgmode == 'stargalx':
                    plt.scatter(dictglob['truexposstar'], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='g')
            plt.scatter(x[0:n], y[0:n], marker='x', s=f[0:n]/sizefac, color='r')
            if strgmode == 'galx':
                plt.scatter(xg[0:ng], yg[0:ng], marker='2', s=fg[0:ng]/sizefac, color='r')
                a, theta, phi = from_moments(xxg[0:ng], xyg[0:ng], yyg[0:ng])
                plt.scatter(xg[0:ng], yg[0:ng], marker='o', s=4*a*a, edgecolors='red', facecolors='none')
            plt.xlim(-0.5, imsz[0]-0.5)
            plt.ylim(-0.5, imsz[1]-0.5)
            plt.subplot(1,3,2)
            plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
            if j == 0:
                plt.tight_layout()
            plt.subplot(1,3,3)

            if datatype == 'mock':
                plt.hist(np.log10(truef), range=(np.log10(trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
                plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
            else:
                plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.ceil(np.log10(np.max(f[0:n])))), log=True, alpha=0.5, label='Chain', histtype='step')
            plt.legend()
            plt.xlabel('log10 flux')
            plt.ylim((0.5, nstar))
            plt.draw()
            plt.pause(1e-5)


    return n, ng, chi2

'''def pool_task(k):
    return run_sampler(x_all[k], y_all[k], f_all[k], n_all[k], \
            xg_all[k], yg_all[k], fg_all[k], xxg_all[k], xyg_all[k], yyg_all[k], ng_all[k], temps[k])

from multiprocessing import Pool
pool = Pool(ntemps)'''

if visual:
    plt.ion()
    plt.figure(figsize=(15,5))
for j in xrange(nsamp):
    chi2_all = np.zeros(ntemps)
    print 'Loop', j
    #results = pool.map(pool_task, range(ntemps))
    #for k, r in enumerate(results):
    #    n_all[k], ng_all[k], chi2_all[k] = r
    #    print k, chi2_all[k]

    temptemp = max(temp_init - cooling_rate*j, 1)
    for k in xrange(ntemps):
        n_all[k], ng_all[k], chi2_all[k] = run_sampler(x_all[k], y_all[k], f_all[k], n_all[k], \
            xg_all[k], yg_all[k], fg_all[k], xxg_all[k], xyg_all[k], yyg_all[k], ng_all[k], temptemp, visual=(k==0)*visual)
        #    xg_all[k], yg_all[k], fg_all[k], xxg_all[k], xyg_all[k], yyg_all[k], ng_all[k], temps[k], visual=(k==0))

    for k in xrange(ntemps-1, 0, -1):
        logfac = (chi2_all[k-1] - chi2_all[k]) * (1./temps[k-1] - 1./temps[k]) / 2.
        if np.log(np.random.uniform()) < logfac:
            print 'swapped', k-1, k
            n_all[k-1], n_all[k] = n_all[k], n_all[k-1]
            x_all[k-1], x_all[k] = x_all[k], x_all[k-1]
            y_all[k-1], y_all[k] = y_all[k], y_all[k-1]
            f_all[k-1], f_all[k] = f_all[k], f_all[k-1]
            ng_all[k-1], ng_all[k] = ng_all[k], ng_all[k-1]
            xg_all[k-1], xg_all[k] = xg_all[k], xg_all[k-1]
            yg_all[k-1], yg_all[k] = yg_all[k], yg_all[k-1]
            fg_all[k-1], fg_all[k] = fg_all[k], fg_all[k-1]
            xxg_all[k-1], xxg_all[k] = xxg_all[k], xxg_all[k-1]
            xyg_all[k-1], xyg_all[k] = xyg_all[k], xyg_all[k-1]
            yyg_all[k-1], yyg_all[k] = yyg_all[k], yyg_all[k-1]

    nsample[j] = n_all[0]
    xsample[j,:] = x_all[0]
    ysample[j,:] = y_all[0]
    fsample[j,:] = f_all[0]
    ngsample[j] = ng_all[0]
    xgsample[j,:] = xg_all[0]
    ygsample[j,:] = yg_all[0]
    fgsample[j,:] = fg_all[0]
    xxgsample[j,:] = xxg_all[0]
    xygsample[j,:] = xyg_all[0]
    yygsample[j,:] = yyg_all[0]
    chi2sample[j] = chi2_all[0]

print 'saving...'
fname = 'chain_T' + str(temp_init) + '.npz'
np.savez(fname, n=nsample, x=xsample, y=ysample, f=fsample, ng=ngsample, xg=xgsample, yg=ygsample, fg=fgsample, xxg=xxgsample, xyg=xygsample, yyg=yygsample,chi2=chi2sample)
