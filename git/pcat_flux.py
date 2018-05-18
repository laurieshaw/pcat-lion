import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double, c_float
#import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
#import matplotlib
##matplotlib.use('TkAgg')
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import time
#import astropy.wcs
#import astropy.io.fits
import sys
import os
import warnings
from pcat_subroutines import *

from image_eval import psf_poly_fit, image_model_eval, image_model_eval_flux, image_model_eval_flux_inregions, image_model_eval_determinants
from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

# c declarations
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
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
#
libmmult.pcat_flux_estimate.restype = None
libmmult.pcat_flux_estimate.argtypes = [c_int, c_int, c_int, c_int, c_int, c_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, array_2d_double, array_2d_double, array_1d_int, array_2d_float, array_1d_float, array_1d_float]
#
libmmult.invert_matrix.restype = None
libmmult.invert_matrix.argtypes = [array_2d_double, array_2d_double, array_1d_int, c_int]
#
libmmult.pcat_model_subtract.restype = None
libmmult.pcat_model_subtract.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float]
#
#
libmmult.position_sample_accept.restype = None
libmmult.position_sample_accept.argtypes = [array_1d_float, array_1d_float, c_int, c_int, c_int, array_1d_int, c_float]
#

if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
    warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)

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
#psf = np.loadtxt('Data/'+'mock500_LDS_psf0p95'+'_psf.txt', skiprows=1).astype(np.float32) # test case: mismatched psf
psf = np.loadtxt('Data/'+dataname+'_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]


if visual and testpsfn:
    testpsf(nc, cf, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

f = open('Data/'+dataname+'_pix.txt')
w, h, nband = [np.int32(i) for i in f.readline().split()]
imsz = (w, h)
print imsz
assert nband == 1
bias, gain = [np.float32(i) for i in f.readline().split()]
f.close()
print bias, gain
data = np.loadtxt('Data/'+dataname+'_cts.txt').astype(np.float32)
data -= bias

print 'Lion mode:', strgmode
print 'datatype:', datatype

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'_tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2]
        truex = truex[truef>250]
        truey = truey[truef>250]
        truef = truef[truef>250]
        print "True N: %d" % (len(truef))
        ntrue = len(truef)
        #plt.plot(truex,truey,'k+')
        #plt.show()
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

trueback = np.float32(179.)
s_psf = 0.5/2.355/0.2 # FWHM/2.355/pixel_side, FWHM=0.5arcmin,pixel_side=0.2arcmin
Neff = 4*np.pi*s_psf**2
dpos_rms_floor = 0.005 # from Stephen's paper
dpos_burnin_fact = 100.
dpos_burnin_scale = 2000.
print "Trueback = %d" % (trueback)
variance = trueback *np.ones_like(data) / gain
weight = 1. / variance # inverse variance
noise_per_pixel = np.sqrt( trueback/gain )

# number of stars to use in fit
nstar = 2000#
trueminf = np.float32(250) #  np.float32(250.*136) WHY *136??
truealpha = np.float32(2.00)

flux_prior = 0.01*calc_flux_prior(nstar, max(truealpha,2.0), trueminf)

back = trueback
ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)


min_sep = 1.0
#pos_prop_fact = 0.2
penalty = 1.5
regsize = 20
print "regsize = %d" % (regsize)
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10
print "Margin = %d" % (margin)
#print "Parity currently = 0"
kickrange = 2.0 


n = ntrue/2 # Don't want this to start too small as need at least one phonion/region np.random.randint(nstar)+1
x = np.zeros(nstar).astype(np.float32)
y = np.zeros(nstar).astype(np.float32)
f = np.zeros(nstar).astype(np.float32)
x[:n],y[:n] = init_phonion_distribution(n,imsz,min_sep=min_sep,perturb=True, img_buf = 0) # LDS: Make img_buf a proper parameter
assert position_sample_accept(x[:n],y[:n],x[:n],y[:n],min_sep2=min_sep*min_sep,lib=libmmult.position_sample_accept)[0]

temp, temp, f[:n], temp = image_model_eval_flux(x[:n], y[:n], back, imsz, nc, cf, weights=weight, ref=data.copy(), 
                                                lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=0, offsety=0, lib2=libmmult.pcat_flux_estimate)
x[n:] = 0.
y[n:] = 0.
f[n:] = 0.

nsamp = 1000
nloop = 1000
fname_tag = dataname +'_f1p0'
#temp_init = 1.0
#cooling_rate = 2.*temp_init/float(nsamp) # for simulated annealing
nsample = np.zeros(nsamp, dtype=np.int32)
chi2sample = np.zeros(nsamp,dtype=float)
detOsample = np.zeros(nsamp,dtype=float)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
fsample = np.zeros((nsamp, nstar), dtype=np.float32)


def run_sampler(x, y, f, n, nloop=1000, visual=False, samplenum=0):
    t0 = time.clock()
    nmov = np.zeros(nloop)
    movetype = np.zeros(nloop)
    accept = np.zeros(nloop)
    outbounds = np.zeros(nloop)
    dt1 = np.zeros(nloop)
    dt2 = np.zeros(nloop)
    dt3 = np.zeros(nloop)
    dts_imeval = np.zeros((nloop,3))

    # offset gives offset of regions from edge of image
    offsetx = np.random.randint(regsize)
    offsety = np.random.randint(regsize)
    nregx = imsz[0] / regsize + 1
    nregy = imsz[1] / regsize + 1

    resid = data.copy() # residual for zero image is data
    if strgmode == 'star':
        evalx = x[0:n]
        evaly = y[0:n]
        evalf = f[0:n]

    # make sure that proximity criterion is still upheld (it won't be because of M-S moves)
    good,i =  position_sample_accept(x[:n],y[:n],x[:n],y[:n],min_sep2=min_sep*min_sep,lib=libmmult.position_sample_accept)
    if not good:
        print "FAIL", x[i],y[i]
        assert False

    model, diff2, logdetO = image_model_eval_determinants(evalx, evaly, evalf, back, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval,
                                                          regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety, lib2=libmmult.pcat_flux_estimate )

    logL = -0.5*diff2 - 0.5*logdetO
    resid -= model

    # DEBUG - turn everything but move off
    if samplenum*nloop<5000:
        moveweights = np.array([80.,0.,0.,0.])
    else:
        moveweights = np.array([80.,0.,80.,0.])
    moveweights /= np.sum(moveweights)
    movetypes = 'Birth/Death: '
    if moveweights[2]==0:
        movetypes += "off, "
    else:
        movetypes += "on, "
    movetypes += "Split/Merge: "
    if moveweights[3]==0:
        movetypes += "off "
    else:
        movetypes += "on "
    print "%s" % (movetypes)

    # ADDED BY LDS TO PREVENT OOB
    img_buf = 1

    pcheck = np.zeros(nloop)

    for i in xrange(nloop):
        t1 = time.clock()
        rtype = np.random.choice(moveweights.size, p=moveweights)
        #print rtype
        movetype[i] = rtype
        #print "rtype: %d" % (rtype)
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
	do_birth = False
	idx_kill = None
        # mover
        if rtype == 0:
            # get indices of stars in regions
            idx_move = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            idx_fixed = not_idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            nw = idx_move.size

            f0 = f.take(idx_move)
            ftemp = f0.copy()
            ftemp[ftemp<trueminf] = trueminf

            # ** What do I do about the fluxes in dpos_rms? **
            dpos_rms = np.clip( np.sqrt( 2.0 * Neff * s_psf**2 * trueback )/ftemp, a_min=dpos_rms_floor, a_max=None )
            dpos_rms = np.float32( dpos_rms / np.sqrt( nw/ float(nregx*nregy/4.0 ) ) ) # factor of 6 because stars moved are spread over roughly 6 regions 
            # allow larger moves in burn-in
            if samplenum*nloop + i < 10*dpos_burnin_scale:
                dpos_rms *= np.float32( dpos_burnin_fact * np.exp( -1.*(samplenum*nloop + i) / float(dpos_burnin_scale) ) + 1 )
            #print np.min(dpos_rms),np.max(dpos_rms)
            x0 = x.take(idx_move)
            y0 = y.take(idx_move)
            xf = x.take(idx_fixed)
            yf = y.take(idx_fixed)

            pcheck[i] = time.clock()
            dx,dy,dpos_rms = sample_positions_with_constraint(xf,yf,x0,y0,dpos_rms=dpos_rms,imsz=imsz,img_buf=img_buf,min_sep=min_sep,lib=libmmult.position_sample_accept)
            pcheck[i] = time.clock() - pcheck[i]
            px = x0 + dx
            py = y0 + dy           

            # *** Set image edges further in to allow flux estimation. Will need to turn this off.

            xregions = get_region(x0, offsetx, regsize)
            yregions = get_region(y0, offsety, regsize)

            goodmove = True # always True because we bounce off the edges of the image and fmin
        # background change
        elif rtype == 1:
            dback = np.float32(np.random.normal())
            goodmove = True 
        # birth and death
        elif rtype == 2:
            lifeordeath = np.random.randint(2)
            nbd = (nregx * nregy) / 4
            # LDS: moving this here - will need it for flux evaluation
            idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            #print idx_reg
            idx_fixed = not_idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            x0 = x.take(idx_reg)
            y0 = y.take(idx_reg)
            f0 = f.take(idx_reg)
            # birth
            if lifeordeath and n < nstar: # need room for at least one source
                nbd = min(nbd, nstar-n) # add nbd sources, or just as many as will fit
                                        # mildly violates detailed balance when n close to nstar
                # want number of regions in each direction, divided by two, rounded up
                mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
                mregy = ((imsz[1] / regsize + 1) + 1) / 2
                bx = ((np.random.randint(mregx, size=nbd)*2 + parity_x + np.random.uniform(size=nbd))*regsize - offsetx).astype(np.float32)
                by = ((np.random.randint(mregy, size=nbd)*2 + parity_y + np.random.uniform(size=nbd))*regsize - offsety).astype(np.float32)

                # don't birth things too close to another source
                iret = np.array([0],dtype=np.int32)
                while ( not position_sample_accept(np.concatenate((bx,x[:n])),np.concatenate((by,y[:n])),bx,by,min_sep2=min_sep*min_sep, lib=libmmult.position_sample_accept,iret=iret)[0] ):
                    bx = ((np.random.randint(mregx, size=nbd)*2 + parity_x + np.random.uniform(size=nbd))*regsize - offsetx).astype(np.float32)
                    by = ((np.random.randint(mregy, size=nbd)*2 + parity_y + np.random.uniform(size=nbd))*regsize - offsety).astype(np.float32)

		# some sources might be generated outside image
		inbounds = (bx > img_buf) * (bx < (imsz[0] - img_buf)) * (by > img_buf) * (by < imsz[1] - img_buf)
		idx_in = np.flatnonzero(inbounds)
                nw = idx_in.size
		bx = bx.take(idx_in)
                by = by.take(idx_in)

                if len(bx)==0 and len(by)==0:
                    goodmove=False
                else:
                    # put all this in a function eventually
                    # find indices of things in the same regions as bx, by
                    bx_reg = get_region(bx, offsetx, regsize )
                    by_reg = get_region(by, offsety, regsize )
                    x0_reg = get_region(x0, offsetx, regsize )
                    y0_reg = get_region(y0, offsety, regsize )

                    b_regs =  zip(bx_reg,by_reg)
                    in_region_mask = np.array([item in b_regs for item in zip(x0_reg,y0_reg)])

                    x0 = x0.compress(in_region_mask)
                    y0 = y0.compress(in_region_mask)
                    f0 = f0.compress(in_region_mask)
                    idx_reg = idx_reg.compress(in_region_mask)

                    do_birth = True

                    px = np.concatenate( (x0,bx ) )
                    py = np.concatenate( (y0,by ) )

                    xregions = get_region(px, offsetx, regsize)
                    yregions = get_region(py, offsety, regsize)

                    # modify this to deal with flux marginalisation
                    factor = np.full(nw, -penalty) + 0.5*np.log(np.pi/2.) + np.log(noise_per_pixel) - np.log( flux_prior )
                    goodmove = True
            # death
            # does region based death obey detailed balance?
            elif not lifeordeath and n > 0: # need something to kill
		nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
                nw = nbd
                if nbd > 0:
                    idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
		    xk = x.take(idx_kill)
                    yk = y.take(idx_kill)
                    fk = f.take(idx_kill)

                    # first eliminate regions that are unaffected
                    # put all this in a function eventually
                    # find indices of things in the same regions as xk, yk
                    xk_reg = get_region(xk, offsetx, regsize )
                    yk_reg = get_region(yk, offsety, regsize )
                    x0_reg = get_region(x0, offsetx, regsize )
                    y0_reg = get_region(y0, offsety, regsize )
                    
                    k_regs = zip(xk_reg,yk_reg)
                    in_region_mask = np.array([item in k_regs for item in zip(x0_reg,y0_reg)])

                    #print in_region_mask

                    # ignore phonions not in regions in which a phonion was killed
                    x0 = x0.compress(in_region_mask)
                    y0 = y0.compress(in_region_mask)
                    f0 = f0.compress(in_region_mask)
                    idx_reg = idx_reg.compress(in_region_mask)

                    # idx_reg now contains indices of phonions in a region in which a phonion was killed (included phonions that are killed, themselves)

                    # we want indices of things remaining in regions
                    # i.e, the indices in idx_reg that are not in idx_kill
                    idx_reg = np.setdiff1d( idx_reg, idx_kill , assume_unique=True) # flag speeds up calculation apparently

                    # idx_reg should now contain the indices of remaining phonions in regions in which a phonion was killed,

                    px = x.take(idx_reg)
                    py = y.take(idx_reg)

                    xregions = get_region(px, offsetx, regsize)
                    yregions = get_region(py, offsety, regsize)

                    factor = np.full(nbd, penalty) - 0.5*np.log(np.pi/2.) - np.log(noise_per_pixel) + np.log( flux_prior )
                    goodmove = True
                else:
                    goodmove = False
        # merges and splits
        elif rtype == 3:
            # should now be able to set trueminf = 0
            splitsville = np.random.randint(2)
            idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            idx_fixed = not_idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            # these are the phonions in region
            x0 = x.take(idx_reg)
            y0 = y.take(idx_reg)
            f0 = f.take(idx_reg)
            sum_f = 0
            low_n = 0
            # select objects suitable for splitting
            idx_bright = idx_reg.take(np.flatnonzero( np.abs( f.take(idx_reg) ) > 2*trueminf)) # in region!
            bright_n = idx_bright.size

            nms = (nregx * nregy) / 4
            # split
            if splitsville and n > 0 and n < nstar and bright_n > 0: # need something to split, but don't exceed nstar

                #print "SPLIT MOVE"
                nms = min(nms, bright_n, nstar-n) # need bright source AND room for split source

                idx_move = np.random.choice(idx_bright, size=nms, replace=False)
                # original positions and fluxes of phonions we will move
                xm = x.take(idx_move)
                ym = y.take(idx_move)
		fm = f.take(idx_move)

                # now need to know positions of everything that didn't move
                xtemp = x.copy()
                ytemp = y.copy()
                xtemp[idx_move] = -10000
                ytemp[idx_move] = -10000
                '''
                dx = (np.random.normal(size=nms)*kickrange).astype(np.float32)
                dy = (np.random.normal(size=nms)*kickrange).astype(np.float32)

                # force kicks to be at least a certain distance
                kickmask =  dx**2 + dy**2 < min_sep*min_sep
                while np.sum( kickmask ) > 0:
                    dx[kickmask] =  (np.random.normal(size=np.sum(kickmask))*kickrange).astype(np.float32)
                    dy[kickmask] =  (np.random.normal(size=np.sum(kickmask))*kickrange).astype(np.float32)
                    kickmask[kickmask] = dx[kickmask]**2 + dy[kickmask]**2 < min_sep*min_sep

                # new positions for phonions that move
                frac = 0.5 # hard code this for now. (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                pxm = xm + ((1-frac)*dx)
                pym = ym + ((1-frac)*dy)
                do_birth = True
                bx = xm - frac*dx
                by = ym - frac*dy
                '''

                do_birth = True
                pxm,pym,bx,by = sample_kicks_with_constraint(xtemp,ytemp,xm,ym,nms,kickrange,min_sep=0.5)

                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (pxm > 0) * (pxm < imsz[0] - 1) * (pym > 0) * (pym < imsz[1] - 1) * \
                           (bx > 0) * (bx < imsz[0] - 1) * (by > 0) * (by < imsz[1] - 1)
                idx_in = np.flatnonzero(inbounds)
                xm = xm.take(idx_in)
                ym = ym.take(idx_in)
                pxm = pxm.take(idx_in)
                pym = pym.take(idx_in)
                bx = bx.take(idx_in)
                by = by.take(idx_in)
                idx_move = idx_move.take(idx_in)
                goodmove = idx_in.size > 0

                # idx_move gives indices of phonions that split. need array like x,y but with moved phonions replaced with their new positions.
                px = x.copy()
                py = y.copy()
                # change positions of phonions that were moved
                px[idx_move] = pxm
                py[idx_move] = pym

                px = px.take(idx_reg)
                py = py.take(idx_reg)

                # for birth moves, take the position of the parent phonion (need to evaluate the regions based on original positions)
                xregions = get_region(np.concatenate( (x0, xm) ), offsetx, regsize)
                yregions = get_region(np.concatenate( (y0, ym) ), offsety, regsize)

                # now combine with the new phonions
                px = np.concatenate((px,bx))
                py = np.concatenate((py,by))

                # need to calculate factor
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
                #print "MERGE MOVE"
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

                xm = x.take(idx_move) # phonions to move
                ym = y.take(idx_move) # phonions to move
                #fm = f.take(idx_move) 
                xk = x.take(idx_kill) # phonions to kill
                yk = y.take(idx_kill) # phonions to kill

                frac = 0.5#f0 / sum_f # hardcode to 0.5 for now
                pxm = frac*xm + (1-frac)*xk
                pym = frac*ym + (1-frac)*yk
                
                # idx_move gives indices of phonions that split. need array like x,y but with moved phonions replaced with their new positions.
                px = x.copy()
                py = y.copy()
                # change positions of phonions that were moved
                px[idx_move] = pxm
                py[idx_move] = pym

                # we want indices of things remaining in regions
                # i.e, the indices in idx_reg that are not in idx_kill
                idx_reg = np.setdiff1d( idx_reg, idx_kill , assume_unique=True) # flag speeds up calculation apparently
                # idx_reg should now contain the indices of remaining phonions in regions in which a phonion was killed,

                px = px.take(idx_reg) # remove phonions that were killed from region position arrays
                py = py.take(idx_reg)

                xregions = get_region(x.take(idx_reg), offsetx, regsize)
                yregions = get_region(y.take(idx_reg), offsety, regsize)

                #pf = f0 + fk
                # turn bright_n into an array. related to the assymettry in bright_n moving fowards and backwards in the chain
                #bright_n = bright_n - (f0 > 2*trueminf) - (fk > 2*trueminf) + (pf > 2*trueminf) # LDS: no longer need this for flux marginalisation
            if goodmove:
                factor = 0.5*np.log(np.pi/2.) + np.log(noise_per_pixel) - np.log( flux_prior ) + np.log(2*np.pi*kickrange*kickrange) - min_sep*min_sep / (2.*kickrange*kickrange) - np.log(imsz[0]*imsz[1]) +  np.log(bright_n-1) + np.log(invpairs) # Jacobian is now 1. min_sep added to deal with lower limit on split distances.
                if not splitsville:
                    factor *= -1
                    factor += penalty
                else:
                    factor -= penalty

        # endif rtype   
        nmov[i] = nw
        dt1[i] = time.clock() - t1

        if goodmove:
            t2 = time.clock()
  
            #plt.imshow(model,interpolation='None')
            #plt.show()
            if rtype<=3:
                # rather than the difference in models, this is now returning the latest model itself
                dmodel, diff2, pf, plogdetO, dts = image_model_eval_flux_inregions(x0, y0, f0, px, py, xregions, yregions, model, resid, back, imsz, nc, cf, weights=weight, ref=data, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety, lib2=libmmult.pcat_flux_estimate, lib3=libmmult.pcat_model_subtract )
                plogL = -0.5*diff2 - 0.5*plogdetO
            else:
                assert False # should never be here

            dts_imeval[i,:] = dts
            dt2[i] = time.clock() - t2
            
            t3 = time.clock()
            nregx = imsz[0] / regsize + 1
            nregy = imsz[1] / regsize + 1
            refx = None
            refy = None
            refx_orig = None
            refy_orig = None
            if idx_move is not None and rtype!=3:
                # original positions of phonions that were moved
                refx = x0
                refy = y0
                # calculate factors here now we have new fluxes
                fnew = pf.copy()
                fnew[fnew<trueminf] = trueminf 
                # what would dpos_rms be with new fluxes
                dpos_rms_p = np.clip( np.sqrt( 2.0 * Neff * s_psf**2 * trueback )/fnew, a_min=dpos_rms_floor, a_max=None )
                dpos_rms_p = np.float32( dpos_rms_p / np.sqrt( nw/ float(nregx*nregy/4.) ) ) # factor of 6 because phonions moves are spread over roughly 6 regions
                # allow larger moves in burn-in
                if samplenum*nloop + i < 10.0*dpos_burnin_scale:
                    dpos_rms_p *= np.float32( dpos_burnin_fact * np.exp( -1.*(samplenum*nloop + i + 1) / float(dpos_burnin_scale) ) + 1 )
                    
                factor = 2.0* np.log( dpos_rms / dpos_rms_p ) - (dx**2 + dy**2) / 4.0 * ( 1./ dpos_rms_p**2 - 1./dpos_rms**2 ) #  divided by 4 not 2 because dpos_rms should be for both x&y components here
                #print np.min( factor ), np.max(factor)
            else: # merges and splits evaluated in idx_move region
                if do_birth:
                    refx = bx # if bx.ndim == 1 else bx[:,0]
                    refy = by # if by.ndim == 1 else by[:,0]
                    # phonions in regions that may have had a bd
                    refx_orig = x0
                    refy_orig = y0
                elif idx_kill is not None:
                    refx = xk # if xk.ndim == 1 else xk[:,0]
                    refy = yk # if yk.ndim == 1 else yk[:,0]
                     # phonions in regions that may have had a bd
                    refx_orig = px # because these are length pf
                    refy_orig = py

            # regions of each phonion to move or have flux changed
            regionx = get_region(refx, offsetx, regsize)
            regiony = get_region(refy, offsety, regsize)

            plogL[(1-parity_y)::2,:] = float('-inf') # don't accept off-parity regions
            plogL[:,(1-parity_x)::2] = float('-inf')

            dlogP = (plogL - logL)

            if factor is not None:
                dlogP[regiony, regionx] += factor
            acceptreg = (np.log(np.random.uniform(size=(nregy, nregx))) < dlogP).astype(np.int32)
            acceptprop = acceptreg[regiony, regionx]
            numaccept = np.count_nonzero(acceptprop)
            
            if refx_orig is not None and refy_orig is not None:
                region_origx =  get_region(refx_orig, offsetx, regsize)
                region_origy =  get_region(refy_orig, offsety, regsize)
                acceptprop_orig = acceptreg[region_origy, region_origx]


            # only keep dmodel in accepted regions+margins
            dmodel_acpt = np.zeros_like(dmodel)
            libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodel, dmodel_acpt, acceptreg, regsize, margin, offsetx, offsety)
            # using this dmodel containing only accepted moves, update logL
            diff2.fill(0)
            libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resid, weight, diff2, regsize, margin, offsetx, offsety)

            logdetO[acceptreg.astype(bool)] = plogdetO[acceptreg.astype(bool)]
            
            logL = -0.5*diff2 - 0.5*logdetO # new logL including determinants for each region

            resid -= dmodel_acpt # has to occur after pcat_like_eval, because resid is used as ref
            model += dmodel_acpt

            # implement accepted moves
            if idx_move is not None and rtype!=3: # can NO LONGER use this for merge/splits
                px_a = px.compress(acceptprop)
                py_a = py.compress(acceptprop)
                pf_a = pf.compress(acceptprop)
                idx_move_a = idx_move.compress(acceptprop)
                x[idx_move_a] = px_a
                y[idx_move_a] = py_a
                f[idx_move_a] = pf_a
            if do_birth:
                # split the fluxes into existing and new phonions
                nphon_reg = pf.size
                pf_orig = pf[:nphon_reg-nw] # original phonions in regions
                bf = pf[nphon_reg-nw:] # birth phonions
                # get fluxes for existing phonions in accepted regions
                pf_orig_a = pf_orig.compress(acceptprop_orig)
                # get indices for phonions in accepted regions
                idx_reg_a = idx_reg.compress(acceptprop_orig)
                # update fluxes for original phonions in accepted regions
                f[idx_reg_a] = pf_orig_a
                if rtype==3: # split move: need to update positions of phonions that moved
                    px = px[:nphon_reg-nw] 
                    py = py[:nphon_reg-nw] 
                    # note that most of them didn't move though.
                    px = px.compress(acceptprop_orig)
                    py = py.compress(acceptprop_orig)
                    x[idx_reg_a] = px
                    y[idx_reg_a] = py
                # now collect positions/fluxes for accepted births
                bx_a = bx.compress(acceptprop, axis=0).flatten()
                by_a = by.compress(acceptprop, axis=0).flatten()
                bf_a = bf.compress(acceptprop, axis=0).flatten()
                num_born = bf_a.size # works for 1D or 2D
                # add accepted birth phonions
                x[n:n+num_born] = bx_a 
                y[n:n+num_born] = by_a 
                f[n:n+num_born] = bf_a 
                n += num_born
            if idx_kill is not None:
                # issue here again is updating fluxes
                # indices of killed phonions in accepted regions
                idx_kill_a = idx_kill.compress(acceptprop, axis=0).flatten()
                num_kill = idx_kill_a.size
                # retain indices of remaining phonions in accepted regions
                idx_reg_a = idx_reg.compress(acceptprop_orig)
                # update fluxes of remaining phonions in accepted regions
                f[idx_reg_a] = pf.compress(acceptprop_orig)
                if rtype==3:
                    # for merge steps (of course, most phonions didn't move)
                    x[idx_reg_a] = px.compress(acceptprop_orig)
                    y[idx_reg_a] = py.compress(acceptprop_orig)
                # delete positions & fluxes for killed phonions
                f[0:nstar-num_kill] = np.delete(f, idx_kill_a)
                x[0:nstar-num_kill] = np.delete(x, idx_kill_a)
                y[0:nstar-num_kill] = np.delete(y, idx_kill_a)

                x[nstar-num_kill:] = 0
                y[nstar-num_kill:] = 0
                f[nstar-num_kill:] = 0
                n -= num_kill

            dt3[i] = time.clock() - t3

            if np.mod(i,500)==0 and False:
                success,ifail = position_sample_accept(x[:n],y[:n],x[:n],y[:n],min_sep2=min_sep*min_sep,lib=libmmult.position_sample_accept)
                smax = 1000
                sizefac = 10.*136
                truesize = truef/sizefac*20
                size = np.abs( f[0:n] )/sizefac*20
                truesize[truesize>smax] = smax
                size[size>smax] = smax
                truesize[truesize<10] = 10
                size[size<10] = 10
                negf = f[:n]<0
                posf = f[:n]>=0
                #negsize = -1*f[0:n][f[0:n]<0]/sizefac*20
                #negsize[negsize>smax] = smax
                plt.figure(2)
                plt.clf()
                plt.scatter(truex, truey, marker='+', s=truesize, color='r')
                plt.scatter(x[:n][posf], y[:n][posf], marker='o', s=size[posf], edgecolor='b',facecolor='none')
                plt.scatter(x[:n][negf], y[:n][negf],s=size[negf],edgecolor='g',facecolor='none') 
                plt.text(40,110,'Step = %d' % (i+samplenum*nloop))
                plt.pause(0.00001)
                plt.show()
                if not success and False:
                    print "FAIL", ifail,x[ifail],y[ifail]
                    plt.pause(100)
                    assert False

            if acceptprop.size > 0: 
                accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
            else:
                accept[i] = 0
        else:
            outbounds[i] = 1

    chi2 = np.sum(weight*(data-model)*(data-model))
    fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f'
    print 'background', back, 'N_star', n, 'chi^2', chi2
    print 'Acceptance'+fmtstr % (np.mean(accept), np.mean(accept[movetype == 0]), np.mean(accept[movetype == 2]), np.mean(accept[movetype == 3]) )
    print 'Out of bounds'+fmtstr % (np.mean(outbounds), np.mean(outbounds[movetype == 0]), np.mean(outbounds[movetype == 2]), np.mean(outbounds[movetype == 3]) )
    print '# src pert\t(all) %0.1f (P) %0.1f (B-D) %0.1f (M-S) %0.1f' % (np.mean(nmov), np.mean(nmov[movetype == 0]), np.mean(nmov[movetype == 2]), np.mean(nmov[movetype == 3]))
    print '# fluxes < 0: %d, # fluxes < -trueminf: %d' % (np.sum(f<0),np.sum(f<-trueminf))
    print '-'*16
    dt1 *= 1000
    dt2 *= 1000
    dt3 *= 1000
    print 'Proposal (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt1), np.mean(dt1[movetype == 0]) , np.mean(dt1[movetype == 2]), np.mean(dt1[movetype == 3]))
    print 'Likelihood (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt2), np.mean(dt2[movetype == 0]) , np.mean(dt2[movetype == 2]), np.mean(dt2[movetype == 3]))
    print 'Implement (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt3), np.mean(dt3[movetype == 0]) , np.mean(dt3[movetype == 2]), np.mean(dt3[movetype == 3]))
    print '='*16
    print np.mean(dts_imeval,axis=0)*1000
    print np.mean(pcheck[movetype==0])*1000

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
            plt.scatter(x[0:n], y[0:n], marker='x', s=f[0:n]/sizefac, color='r')
            plt.xlim(-0.5, imsz[0]-0.5)
            plt.ylim(-0.5, imsz[1]-0.5)
            plt.subplot(1,3,2)
            plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
            if j == 0:
                plt.tight_layout()
            plt.subplot(1,3,3)

            if datatype == 'mock':
                plt.hist(np.log10(truef), bins=12, range=(np.log10(trueminf)-1, np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
                plt.hist(np.log10(f[0:n][f[0:n]>0]), bins=12, range=(np.log10(trueminf)-1, np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
            else:
                plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.ceil(np.log10(np.max(f[0:n])))), log=True, alpha=0.5, label='Chain', histtype='step')
            plt.legend()
            plt.xlabel('log10 flux')
            plt.ylim((0.5, nstar))
            plt.pause(0.00001)
            plt.draw()
            plt.show()
    return x, y, f, n, chi2

if visual:
    plt.ion()
    plt.figure(figsize=(15,5))

start_time = time.time()

for j in xrange(nsamp):
    print 'Loop', j

    x, y, f, n, chi2 = run_sampler(x, y, f, n,  visual=visual, nloop=nloop,samplenum=j)
    nsample[j] = n
    xsample[j,:] = x
    ysample[j,:] = y
    fsample[j,:] = f
    chi2sample[j] = chi2


print 'saving...'
fname = 'chain_' + str(fname_tag) + '.npz'
np.savez(fname, n=nsample, x=xsample, y=ysample, f=fsample, chi2=chi2sample)

print("--- %s seconds ---" % (time.time() - start_time))
