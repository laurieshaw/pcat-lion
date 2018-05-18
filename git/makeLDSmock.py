import numpy as np
from image_eval import image_model_eval, psf_poly_fit
import matplotlib.pyplot as plt

normalPSF = True
loadTruth = False
RANDOM = True
truthFile = 'Data/mock1000_LDS_tru.txt'
flabel = '_LDS_TEST2_'
psf_fact = 1.0

imsz = (100, 100) # image size width, height
trueback = np.float32(179.) # 179.
gain = np.float32(1.) 

if normalPSF:
    print "**** USING GAUSSIAN PSF **** "
    pixel_scale = 0.2 # arcsec
    sigma_psf = psf_fact*0.5/2.355 # arcsec, fwhm=0.5
    sigma_in_pix = sigma_psf / pixel_scale
    nc = 25 # PSF postage stamp size
    nbin = 5 # upsampling
    x = np.arange(-(nc/2), (nc/2)+1, 1/float(nbin))
    r2 = x[:,None]*x[:,None] + x[None,:]*x[None,:]
    psf = np.exp(-r2/(2.*sigma_in_pix*sigma_in_pix)) / (2*np.pi*sigma_in_pix*sigma_in_pix)
    cf = psf_poly_fit(psf, nbin=nbin)
    npar = cf.shape[0]
    print np.sum(psf)*0.2*0.2
else:
    f = open('Data/sdss.0921_psf.txt')
    nc, nbin = [np.int32(i) for i in f.readline().split()]
    f.close()
    psf = np.loadtxt('Data/sdss.0921_psf.txt', skiprows=1).astype(np.float32)
    cf = psf_poly_fit(psf, nbin=nbin)
    npar = cf.shape[0]


#plt.imshow(psf)
#plt.show()
print npar, nc, nbin
#np.random.seed(20170501) # set seed to always get same catalogue

if loadTruth:
    truth = np.loadtxt(truthFile)
    truex = truth[:,0].astype(np.float32)
    truey = truth[:,1].astype(np.float32)
    truef = truth[:,2].astype(np.float32)
    nstar = len(truef)
else:
    nstar = 2000
    buf = 2
    if RANDOM:
        truex = (np.random.uniform(size=nstar,low=buf,high=imsz[0]-buf)).astype(np.float32)
        truey = (np.random.uniform(size=nstar,low=buf,high=imsz[1]-buf)).astype(np.float32)    
        truealpha = np.float32(1.5)
        trueminf = np.float32(100.)
        truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)
        truef = trueminf * np.exp(truelogf)
        truef[truef>1e7] = 1.e7
        #truef[0] = 1.0e9
    else:
        minx = buf
        miny = buf
        max_x = imsz[0]-buf
        max_y = imsz[1]-buf
        rhol = np.sqrt( nstar/float( max_x*max_y ) )
        nx = np.ceil( max_x*rhol )
        ny = np.ceil( max_y*rhol )
        lx = max_x/float(nx)
        ly = max_y/float(ny)
        truex = np.tile( np.arange(nx)*lx + lx/2. + minx, (ny,1) ).flatten().astype(np.float32)
        truey = np.tile( np.arange(ny)*ly + ly/2. + minx, (nx,1) ).flatten('F').astype(np.float32)
        ind = np.random.choice(int(nx*ny), size=nstar, replace=False)
        truef = np.ones_like(truex)*20000

print "Max flux %1.3e" % (max(truef))

noise = np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32)
mock = image_model_eval(truex, truey, truef, trueback, imsz, nc, cf)
mock[mock < 1] = 1. # maybe some negative pixels
variance = trueback / gain # mock/gain
mock += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

plt.imshow(mock,origin='lower',interpolation='None')
plt.show()

fname = 'Data/mock' + flabel

f = open(fname+'pix.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f' % (imsz[0], imsz[1], gain))
f.close()

np.savetxt(fname+'cts.txt', mock)

np.savetxt(fname+'psf.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

truth = np.array([truex, truey, truef]).T
np.savetxt(fname+'tru.txt', truth)
