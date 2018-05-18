import numpy as np
import scipy.linalg as linalg
#import matplotlib.pyplot as plt
import time
import decimal as decimal

def psf_poly_fit(psf0, nbin):
        assert psf0.shape[0] == psf0.shape[1] # assert PSF is square
        npix = psf0.shape[0]

        # pad by one row and one column
        psf = np.zeros((npix+1, npix+1), dtype=np.float32)
        psf[0:npix, 0:npix] = psf0

        # make design matrix for each nbin x nbin region
        nc = npix/nbin # dimension of original psf
        nx = nbin+1
        y, x = np.mgrid[0:nx, 0:nx] / np.float32(nbin)
        x = x.flatten()
        y = y.flatten()
        A = np.column_stack([np.ones(nx*nx, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).astype(np.float32)
        # output array of coefficients
        cf = np.zeros((A.shape[1], nc, nc), dtype=np.float32)

        # loop over original psf pixels and get fit coefficients
        for iy in xrange(nc):
         for ix in xrange(nc):
                # solve p = A cf for cf
                p = psf[iy*nbin:(iy+1)*nbin+1, ix*nbin:(ix+1)*nbin+1].flatten()
                AtAinv = np.linalg.inv(np.dot(A.T, A))
                ans = np.dot(AtAinv, np.dot(A.T, p))
                cf[:,iy,ix] = ans

        return cf.reshape(cf.shape[0], cf.shape[1]*cf.shape[2])

def image_model_eval(x, y, f, back, imsz, nc, cf, regsize=None, margin=0, offsetx=0, offsety=0, weights=None, ref=None, lib=None):
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert f.dtype == np.float32
        assert cf.dtype == np.float32
        if ref is not None:
                assert ref.dtype == np.float32

        if weights is None:
                weights = np.ones(imsz, dtype=np.float32)
	if regsize is None:
		regsize = max(imsz[0], imsz[1])

        # sometimes phonions are outside image... what is best way to handle?
        goodsrc = (x > 0) * (x < imsz[0] - 1) * (y > 0) * (y < imsz[1] - 1)
        x = x.compress(goodsrc)
        y = y.compress(goodsrc)
        f = f.compress(goodsrc)

        nstar = x.size
        rad = nc/2 # 12 for nc = 25

	nregy = imsz[1]/regsize + 1 # assumes imsz % regsize = 0?
	nregx = imsz[0]/regsize + 1

        ix = np.ceil(x).astype(np.int32)
        dx = ix - x
        iy = np.ceil(y).astype(np.int32)
        dy = iy - y

        dd = np.column_stack((np.ones(nstar, dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f[:, None]

        if lib is None:
                image = back*np.ones((imsz[1]+2*rad+1,imsz[0]+2*rad+1), dtype=np.float32)
                recon = np.dot(dd, cf).reshape((nstar,nc,nc))
                for i in xrange(nstar):
			image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

                image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

                if ref is not None:
                        diff = ref - image
			diff2 = np.zeros((nregy, nregx), dtype=np.float64)
			for i in xrange(nregy):
				y0 = max(i*regsize - offsety - margin, 0)
				y1 = min((i+1)*regsize - offsety + margin, imsz[1])
				for j in xrange(nregx):
					x0 = max(j*regsize - offsetx - margin, 0)
					x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
					subdiff = diff[y0:y1,x0:x1]
					diff2[i,j] = np.sum(subdiff*subdiff*weights[y0:y1,x0:x1])
        else:
                image = back*np.ones((imsz[1], imsz[0]), dtype=np.float32)
                recon = np.zeros((nstar,nc*nc), dtype=np.float32)
                reftemp = ref
                if ref is None:
                        reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
		diff2 = np.zeros((nregy, nregx), dtype=np.float64)
                lib(imsz[0], imsz[1], nstar, nc, cf.shape[0], dd, cf, recon, ix, iy, image, reftemp, weights, diff2, regsize, margin, offsetx, offsety)

        if ref is not None:
                return image, diff2
	else:
		return image

def test_pcat_source_images_cov(x, y, back, imsz, nc, cf, image, lib, lib2, lib3, lib4):
	assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert cf.dtype == np.float32

	image = image.astype(np.float32)
	
        nstar = x.size
        rad = nc/2 # 12 for nc = 25

        ix = np.ceil(x).astype(np.int32)
        dx = ix - x
        iy = np.ceil(y).astype(np.int32)
        dy = iy - y

        dd = np.column_stack((np.ones(nstar, dtype=np.float64), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32)

	t1 = time.clock()
	print ix,iy
	# *******************
	# now get flux vectors: LDS added
	# postage stamp image for each star
	recon = np.dot(dd, cf).reshape((nstar,nc,nc))
	O = SourceSourceCovariance(imsz,x,y,ix,iy,recon,rad)
	# inverse of overlap integral
	Oinv = np.linalg.inv( O ).astype(np.float64)
	logdetO = np.log( np.linalg.det( O ) ).astype(np.float32)
	f = np.empty(nstar).astype(np.float32)
	#print np.shape(recon),np.shape(ref)
	for i in xrange(nstar):
		# ***** assumes no stars near edge of image near edge of (or off) image ******
		# subtracting out background: is this right?
		xmin = max( ix[i]-rad, 0 )
		ymin = max( iy[i]-rad, 0 )
		xmax = min( ix[i]+rad+1, imsz[0])
		ymax = min( iy[i]+rad+1, imsz[1])
		f[i] = np.sum( ( image[ ymin:ymax,xmin:xmax]-back ) * recon[ i,ymin-iy[i]+rad:nc+ymax-iy[i]-rad-1, xmin-ix[i]+rad:nc+xmax-ix[i]-rad-1 ] )
		#f[i] = np.sum( ( image[iy[i]-rad:iy[i]+rad+1,ix[i]-rad:ix[i]+rad+1]-back ) * recon[i,:,:] )
	f = np.matmul( Oinv, f ).astype(np.float32)
	# Test c version
	t2 = time.clock()-t1
	recon_c = np.zeros((nstar,nc*nc), dtype=np.float32)
	O2 = np.zeros([nstar,nstar],dtype=np.float64)
	O2inv = np.zeros([nstar,nstar],dtype=np.float64)
	fluxes = np.zeros( nstar, dtype = np.float32 )
	#lib(nstar, nc, cf.shape[0], dd, cf, recon_c)
	#lib2(imsz[0],imsz[1],nstar, nc, ix, iy, recon_c, O2)
	lib3(imsz[0],imsz[1],nstar, nc, cf.shape[0], back, ix, iy, dd, cf, recon_c, O2, O2inv, image, fluxes)
	Oinv3 = O2.copy()
	lib4(Oinv3,nstar)
	Oinv2 = np.linalg.inv( O2 ).astype(np.float64)
	fluxes = np.matmul( Oinv2, fluxes ).astype(np.float32)
	t3 = time.clock()
	print (t2-t1)*1000,(t3-t2)*1000
	return recon, recon_c, O, O2, f, fluxes, Oinv2, Oinv3


def image_model_eval_flux(x, y, back, imsz, nc, cf, regsize=None, margin=0, offsetx=0, offsety=0, weights=None, ref=None, lib=None, lib2=None):
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert cf.dtype == np.float32
        if ref is not None:
                assert ref.dtype == np.float32

        if weights is None:
                weights = np.ones(imsz, dtype=np.float32)
	if regsize is None:
		regsize = max(imsz[0], imsz[1])

        # sometimes phonions are outside image... what is best way to handle?
        goodsrc = (x >= 0) * (x <= imsz[0] - 1) * (y >= 0) * (y <= imsz[1] - 1)
        x = x.compress(goodsrc)
        y = y.compress(goodsrc)

        nstar = x.size
        rad = nc/2 # 12 for nc = 25

	nregy = imsz[1]/regsize + 1 # assumes imsz % regsize = 0?
	nregx = imsz[0]/regsize + 1

        ix = np.ceil(x).astype(np.int32)
        dx = ix - x
        iy = np.ceil(y).astype(np.int32)
        dy = iy - y
	dd = np.column_stack((np.ones(nstar, dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32)

	# *******************
	if lib2 is None:
		# postage stamp image for each star
		recon = np.dot(dd, cf).reshape((nstar,nc,nc))
		# overlap integral between all phonions in the map
		O = SourceSourceCovariance(imsz,x,y,ix,iy,recon,rad)
		# inverse of overlap integral
		Oinv = np.linalg.inv( O ).astype(np.float64)
		# determinant
		logdetO = np.log( np.linalg.det( O ) ).astype(np.float32)
		# flux estimates
		f = np.empty(nstar).astype(np.float32)
		for i in xrange(nstar):
			# ***** assumes no stars near edge of image near edge of (or off) image ******
			# subtracting out background: is this right?
			xmin = max( ix[i]-rad, 0 )
			ymin = max( iy[i]-rad, 0 )
			xmax = min( ix[i]+rad+1, imsz[0])
			ymax = min( iy[i]+rad+1, imsz[1])
			f[i] = np.sum( ( ref[ ymin:ymax,xmin:xmax]-back ) * recon[ i,ymin-iy[i]+rad:nc+ymax-iy[i]-rad-1, xmin-ix[i]+rad:nc+xmax-ix[i]-rad-1 ] )
		f = np.matmul( Oinv, f ).astype(np.float32)
	else:
		freg,logdetO = extract_source_fluxes(nstar,nc,imsz,cf,back,ix,iy,dd,ref,lib2)

	#********************
	
	# combine fluxes of phonions to remove with those estimated from image
	dd = dd * freg[:, None]

        if lib is None:
                image = back*np.ones((imsz[1]+2*rad+1,imsz[0]+2*rad+1), dtype=np.float32)
                recon = np.dot(dd, cf).reshape((nstar,nc,nc))
                for i in xrange(nstar):
			image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]
                image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

                if ref is not None:
                        diff = ref - image
			diff2 = np.zeros((nregy, nregx), dtype=np.float64)
			for i in xrange(nregy):
				y0 = max(i*regsize - offsety - margin, 0)
				y1 = min((i+1)*regsize - offsety + margin, imsz[1])
				for j in xrange(nregx):
					x0 = max(j*regsize - offsetx - margin, 0)
					x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
					subdiff = diff[y0:y1,x0:x1]
					diff2[i,j] = np.sum(subdiff*subdiff*weights[y0:y1,x0:x1])
        else:
                image = back*np.ones((imsz[1], imsz[0]), dtype=np.float32)
                recon = np.zeros((nstar,nc*nc), dtype=np.float32)
                reftemp = ref
                if ref is None:
                        reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
		diff2 = np.zeros((nregy, nregx), dtype=np.float64)
                lib(imsz[0], imsz[1], nstar, nc, cf.shape[0], dd, cf, recon, ix, iy, image, reftemp, weights, diff2, regsize, margin, offsetx, offsety)

        if ref is not None:
                return image, diff2, freg, logdetO
	else:
		return image



def image_model_eval_flux_inregions(x0, y0, f0, px, py, xregions, yregions, model, resid, back, imsz, nc, cf, regsize=None, margin=0, offsetx=0, offsety=0, weights=None, ref=None, lib=None, lib2=None, lib3=None):
        assert px.dtype == np.float32
        assert py.dtype == np.float32
	assert x0.dtype == np.float32
	assert y0.dtype == np.float32
	assert f0.dtype == np.float32
        assert cf.dtype == np.float32

        if ref is not None:
                assert ref.dtype == np.float32

        if weights is None:
                weights = np.ones(imsz, dtype=np.float32)
	if regsize is None:
		regsize = max(imsz[0], imsz[1])

	nregy = imsz[1]/regsize + 1 # assumes imsz % regsize = 0?
	nregx = imsz[0]/regsize + 1

	if False: # This shouldn't happen in current implementation of pcat & mocks
		# sometimes phonions are outside image... what is best way to handle?
		goodsrc = (x >= 0) * (x <= imsz[0] - 1) * (y >= 0) * (y <= imsz[1] - 1)
		x = x.compress(goodsrc)
		y = y.compress(goodsrc)

        rad = nc/2 # 12 for nc = 25

	# loop over regions and estimate fluxes & determinants in each
	nstar = x0.size

	time1 = time.clock()

	ix0 = np.ceil(x0).astype(np.int32)
        dx = ix0 - x0
	iy0 = np.ceil(y0).astype(np.int32)
        dy = iy0 - y0

	dd = np.column_stack((np.ones(nstar, dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f0[:, None]

	# construct model image of only those stars in regions (to subtract from image)
	ref_subtracted  = back*np.ones((imsz[1], imsz[0]), dtype=np.float32)
	recon = np.zeros((nstar,nc*nc), dtype=np.float32)
	lib3( imsz[0], imsz[1], nstar, nc, cf.shape[0], dd, cf, recon, ix0, iy0, ref_subtracted, ref,  model )

	# new sections
	pnstar = len(px)

	# determinant array 
	logdetO = np.zeros((nregy, nregx), dtype=np.float32)

	x_region_ids = np.arange(min(xregions) ,nregx, 2) # because I'm not explicitly passing the parity
	y_region_ids = np.arange(min(yregions) ,nregy, 2)

	nrx = len(x_region_ids)
	nry = len(y_region_ids)

	# now loop over regions and solve for flux within regions
	time2 = time.clock()

	# new fluxes
	fnew = np.zeros( pnstar, dtype = np.float32 )
	
	# sub-pixel shifts
	pix = np.ceil(px).astype(np.int32)
	pdx = pix - px
	piy = np.ceil(py).astype(np.int32)
	pdy = piy - py
	pdd = np.column_stack((np.ones(pnstar, dtype=np.float64), pdx, pdy, pdx*pdx, pdx*pdy, pdy*pdy, pdx*pdx*pdx, pdx*pdx*pdy, pdx*pdy*pdy, pdy*pdy*pdy)).astype(np.float32)
	
	for j in xrange(nry):
		phonions_region_y = yregions==y_region_ids[j]
		for i in xrange(nrx):
			phonions_region_x = xregions==x_region_ids[i]
			phonions_in_region =  np.flatnonzero( np.logical_and( phonions_region_x , phonions_region_y ) )
			nstarreg = phonions_in_region.size

			if nstarreg>0:
				ixreg = pix.take(phonions_in_region)
				iyreg = piy.take(phonions_in_region)
				dd = pdd.take(phonions_in_region,axis=0)

				# *******************
				if lib2 is None:
					xreg = px.take(phonions_in_region)
					yreg = py.take(phonions_in_region)
					# postage stamp image for each star
					recon = np.dot(dd, cf).reshape((nstarreg,nc,nc))
					# overlap integral between all phonions in the map
					O = SourceSourceCovariance(imsz,xreg,yreg,ixreg,iyreg,recon,rad)
					# inverse of overlap integral
					Oinv = np.linalg.inv( O ).astype(np.float64)
					# (log) determinant
					logdetO[y_region_ids[j],x_region_ids[i]] = ( np.log( linalg.det( 10. * O ) ) - nstarreg*np.log(10.) ).astype(np.float32)
					# flux estimates. Rescale matrix to prevent determinant calculation = 0
					freg = np.empty(nstarreg).astype(np.float32)
					for k in xrange(nstarreg):
						# ***** assumes no stars near edge of image near edge of (or off) image ******
						# subtracting out background: is this right?
						xmin = max( ixreg[k]-rad, 0 )
						ymin = max( iyreg[k]-rad, 0 )
						xmax = min( ixreg[k]+rad+1, imsz[0])
						ymax = min( iyreg[k]+rad+1, imsz[1])
						freg[k] = np.sum( ( ref_subtracted[ ymin:ymax,xmin:xmax]-back ) * recon[ k,ymin-iy[k]+rad:nc+ymax-iy[k]-rad-1, xmin-ixreg[k]+rad:nc+xmax-ixreg[k]-rad-1 ] )
					freg = np.matmul( Oinv, freg ).astype(np.float32)
				else:
					freg,ld = extract_source_fluxes(nstarreg,nc,imsz,cf,back,ixreg,iyreg,dd,ref_subtracted,lib2)
					logdetO[y_region_ids[j],x_region_ids[i]] = ld
				#********************
				# put the fluxes back into the master flux array
				fnew[phonions_in_region] = freg

	time3 = time.clock()

	# combine fluxes of phonions to remove with those estimated from image
	# we can now construct a differenced model using the old and new positions - everythng else remains the same.
	ftemp = np.concatenate((-f0,fnew))
	xtemp = np.concatenate((x0,px))
	ytemp = np.concatenate((y0,py))
	ntemp = xtemp.size

	ix = np.ceil(xtemp).astype(np.int32)
        dx = ix - xtemp
	iy = np.ceil(ytemp).astype(np.int32)
        dy = iy - ytemp
	dd = np.column_stack((np.ones(ntemp, dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * ftemp[:, None]

        if lib is None:
                image = back*np.ones((imsz[1]+2*rad+1,imsz[0]+2*rad+1), dtype=np.float32)
                recon = np.dot(dd, cf).reshape((ntemp,nc,nc))
                for i in xrange(ntemp):
			image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

                image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

                if ref is not None:
                        diff = ref - image
			diff2 = np.zeros((nregy, nregx), dtype=np.float64)
			for i in xrange(nregy):
				y0 = max(i*regsize - offsety - margin, 0)
				y1 = min((i+1)*regsize - offsety + margin, imsz[1])
				for j in xrange(nregx):
					x0 = max(j*regsize - offsetx - margin, 0)
					x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
					subdiff = diff[y0:y1,x0:x1]
					diff2[i,j] = np.sum(subdiff*subdiff*weights[y0:y1,x0:x1])
        else:
                image = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
                recon = np.zeros((ntemp,nc*nc), dtype=np.float32)
		diff2 = np.zeros((nregy, nregx), dtype=np.float64)
                lib(imsz[0], imsz[1], ntemp, nc, cf.shape[0], dd, cf, recon, ix, iy, image, resid, weights, diff2, regsize, margin, offsetx, offsety)

	time4 = time.clock()
	dt = np.array([time2-time1,time3-time2,time4-time3])

        if ref is not None:
                return image, diff2, fnew, logdetO, dt
	else:
		return image

def extract_source_fluxes(nstarreg,nc,imsz,cf,back,ixreg,iyreg,dd,ref,lib):
	# postage stamp image for each star
	recon = np.zeros((nstarreg,nc*nc), dtype=np.float32)
	# overlap integral between all phonions in the map
	O = np.zeros([nstarreg,nstarreg],dtype=np.float64)
	Oinv2 = np.zeros([nstarreg,nstarreg],dtype=np.float64)
	WORK = np.zeros([nstarreg,nstarreg],dtype=np.float64)
	IPIV = np.zeros(nstarreg, dtype=np.int32)
	# fluxes
	freg = np.zeros( nstarreg, dtype = np.float32 )
	fluxes = np.zeros( nstarreg, dtype = np.float32 )
	# get recon, O and f
	lib(imsz[0],imsz[1],nstarreg, nc, cf.shape[0], back, ixreg, iyreg, dd, cf, recon, O, Oinv2, WORK, IPIV, ref, fluxes, freg)
	# calc log determinant
	logdetreg = np.linalg.slogdet(O)[1] #  calculates log(determinant) [2nd param returned]	
	return freg, logdetreg.astype(np.float32)

def image_model_eval_determinants(x, y, f, back, imsz, nc, cf, regsize=None, margin=0, offsetx=0, offsety=0, weights=None, ref=None, lib=None, lib2=None):
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert f.dtype == np.float32
        assert cf.dtype == np.float32
        if ref is not None:
                assert ref.dtype == np.float32

        if weights is None:
                weights = np.ones(imsz, dtype=np.float32)
	if regsize is None:
		regsize = max(imsz[0], imsz[1])

	if False:
		# sometimes phonions are outside image... what is best way to handle?
		goodsrc = (x > 0) * (x < imsz[0] - 1) * (y > 0) * (y < imsz[1] - 1)
		x = x.compress(goodsrc)
		y = y.compress(goodsrc)
		f = f.compress(goodsrc)

        nstar = x.size
	assert nstar==y.size
        rad = nc/2 # 12 for nc = 25

	nregy = imsz[1]/regsize + 1 # assumes imsz % regsize = 0?
	nregx = imsz[0]/regsize + 1


	# new sections
	logdetO = np.zeros((nregy, nregx), dtype=np.float32)
	# regions of each phonion moved
	regionx = get_region(x, offsetx, regsize)
	regiony = get_region(y, offsety, regsize)
	x_region_ids = np.arange(nregx)
	y_region_ids = np.arange(nregy)

	#sub-pixel shifts
	ix = np.ceil(x).astype(np.int32)
	dx = ix - x
	iy = np.ceil(y).astype(np.int32)
	dy = iy - y

	dd = np.column_stack((np.ones(nstar, dtype=np.float64), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32)

	# just calculatting determinants, so don't actually need the image
	ref_temp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)

	# now loop over regions to get determinants
	for j in xrange(nregy):
		phonions_region_y = regiony==y_region_ids[j]
		for i in xrange(nregx):
			phonions_region_x = regionx==x_region_ids[i]
			phonions_in_region =   np.flatnonzero( np.logical_and( phonions_region_x , phonions_region_y ) )
			nstarreg = phonions_in_region.size

			if nstarreg>0:
				ixreg = ix.take(phonions_in_region)
				iyreg = iy.take(phonions_in_region)
				ddreg = dd.take(phonions_in_region,axis=0)

				# *******************
				if lib2 is None:
					# postage stamp image for each star
					recon = np.dot(ddreg, cf).reshape((nstarreg,nc,nc))
					# overlap integral between all phonions in the map
					O = SourceSourceCovariance(imsz,xreg,yreg,ixreg,iyreg,recon,rad)
					# inverse of overlap integral
					Oinv = np.linalg.inv( O ).astype(np.float64)
					# (log) determinant
					logdetO[y_region_ids[j],x_region_ids[i]] = ( np.log( linalg.det( 10. * O ) ) - nstarreg*np.log(10.) ).astype(np.float32)
				else:
					freg,ld = extract_source_fluxes(nstarreg,nc,imsz,cf,back,ixreg,iyreg,ddreg,ref_temp,lib2)
					logdetO[y_region_ids[j],x_region_ids[i]] = ld


        dd = dd * f[:, None]

        if lib is None:
                image = back*np.ones((imsz[1]+2*rad+1,imsz[0]+2*rad+1), dtype=np.float32)
                recon = np.dot(dd, cf).reshape((nstar,nc,nc))
                for i in xrange(nstar):
			image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

                image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

                if ref is not None:
                        diff = ref - image
			diff2 = np.zeros((nregy, nregx), dtype=np.float64)
			for i in xrange(nregy):
				y0 = max(i*regsize - offsety - margin, 0)
				y1 = min((i+1)*regsize - offsety + margin, imsz[1])
				for j in xrange(nregx):
					x0 = max(j*regsize - offsetx - margin, 0)
					x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
					subdiff = diff[y0:y1,x0:x1]
					diff2[i,j] = np.sum(subdiff*subdiff*weights[y0:y1,x0:x1])
        else:
                image = back*np.ones((imsz[1], imsz[0]), dtype=np.float32)
                recon = np.zeros((nstar,nc*nc), dtype=np.float32)
                reftemp = ref
                if ref is None:
                        reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
		diff2 = np.zeros((nregy, nregx), dtype=np.float64)
                lib(imsz[0], imsz[1], nstar, nc, cf.shape[0], dd, cf, recon, ix, iy, image, reftemp, weights, diff2, regsize, margin, offsetx, offsety)

        if ref is not None:
                return image, diff2, logdetO
	else:
		return image


def get_region(x, offsetx, regsize):
    return np.floor(x + offsetx).astype(np.int) / regsize

def SourceSourceCovariance(imsz,x,y,ix,iy,recon,rad,sep_thresh=7.):
	nsources = x.size
	C = np.zeros([nsources,nsources],dtype=np.float64)
	pssize = 2*rad+1
	for i in range(nsources):
		for j in range(i,nsources):
			#if ix[i]==ix[j] and iy[i]==iy[j]:
			#	C[i,j] = np.sum( recon[i,:,:]*recon[j,:,:] ).astype(np.float64)
			if (x[i]-x[j])**2 + (y[i]-y[j])**2  < 2.0*pssize*pssize:
				C[i,j] = overlap_integral2(imsz,ix[i],iy[i],ix[j],iy[j],recon[i,:,:],recon[j,:,:],rad,pssize)
			else:
				C[i,j] = 0.0
			C[j,i] = C[i,j]
	return C

def overlap_integral(ix1,iy1,ix2,iy2,recon1,recon2,rad):
	Xo = np.intersect1d( np.arange( ix1-rad,ix1+rad+1 ),np.arange( ix2-rad,ix2+rad+1 ) )
	Yo = np.intersect1d( np.arange( iy1-rad,iy1+rad+1 ),np.arange( iy2-rad,iy2+rad+1 ) )
	lenXo = len(Xo)
	lenYo = len(Yo)
	if lenXo*lenYo>0:   
		s1x = min(Xo)-ix1+rad
		s1y = min(Yo)-iy1+rad
		s2x = min(Xo)-ix2+rad
		s2y = min(Yo)-iy2+rad
		overlap = np.sum( recon1[s1y:s1y+lenYo,s1x:s1x+lenXo]*recon2[s2y:s2y+lenYo,s2x:s2x+lenXo] ).astype(np.float64)
		return overlap
	else:
		return 0.0

def overlap_integral2(imsz,ix1,iy1,ix2,iy2,recon1,recon2,rad,pssize):
	ixmin = max(max(ix1,ix2)-rad,0)
	ixmax = min(min(ix1,ix2)+rad+1,imsz[0])
	iymin = max(max(iy1,iy2)-rad,0)
	iymax = min(min(iy1,iy2)+rad+1,imsz[1])
	npix_overlap = (ixmax-ixmin)*(iymax-iymin)
	if npix_overlap>0:   
		s1x_low = ixmin-ix1+rad
		s1x_hi = ixmax-ix1+rad
		s1y_low = iymin-iy1+rad
		s1y_hi = iymax-iy1+rad
		s2x_low = ixmin-ix2+rad
		s2x_hi = ixmax-ix2+rad
		s2y_low = iymin-iy2+rad
		s2y_hi = iymax-iy2+rad
		assert s1x_low>=0 and s2x_low>=0 and s1y_low>=0 and s2y_low>=0
		assert s1x_hi<=pssize and s2x_hi<=pssize and s1y_hi<=pssize and s2y_hi<=pssize
		overlap = np.sum( recon1[s1y_low:s1y_hi,s1x_low:s1x_hi]*recon2[s2y_low:s2y_hi,s2x_low:s2x_hi] ).astype(np.float64)
		return overlap
	else:
		return 0.0
