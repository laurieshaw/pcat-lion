import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double, c_float
#import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
'''
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''

def calc_flux_prior(nstar, truealpha, trueminf):
    # evaluate the width of the flux prior numerically (expected maximum of nstars given the prior)
    dlnf = 0.0001
    xint = 10**np.arange(0,21,dlnf)
    gx = trueminf*(truealpha-1)*nstar*(1-xint**(1-truealpha))**(nstar-1)*xint**(2-truealpha)
    flux_prior = np.float32( 1.00*np.sum( gx*dlnf )/np.log10( np.exp(1.0) ) ) # need to explore this further.
    print "Flat flux prior width = %1.3e" % (flux_prior)
    return flux_prior

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

def not_idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero( np.logical_not(np.logical_and(match_x, match_y)) )

def init_phonion_distribution(n,imsz,min_sep=0.5,perturb=True, img_buf=1):
    rhol = np.sqrt( n/float(imsz[0]*imsz[1]) )
    nx = int( np.ceil( imsz[0]*rhol ) )
    ny = int( np.ceil( imsz[1]*rhol ) )
    lx = imsz[0]/float(nx)
    ly = imsz[1]/float(ny)
    x = np.tile( np.arange(nx)*lx + lx/2., (ny,1) ).flatten().astype(np.float32)
    y = np.tile( np.arange(ny)*ly + ly/2., (nx,1) ).flatten('F').astype(np.float32)
    ind = np.random.choice(int(nx*ny), size=n, replace=False)
    x = x.take(ind)
    y = y.take(ind)
    s = ( min(lx,ly) - min_sep - 0.01).astype(np.float32)
    assert x.size==n and y.size==n
    if perturb:
        dx = np.random.uniform(low=-0.5,high=0.5,size=n).astype(np.float32)*s
        dy = np.random.uniform(low=-0.5,high=0.5,size=n).astype(np.float32)*s
        x = x + dx 
        y = y + dy 
        img_xmin = 0+img_buf
        img_xmax = imsz[0]-img_buf-1
        img_ymin = 0+img_buf
        img_ymax = imsz[1]-img_buf-1
        # bounce off of edges of image
        mask = x < img_xmin
        x[mask] += 2*(img_xmin-x[mask])
        mask = x > img_xmax
        x[mask] -= 2*(x[mask]-img_xmax)
        mask = y < img_ymin
        y[mask] += 2*(img_ymin-y[mask])
        mask = y > img_ymax
        y[mask] -= 2*(y[mask]-img_ymax)
    return x,y

def sample_positions_with_constraint(xfixed,yfixed,x0,y0,dpos_rms,imsz,img_buf,min_sep=0.5,dpos_growth=1.1,nstore=10000,lib=None):
     # prevent phonions from moving too close to one another.
    count = 0
    nmove = x0.size
    dx = np.random.normal(size=nmove).astype(np.float32)*dpos_rms
    dy = np.random.normal(size=nmove).astype(np.float32)*dpos_rms
    mx = x0+dx
    my = y0+dy
    img_xmin = 0+img_buf
    img_xmax = imsz[0]-img_buf-1
    img_ymin = 0+img_buf
    img_ymax = imsz[1]-img_buf-1
    # bounce off of edges of image
    mask = mx < img_xmin
    mx[mask] += 2*(img_xmin-mx[mask])
    mask = mx > img_xmax
    mx[mask] -= 2*(mx[mask]-img_xmax)
    mask = my < img_ymin
    my[mask] += 2*(img_ymin-my[mask])
    mask = my > img_ymax
    my[mask] -= 2*(my[mask]-img_ymax)
    allx = np.concatenate((mx,xfixed)).astype(np.float32)
    ally = np.concatenate((my,yfixed)).astype(np.float32)
    nall = allx.size
    min_sep2 = min_sep*min_sep
    iret = np.array([0],dtype=np.int32)
    success,i = position_sample_accept(allx,ally,mx,my,min_sep2,istart=0, lib=lib, iret=iret)
    dstore = np.random.normal(size=nstore).astype(np.float32)
    j = 0
    while ( not success ):
        dpos_rms[i] *= dpos_growth # grow if can't find solution
        #print i,x0[i]+dx[i],y0[i]+dy[i]
        #dx[i] = np.random.normal(size=1).astype(np.float32)*dpos_rms[i]
        #dy[i] = np.random.normal(size=1).astype(np.float32)*dpos_rms[i]
        mx[i] = x0[i] + dstore[j]*dpos_rms[i]
        my[i] = y0[i] + dstore[j+1]*dpos_rms[i]
        if mx[i] < img_xmin: mx[i] += 2*(img_xmin-mx[i])
        if my[i] < img_ymin: my[i] += 2*(img_ymin-my[i])
        if mx[i] > img_xmax: mx[i] -= 2*(mx[i]-img_xmax)
        if my[i] > img_ymax: my[i] -= 2*(my[i]-img_ymax)
        allx[i] = mx[i]
        ally[i] = my[i]
        success,i = position_sample_accept(allx,ally,mx,my,min_sep2,istart=i, lib=lib, iret=iret)
        j += 2
    return mx-x0,my-y0,dpos_rms 

def position_sample_accept(x,y,xs,ys,min_sep2,istart=0,lib=None,iret=np.array([0],dtype=np.int32)):
    nmove = xs.size
    nall = x.size
    if lib is None:
        for i in xrange(istart,nmove):
            r2 = (x-xs[i])**2 + (y-ys[i])**2
            r2[i] = 1e6 # reset distance of phonion from itself
            if np.min(r2)<min_sep2:
                return False,i
        return True,-1
    else:
        lib(x,y,nmove,nall,istart,iret,min_sep2)
        i = iret[0]
        return i==-1,i

def position_sample_accept2(x,y,xs,ys,mask,min_sep2,istart=0):
    dx = xs[:,None] - x[None,:]
    dy = ys[:,None] - y[None,:]
    dr = dx*dx + dy*dy + mask # mask ignores diagonal
    #mask = np.zeros_like(x)
    #mask[:5,:5] = np.eye(5)*1000
    rmins =  np.min( dr, axis=1)
    i = np.argmin( rmins )
    return rmins[i]>=min_sep2,i


def position_birth_accept(x,y,bx,by,min_sep=0.5):
    for bbx, bby in zip(bx,by):
        r2 = np.min( (x-bbx)**2 + (y-bby)**2 )
        if r2<min_sep*min_sep:
            return False 
    return True

def sample_kicks(nms,kickrange,min_sep):
    dx = (np.random.normal(size=nms)*kickrange).astype(np.float32)
    dy = (np.random.normal(size=nms)*kickrange).astype(np.float32)

    # force kicks to be at least a certain distance
    kickmask =  dx**2 + dy**2 < min_sep*min_sep
    while np.sum( kickmask ) > 0:
        dx[kickmask] =  (np.random.normal(size=np.sum(kickmask))*kickrange).astype(np.float32)
        dy[kickmask] =  (np.random.normal(size=np.sum(kickmask))*kickrange).astype(np.float32)
        kickmask[kickmask] = dx[kickmask]**2 + dy[kickmask]**2 < min_sep*min_sep
    return dx,dy

def selector( successi, successj, i, j):
    success = False
    if not successi and not successj:
        k = min(i,j)
    elif not successi:
        k = i
    elif not successj:
        k = j
    else:
        success = True
        k = -1
    return success, k

def sample_kicks_with_constraint(xfixed,yfixed,xm,ym,nms,kickrange,min_sep=0.5):
    # control split moves to prevent phonions from moving too close
    dx,dy = sample_kicks(nms,kickrange,min_sep)

    # new positions for phonions that move
    frac = 0.5 # hard code this for now. (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
    pxm = xm + ((1-frac)*dx)
    pym = ym + ((1-frac)*dy)
    bx = xm - frac*dx
    by = ym - frac*dy 

    successi,i = position_sample_accept(np.concatenate((pxm,xfixed)),np.concatenate((pym,yfixed)),pxm,pym,min_sep=min_sep,istart=0)
    successj,j = position_sample_accept(np.concatenate((bx,xfixed)),np.concatenate((by,yfixed)),bx,by,min_sep=min_sep,istart=0)
    success, k = selector( successi, successj, i, j)
    while ( not success ):
        dx[k], dy[k] =  sample_kicks(1,kickrange,min_sep)
        pxm[k] = xm[k] + ((1-frac)*dx[k])
        pym[k] = ym[k] + ((1-frac)*dy[k])
        bx[k] = xm[k] - frac*dx[k]
        by[k] = ym[k] - frac*dy[k]
        successi,i = position_sample_accept(np.concatenate((pxm,xfixed)),np.concatenate((pym,yfixed)),pxm,pym,min_sep=min_sep,istart=k)
        successj,j = position_sample_accept(np.concatenate((bx,xfixed)),np.concatenate((by,yfixed)),bx,by,min_sep=min_sep,istart=k)
        success, k = selector( successi, successj, i, j)
        #count += 1
    #print count
    return pxm,pym,bx,by


