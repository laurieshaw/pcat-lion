import numpy as np
import sys
from image_eval import image_model_eval, psf_poly_fit
import matplotlib.pyplot as plt


def load_catalogues(catfile,catname,burnin=0):
    chain_name = 'chain_' + catfile + '.npz'
    a = np.load('chain_' + catfile + '.npz')
    nsamples = len(a['n'])
    nloops = 1000
    cn = a['n']
    cf = a['f']
    cx = a['x']
    cy = a['y']
    print a.files
    if 'chi2' in a.files:
        cchi2 = a['chi2']
    else:
        cchi2 = np.zeros( cn.size ) * np.nan
    cats = []
    for i in np.arange(burnin,nsamples):
       cats.append( catalogue( cn[i], cx[i], cy[i], cf[i], cchi2[i], catname ) )
    return cats

def match_catalogues( catalogues, truthname, matchrad=0.75, matchlogflux=0.5 ):
    # get truth data
    truth = np.loadtxt('Data/'+truthname+'_tru.txt')
    truth = truth[ truth[:,2].argsort(), : ] # sort by flux
    truth = truth[::-1] # largest->smallest
    truex = np.array( truth[:,0] )
    truey = np.array( truth[:,1] )
    truef = np.array( truth[:,2] )
    for cat in catalogues:
        cat.matchTruth( truex, truey, truef, matchrad=matchrad, matchlogflux=matchlogflux )
    return catalogues

def make_plots( catalogues ):
    completeness_purity( catalogues )
    flux_reconstruction( catalogues )
    plt.show()

def make_image( catalogues, ax ):
    smax = 2000
    smin = 10
    sizefac = 10.*136
    allx = np.array([],dtype=float)
    ally = np.array([],dtype=float)
    allf = np.array([],dtype=float)
    for c in catalogues:
        allx = np.concatenate((allx,c.x))
        ally = np.concatenate((ally,c.y))
        allf = np.concatenate((allf,c.f))
    print np.shape(allf)
    size = np.abs( allf )/sizefac*20
    size[size>smax] = smax
    size[size<smin] = smin
    posmask = allf>0
    negmask = allf<0
    ax.scatter( allx[posmask],ally[posmask],marker='+',s=size[posmask],color='r',alpha=0.05)
    ax.scatter( allx[negmask],ally[negmask],marker='+',s=size[negmask],color='r',alpha=0.05)
    # now add truth
    truex = catalogues[0].truex
    truey = catalogues[0].truey
    truef = catalogues[0].truef
    truesize = truef/sizefac*20
    truesize[truesize>smax] = smax
    truesize[truesize<smin] = smin
    ax.scatter(truex, truey, marker='o', s=truesize, edgecolor='b',facecolor='none',linewidth=1.5)
    
def make_catalogue_images( cat_array ):
    ncats = len(cat_array)
    if ncats==1:
        fig,ax = plt.subplots()
        make_image( cat_array[0], ax )
    elif ncats<=3:
        fig,axes = plt.subplots(1,ncats)
        [make_image( c, ax ) for c,ax in zip(cat_array, axes )]
    elif ncats>=4:
        fig,axes = plt.subplots((2,np.ceil(ncats/2)))
        [make_image( c, ax ) for c,ax in zip(cat_array, axes[:ncats] )]
    plt.show()


def get_catalogues(catfiles,catnames,truthname,matchrad=0.75, matchlogflux=0.5 ):
    ncats = len(catfiles)
    assert len(catfiles)==len(catnames)
    cats = []
    for catfile,catname in zip(catfiles,catnames):
        cat = load_catalogues(catfile,catname)
        cat = match_catalogues( cat, truthname, matchrad=matchrad, matchlogflux=matchlogflux)
        cats.append(cat)
    return tuple(cats)

def compare_chains_plots( cat1, cat2, save=False ):
    fvals, completeness1, purity1 =  completeness_purity( cat1, dlogf = 0.2, ploton = False )
    fvals, completeness2, purity2 =  completeness_purity( cat2, dlogf = 0.2, ploton = False )
    fig,ax = plt.subplots()
    ax.semilogx( fvals, completeness1, 'r',label='Completeness'+'/'+cat1[0].name )
    ax.semilogx( fvals, completeness2, 'r--',label='Completeness'+'/'+cat2[0].name )
    ax.semilogx( fvals, purity1, 'b' ,label='FDR'+'/'+cat1[0].name)
    ax.semilogx( fvals, purity2, 'b--',label='FDR'+'/'+cat2[0].name)
    ax.legend(loc='best',framealpha=0.2,frameon=False,fontsize=10)
    ax.set_xlabel('Flux')
    ax.set_ylabel('Completeness/FDR')
    ax.yaxis.grid()
    ax.xaxis.grid()
    if save:
        fname = "/n/home03/lshaw/PCat/pcat-lion/plots/" + cat1[0].name + "_" + cat2[0].name + "_compFDR.png"
        fig.savefig(fname,dpi=400,bbox_inches='tight',pad_inches=0.1)
    flux_reconstruction( cat1 ,save=save)
    flux_reconstruction( cat2 ,save=save)
    position_error( cat1 ,save=save)
    position_error( cat2 ,save=save)
    plot_nsources_chi2( cat1, cat2, plot_chi2=True, save=save )
    plt.show()

def plot_nsources_chi2( cat1, cat2, plot_chi2=False, save=False ):
    nsource1 = [c.n for c in cat1]
    nsource2 = [c.n for c in cat2]
    if plot_chi2:
        fig2,ax2 = plt.subplots(2,1)
        ax2[0].plot( nsource1, 'b' ,label=cat1[0].name)
        ax2[0].plot( nsource2, 'r',label=cat2[0].name)
        chi1 = [c.chi2 for c in cat1]
        chi2 = [c.chi2 for c in cat2]
        ax2[1].plot( chi1, 'b' )
        ax2[1].plot( chi2, 'r' )
        ax2[1].set_xlabel('Sample')
        ax2[1].set_ylabel('Chi2')
        ax2[0].set_ylabel('n sources')
        ax2[0].legend(loc='best',framealpha=0.2,frameon=False,fontsize=10)
    else:
        fig2,ax2 = plt.subplots()
        ax2.plot( nsource1, 'b' )
        ax2.plot( nsource2, 'r' )
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('n sources')
        ax2.legend(loc='best',framealpha=0.2,frameon=False,fontsize=10)
    if save:
        fname = "/n/home03/lshaw/PCat/pcat-lion/plots/" + cat1[0].name + "_" + cat2[0].name + "_nsource.png"
        fig2.savefig(fname,dpi=400,bbox_inches='tight',pad_inches=0.1)
    return

def completeness_purity( catalogues, dlogf = 0.2, ploton = False ):
    fbins = 10**np.arange(2,6, dlogf )
    ncats = len(catalogues)
    comp = np.zeros( shape = ( ncats,  fbins.size-1) )
    purity = np.zeros( shape= ( ncats, fbins.size-1 ) )
    for i in np.arange( fbins.size - 1 ):
        for j in np.arange( ncats ):
            comp[j,i] = catalogues[j].completeness_in_fluxbin( fbins[i], fbins[i+1] )[1]
            purity[j,i] = catalogues[j].FDR_in_fluxbin( fbins[i], fbins[i+1] )[1]
    completeness = np.mean ( comp, axis = 0 )
    purity = np.mean( purity, axis = 0 )
    fvals = fbins[:-1] + dlogf/0.2
    if ploton:
        fig,ax = plt.subplots()
        ax.semilogx( fbins[:-1] + dlogf/0.2, np.mean ( comp, axis = 0 ), 'r' )
        ax.semilogx( fbins[:-1] + dlogf/0.2, np.mean ( purity, axis = 0 ), 'g' )
    return fvals, completeness, purity

def flux_reconstruction( catalogues, dlogf = 0.5, save=False ):
    truefluxes = catalogues[0].truef
    fmin = 2.5 #np.floor( np.min( np.log10( truefluxes) *2. ) ) / 2.
    fmax = np.floor( np.max( np.log10( truefluxes) *2. ) ) / 2.
    fbins = 10**np.arange(fmin, fmax, dlogf )
    ntrue = truefluxes.size
    ncats = len(catalogues)
    catfluxes = np.empty( shape = (ncats, ntrue ) )
    for i in xrange(ncats):
        catfluxes[i,:] = catalogues[i].matchf
    meanflux = np.nanmean( catfluxes, axis=0 )
    x = np.tile( truefluxes, (ncats, 1) ).flatten()
    catfluxes = catfluxes.flatten()
    fig1,ax1 = plt.subplots(2,1)
    ax1[0].loglog( x, catfluxes, 'k.', alpha= 0.05, label='cat flux' )
    ax1[0].plot( [10,10**6],[10,10**6],'r--')
    ax1[0].plot( truefluxes, meanflux, 'bd', label = 'mean flux (all cats')
    ax1[0].set_xlim( 100, 10**6 )
    ax1[0].set_ylim( 100, 10**6 )
    ferrdists = []
    for i in np.arange( fbins.size - 1 ):
        fcut = (x >= fbins[i]) & (x < fbins[i+1])
        xcut = x.compress(fcut)
        catcut = catfluxes.compress(fcut)
        ferr = np.log10( catcut ) - np.log10( xcut )
        ferr = ferr[~np.isnan(ferr)]
        ferrdists.append(ferr)
    print ferrdists
    ax1[1].violinplot(ferrdists, np.log10(fbins[:-1])+dlogf/2., points=60, widths=0.45, showmeans=False, showextrema=False, showmedians=True, bw_method=0.5)
    ax1[0].set_xlabel('True Flux')
    ax1[0].set_ylabel('Measured Flux')
    ax1[1].set_xlabel('Flux bin (log10)')
    ax1[1].set_ylabel('Flux error dist')
    ax1[0].legend(loc='best',framealpha=0.2,frameon=False,fontsize=10,numpoints=1)
    ax1[1].yaxis.grid()
    if save:
        fname = "/n/home03/lshaw/PCat/pcat-lion/plots/" + catalogues[0].name + "_flux.png"
        fig1.savefig(fname,dpi=400,bbox_inches='tight',pad_inches=0.1)

def position_error( catalogues,  dlogf = 0.5, matchrad=0.75, save=False ):
    truefluxes = catalogues[0].truef
    truex = catalogues[0].truex
    truey = catalogues[0].truey
    fmin = 2.5 #np.floor( np.min( np.log10( truefluxes) *2. ) ) / 2.
    fmax = np.floor( np.max( np.log10( truefluxes) *2. ) ) / 2.
    fbins = 10**np.arange(fmin, fmax, dlogf )
    ntrue = truefluxes.size
    ncats = len(catalogues)
    cat_pos_error = np.empty( shape = (ncats, ntrue ) )
    for i in xrange(ncats):
        cat_pos_error[i,:] =  np.sqrt( ( catalogues[i].matchx - truex )**2 +  ( catalogues[i].matchy - truey )**2 )
    meanerror = np.nanmean( cat_pos_error, axis=0 )
    x = np.tile( truefluxes, (ncats, 1) ).flatten()
    cat_pos_error = cat_pos_error.flatten()
    fig1,ax1 = plt.subplots(2,1)
    ax1[0].loglog( x, cat_pos_error, 'k.', alpha= 0.05, label='catalogue pos error' )
    ax1[0].plot( truefluxes, meanerror, 'bd', label = 'mean pos error (all cats)' )
    ax1[0].plot([100,10**6],[matchrad,matchrad],'k:')
    ax1[0].set_xlim( 100, 10**6 )
    ax1[0].set_ylim( 0.001, 1 )
    perrdists = []
    for i in np.arange( fbins.size - 1 ):
        fcut = (x >= fbins[i]) & (x < fbins[i+1])
        xcut = x.compress(fcut)
        catcut = cat_pos_error.compress(fcut)
        perr = catcut[~np.isnan(catcut)]
        perrdists.append(perr)
    ax1[1].plot([fmin,fmax-dlogf],[matchrad,matchrad],'k:')
    ax1[1].violinplot(perrdists, np.log10(fbins[:-1])+dlogf/2., points=60, widths=0.45, showmeans=False, showextrema=False, showmedians=True, bw_method=0.5)
    ax1[0].set_xlabel('True Flux')
    ax1[0].set_ylabel('Position error (pixels)')
    ax1[1].set_xlabel('Flux bin (log10)')
    ax1[1].set_ylabel('Position error dist (pixels)')
    ax1[0].legend(loc='best',framealpha=0.2,frameon=False,fontsize=10,numpoints=1)
    ax1[1].yaxis.grid()
    if save:
        fname = "/n/home03/lshaw/PCat/pcat-lion/plots/" + catalogues[0].name + "_position.png"
        fig1.savefig(fname,dpi=400,bbox_inches='tight',pad_inches=0.1)

class catalogue(object):
    def __init__(self,n,x,y,f,chi2,name):
        self.n = n
        self.x = x[:n]
        self.y = y[:n]
        self.f = f[:n]
        self.chi2 = chi2
        self.name = name
        self.sort_by_flux()

    def sort_by_flux(self):
        arr = np.column_stack( (self.f,self.x,self.y) )
        arr = arr[ arr[:,0].argsort(), : ]
        self.f = arr[:,0]
        self.x = arr[:,1]
        self.y = arr[:,2]

    def matchTruth(self,truex,truey,truef,matchrad=0.75,matchlogflux=0.5, verbose=False):
        # go through each source in true catalogue and match the closest source within a given flux range
        ntrue = truex.size
        matchflux = 10**matchlogflux
        self.truex = truex
        self.truey = truey
        self.truef = truef
        self.matched_cat = np.zeros_like(self.f)>0 # all false initially
        self.matched_true = np.zeros(ntrue)>0
        self.matchx = np.nan * np.zeros( ntrue )
        self.matchy = np.nan * np.zeros( ntrue )
        self.matchf = np.nan * np.zeros( ntrue )
        for i in np.arange(ntrue):
            tx = truex[i]
            ty = truey[i]
            tf = truef[i]
            fluxratio = self.f/tf
            idx_pos = np.flatnonzero(np.logical_and( fluxratio<matchflux, fluxratio>1./matchflux, np.logical_not( self.matched_cat ) ) )
            #print idx_pos.size
            if idx_pos.size > 0:
                xpos = self.x.take(idx_pos)
                ypos = self.y.take(idx_pos)
                fpos = self.f.take(idx_pos)
                r2 = (xpos-tx)**2 + (ypos-ty)**2
                #print i, np.min(r2), r2.argmin()
                if np.min(r2) < matchrad*matchrad:
                    #print i, np.min(r2), r2.argmin()
                    self.matched_true[i] = True
                    self.matched_cat[idx_pos[r2.argmin()]] = True
                    self.matchx[i] = xpos[r2.argmin()]
                    self.matchy[i] = ypos[r2.argmin()]
                    self.matchf[i] = fpos[r2.argmin()]
        if verbose:
            print "pct matched cat = %1.2f, pct matched truth = %1.2f" % (np.sum(self.matched_cat)/float(self.n), np.sum(self.matched_true)/float(ntrue))
        
    def completeness_in_fluxbin(self, fmin, fmax):
        cut = ( self.truef>=fmin ) & ( self.truef < fmax ) 
        nmatched = np.sum( self.matched_true[cut] )
        frac_matched = nmatched / float ( np.sum( cut ) )
        return np.sum(cut) , frac_matched

    def FDR_in_fluxbin( self, fmin, fmax):
        cut = ( self.f>=fmin ) & ( self.f < fmax ) 
        nmatched = np.sum( self.matched_cat[cut] )
        frac_matched = 1.0 - nmatched / float ( np.sum( cut ) )
        return np.sum(cut) , frac_matched
