import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import glob
import re
import copy
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import networkx as nx
import kmapper as km
import utils
import matplotlib as mpl


fs = 13.5
# List the name of the lines, their repetitions
# and pairs to compare
reps = [1,2]
lines = ['BA24','BM24','IA24','IM24','ZA24','ZM24']
full_lines = list(itertools.chain(*[[ '{}_{}'.format(l,r) for r in reps ] for l in lines]))
comparisons = [(0,1),(2,3),(4,5),(0,2),(0,4),(1,3),(1,5)]

nlabels = ['min', 'mean', 'max', 'median', 'std', 'size', 'sqrtsize', 'cbrtsize']
weights = [None, 'size', 'invsize', 'arcsize']
titles = ['Avg. Shortest Path Length', 'Shortest Path from Purple', 'Avg * Shortest', 'Avg + Shortest']

p = re.compile('(.*)_B(.*)_Ov(.*)_dbscan(.*)')
zeroweight = 3
alpha = 0.001

def main():
    rsrc = os.pardir + os.sep + 'raw' + os.sep
    msrc = os.pardir + os.sep + 'outputs' + os.sep
    dst = os.pardir + os.sep + 'plots' + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # Read the count number file
    filename = rsrc + 'readCounts_Xa7_msu_v7.csv'
    data = pd.read_csv(filename).set_index('Geneid')
    rpk = data[full_lines].div(data['Length'], axis='index')
    print('Original dataframe dimensions:\t', rpk.shape)

    # Standardize expression levels
    # Remove genes where at least one population reports all zero values
    tpm = utils.count_normalization(data[full_lines], rpk, lines, reps, normalize_flag='DESeq2')
    print('After initial cleanup:\t', tpm.shape)

    # Get all the foldchange values
    # One per pair comparison
    fcs = pd.DataFrame(np.nan, index=tpm.index, columns = ['{} vs {}'.format(lines[comp[0]],lines[comp[1]]) for comp in comparisons])
    for comp in comparisons:
        key = '{} vs {}'.format(lines[comp[0]],lines[comp[1]])
        fcs[key] = utils.foldchange(comp, rpk, tpm, lines, reps, alpha=1e-3)

    fc = fcs.sum(axis=1)
    lfc = np.log10(fc)

    lens = lfc.values.copy()
    
    # Loop through all HTML mapper graphs
    
    for filename in glob.glob(msrc + '*.html'):
        
        bname = os.path.splitext(os.path.split(filename)[1])[0]
        filters, binno, overlap, dbscan = p.findall(bname)[0]
        
        filters = filters.replace('_',' ').replace('+',' & ').title().replace('Sd','SD')
        dbscan = dbscan.replace(',','.')
        overlap += '%'
        print(filters, binno, overlap, dbscan, sep='\t')
        mtitle = f'Filters: {filters}, Bins#: {binno}, Overlap: {overlap}, DBSCAN($\\varepsilon$): {dbscan}'
        
        fname = dst + bname
        
        with open(filename, encoding="utf-8") as f:
            jdict = utils.html2json(f.read())
        graph = utils.json2mapperG(jdict)

        metanode = pd.DataFrame(np.nan, index=graph['nodes'].keys(), columns=nlabels)
        for key in metanode.index:
            nn = lens[ graph['nodes'][key] ]
            for attr in nlabels[:5]:
                metanode.loc[key, attr] = getattr( np, attr)(nn)
            metanode.loc[key, 'size'] = len(nn)

        metanode['sqrtsize'] = np.sqrt(metanode['size'])
        metanode['cbrtsize'] = np.cbrt(metanode['size'])
        
        # Convert the HTML to a networkx object
        G = utils.mapper2networkx(graph)
        Gcpy = G.copy()

        nG = list(G.nodes)

        S = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)]

        nS = list(S[0].nodes)
        lowest = metanode.loc[ nS , 'median'].idxmin()

        columns = list(itertools.chain(*[[f'{m}_{str(w)}' for m in ['aspl','spath','ecc']] for w in weights]))
        gmetrics = pd.DataFrame(index=nS, columns=columns)
        for w in weights:
            gmetrics['ecc_'+str(w)] = pd.Series(nx.eccentricity(S[0], weight=w))[nS]
            gmetrics['spath_'+str(w)]= pd.Series(nx.shortest_path_length(S[0], source=lowest, weight=w))[nS]
            aspl = dict(nx.all_pairs_dijkstra_path_length(S[0], weight=w))
            aspl = pd.Series([ sum(aspl[key].values())/len(aspl) for key in aspl ], index=aspl.keys())
            gmetrics['aspl_'+str(w)] = aspl[nS]

        furthest = gmetrics['spath_None'].idxmax()

        temp_edge = []
        furthest = gmetrics['spath_invsize'].sort_values().index
        for ix in range(1, len(S)):
            temp_edge.append( (furthest[-3*ix], list(S[ix].nodes)[0]) )
            Gcpy.add_edge(temp_edge[-1][0], temp_edge[-1][1], size=zeroweight)

        pos = nx.spring_layout(Gcpy, seed=4, weight='size')
        bfspos = nx.bfs_layout(Gcpy, lowest, scale=len(S)-ix, center=(0,0), align='vertical')
        
        dfpos = pd.concat( (pd.DataFrame(pos, index=[0,1]), pd.DataFrame(bfspos, index=[2,3])) , axis='index')
        dfpos = dfpos.T.loc[nG]

        nx.set_edge_attributes(Gcpy, dict(zip(temp_edge, np.zeros(len(temp_edge)))), 'size')

        width = np.array([ foo[2] for foo in  Gcpy.edges(data='size')])
        drawkw = dict(G=Gcpy, node_size=0, width = 0.75*np.cbrt(width), 
                      edge_color=width, edge_cmap=mpl.colormaps['Greys'], edge_vmin=-7, edge_vmax=7,
                      with_labels=False, hide_ticks=True, font_size=6)
        scatterkw = dict(s=20*metanode.loc[nG, 'cbrtsize'], cmap='viridis', ec='k', zorder=4)
        stdvmax = metanode.loc[nG, 'std'].sort_values().iloc[-3]
        lvmax = metanode.loc[nS, 'mean'].sort_values().iloc[-2]

        posS = dict(zip(nS, [ pos[key] for key in nS ]))
        drawSkw = copy.deepcopy(drawkw)
        widthS = np.array([ foo[2] for foo in  S[0].edges(data='size')])
        drawSkw['G'] = S[0]
        drawSkw['width'] = 0.75*np.cbrt(widthS)
        drawSkw['edge_color'] = widthS
        scatterSkw = dict(x=dfpos.loc[nS, 0], y=dfpos.loc[nS, 1], s=20*metanode.loc[nS, 'cbrtsize'], cmap='viridis', ec='k', zorder=4)

        
        # Plot network for the first time

        fig, ax = plt.subplots(1,2, figsize=(12,5)); i = 0

        nx.draw_networkx(pos=pos, ax=ax[i], **drawkw )
        ax[i].scatter(dfpos[0], dfpos[1] , c=metanode.loc[nG, 'median'], vmin=0, vmax=lvmax, label='LFC', **scatterkw)
        ax[i].text(0.99, 0.01, 'Spring Layout', transform=ax[i].transAxes, fontsize=fs, ha='right', va='bottom')
        i += 1
        nx.draw_networkx(pos=bfspos, ax=ax[i], **drawkw )
        ax[i].scatter(dfpos[2], dfpos[3] , c=metanode.loc[nG, 'median'], vmin=0, vmax=lvmax, label='LFC', **scatterkw)
        ax[i].text(0.99, 0.01, 'BFS Layout', transform=ax[i].transAxes, fontsize=fs, ha='right', va='bottom')

        for a in ax.ravel():
            a.set_facecolor('snow')
        fig.suptitle(mtitle, fontsize=1.25*fs)
        fig.tight_layout();
        fig.savefig(fname + '_Gviz.png', format='png', bbox_inches='tight', dpi=150)
        plt.close()

        # # Match genes to "strandiness"

        fig, ax = plt.subplots(2, 4, figsize=(15,6.5), sharex=True, sharey=True)

        for a in ax.ravel():
            nx.draw_networkx(pos=posS, ax=a, **drawSkw )
            a.set_facecolor('snow')

        for i,w in enumerate([None, 'invsize']):
            ax[i,0].set_ylabel(str(w), fontsize=fs)
            ax[i,0].scatter(c=gmetrics.loc[nS, 'aspl_'+str(w)], **scatterSkw)
            ax[i,1].scatter(c=gmetrics.loc[nS, 'spath_'+str(w)], **scatterSkw)
            ax[i,2].scatter(c=gmetrics.loc[nS, 'aspl_'+str(w)] * gmetrics.loc[nS, 'spath_'+str(w)], **scatterSkw)
            ax[i,3].scatter(c=gmetrics.loc[nS, 'aspl_'+str(w)] + gmetrics.loc[nS, 'spath_'+str(w)], **scatterSkw)

        for i in range(len(titles)):
            ax[1,i].set_xlabel(titles[i], fontsize=fs)
        fig.suptitle(mtitle, fontsize=1.25*fs)
        fig.tight_layout()
        fig.savefig(fname + '_strandiness.png', format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Correlate LFC to strandiness

        LFC = metanode.loc[nS, 'mean']

        w = 'invsize'; strand0 = gmetrics['aspl_'+str(w)] + gmetrics['spath_'+str(w)]
        w = None; strand1 = gmetrics['aspl_'+str(w)] * gmetrics['spath_'+str(w)]
        fig, ax = plt.subplots(1,2, figsize=(10,4), sharey=True); i = 0
        for w,strand in zip(['unweighted','weighted'], [strand1,strand0]):
            
            spearman = stats.spearmanr(strand, LFC)
            pearson = stats.pearsonr(strand, LFC)
            label = 'r = {:.2f}\ns = {:.2f}'.format(pearson.statistic,spearman.statistic)
            ax[i].axhline(1, c='lightgray', ls='dashed', zorder=1)
            ax[i].scatter(strand, LFC, s=scatterSkw['s'], c=metanode.loc[nS,'std'], marker='h', ec='k', vmax=stdvmax, label=label, zorder=4 )
            ax[i].set_xlabel('Strandiness (' + str(w) + ')', fontsize=fs)
            ax[i].tick_params(labelsize = fs)
            ax[i].set_facecolor('snow')
            ax[i].legend(loc='lower right', fontsize=fs)
            i += 1

        fig.suptitle(mtitle, fontsize=1.25*fs)
        ax[0].set_ylabel('Average LFC', fontsize=fs)
        fig.tight_layout()
        fig.savefig(fname + '_LFC_corr.png', format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # # Different clusters highlight different sample pair comparisons (?)

        metanodes = dict()
        for comp in comparisons:
            dkey = '{} vs {}'.format(lines[comp[0]],lines[comp[1]])
            LFC = np.log10(fcs[dkey].copy())
            meta = pd.DataFrame(np.nan, index=graph['nodes'].keys(), columns=nlabels[:5])
            for key in meta.index:
                nn = LFC.iloc[ graph['nodes'][key] ]
                for attr in meta.columns:
                    meta.loc[key, attr] = getattr( np, attr)(nn)
                
            mini, maxi = np.quantile( LFC , [alpha, 1-alpha])
            LFC[LFC < mini] = mini
            LFC[LFC > maxi] = maxi
            
            metanodes[dkey] = meta

        metavmax = dict()
        for comp in comparisons:
            dkey = '{} vs {}'.format(lines[comp[0]],lines[comp[1]])
            metavmax[dkey] = metanodes[dkey]['mean'].sort_values().iloc[-2]


        # Individual comparisons


        fig, ax = plt.subplots(2, 4, figsize=(15,7))
        ax = ax.ravel(); i = 0

        for a in ax:
            a.set_facecolor('snow')
            nx.draw_networkx(pos=pos, ax=a, **drawkw )
        ax[i].scatter(dfpos[0], dfpos[1] , c=metanode.loc[nG, 'mean'], vmin=0, vmax=lvmax, **scatterkw)
        for comp in comparisons:
            i += 1
            key = '{} vs {}'.format(lines[comp[0]],lines[comp[1]])
            ax[i].scatter(dfpos[0], dfpos[1] , c=metanodes[key].loc[nG, 'mean'], vmin=0, vmax=metavmax[key], **scatterkw)
            ax[i].set_xlabel(key, fontsize=fs)

        fig.suptitle(mtitle, fontsize=1.25*fs)
        fig.tight_layout();
        fig.savefig(fname + '_individual.png', format='png', bbox_inches='tight', dpi=150)
        plt.close()

    return 0

if __name__ == '__main__':
    main()
