import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

from sklearn import preprocessing

# A crude way to compute fold-change values adjusting for poorly expressed genes
# Inspired by DESeq2

def foldchange(comparison, rpk, tpm, lines, reps, alpha=1e-3):
    
    laba = ['{}_{}'.format(lines[comparison[0]], r) for r in reps ]
    labm = ['{}_{}'.format(lines[comparison[1]], r) for r in reps ]
    
    foo = rpk.loc[tpm.index, laba + labm].mean(axis=1) - rpk.loc[tpm.index, laba + labm].sem(axis=1)
    attenuate = preprocessing.minmax_scale(foo, (foo.min(), np.quantile(foo, 1-alpha)))
    #attenuate = foo.copy()
    attenuate[attenuate > 1] = 1

    a = tpm[laba].mean(axis=1)
    m = tpm[labm].mean(axis=1)
    
    am = a.div(m)
    ma = m.div(a)
    
    #print(attenuate)
    return  attenuate*am.where(am > ma, ma)
    
# Remove genes where at least one population reports all zero values
def mask_nonzeros(rpk, lines, reps, thr=0):
    #nonzero_mask = pd.Series(True, index=rpk.index)
    #for l in lines:
    #    nonzero_mask[ rpk[ ['{}_{}'.format(l,r) for r in reps] ].mean(axis=1) == 0 ] = False
    nonzero_mask = rpk.sum(axis=1) > thr
        
    return nonzero_mask

def count_normalization(read_count, rpk, lines, reps, normalize_flag = 'DESeq2'):
    # Remove genes where at least one population reports all zero values
    nonzero_mask = mask_nonzeros(rpk, lines,reps)
    if normalize_flag == 'TPM':
        tpm = rpk.div(rpk.sum(axis=0)/1e6, axis='columns')
        tpm = tpm.loc[nonzero_mask].loc[ tpm.loc[nonzero_mask].sum(axis=1) > tpm.shape[1] ]
    
    elif normalize_flag == 'DESeq2':
        # Normalize counts according to DESeq2 instead

        nonzerolog = np.log(read_count.loc[(read_count > 0).all(axis=1)])
        deseq2_scale = np.exp(nonzerolog.sub(nonzerolog.mean(axis=1), axis='index').median(axis=0))
        deseq2 = read_count.div(deseq2_scale, axis='columns')
        tpm = deseq2.loc[nonzero_mask].loc[ deseq2.loc[nonzero_mask].sum(axis=1) > deseq2.shape[1] ]
        
    else:
        tpm = rpk.loc[nonzero_mask]
        
    return tpm
        
        # Scaling factors are close to 1
        # Pearson r correlation = 0.93 between TPM and DESeq2 values
        #print(stats.pearsonr( deseq2.loc[tpm.index].values.ravel(), tpm.values.ravel() ))
    

def html2json(htmlfile):
    jstr = htmlfile.split('<script>')[1].split('</script>')[0].strip()
    for c in ['graph', 'colorscale', 'summary', 'summary_histogram']:
        jstr = jstr.replace('const ' + c + ' =', '"' + c + '" :')
    jstr = '{' + jstr[:-1].replace(';\n    ',', ') + '}'
    return json.loads(jstr)
    

def json2mapperG(jdict):    

    graph = {'nodes': {}, 'links': {}}
    for i in range(len(jdict['graph']['nodes'])):
        key = jdict['graph']['nodes'][i]['name']
        graph['nodes'][key] = jdict['graph']['nodes'][i]['tooltip']['custom_tooltips']

    links = pd.DataFrame(0, index=range(len(jdict['graph']['links'])), columns=['source','target','width'], dtype=int)
    for i in range(len(links)):
        for c in links.columns:
            links.loc[i,c] = jdict['graph']['links'][i][c]

    for i in links['source'].unique():
        key =  jdict['graph']['nodes'][i]['name']
        graph['links'][key] = []

    for i in range(len(links)):
        key =  jdict['graph']['nodes'][links.loc[i,'source']]['name']
        link = jdict['graph']['nodes'][links.loc[i,'target']]['name']
        graph['links'][key].append(link)
        
    return graph
    
def mapper2networkx(graph):
    nodes = graph["nodes"].keys()
    edges = [[start, end] for start, ends in graph["links"].items() for end in ends]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    nx.set_node_attributes(g, values=dict(graph["nodes"]), name = "membership")
    g.add_edges_from(edges)
    props = ['common', 'size', 'invsize', 'arcsize']
    eprops = dict(zip(props, [dict() for _ in range(len(props))]))
    for ekey in g.edges:
        eprops['common'][ekey] =  list(set(graph['nodes'][ekey[0]]) & set(graph['nodes'][ekey[1]]) )
        eprops['size'][ekey] =  len(eprops['common'][ekey])
        eprops['invsize'][ekey] =  1/len(eprops['common'][ekey])

    maxsize = max(eprops['size'].values())
    for ekey in g.edges:
        eprops['arcsize'][ekey] = maxsize - eprops['size'][ekey] + 1
    for prop in eprops:
        nx.set_edge_attributes(g, values=eprops[prop], name=prop)
    
    return g
    
    
