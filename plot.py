import branca
import folium

import arrow
import random
#import torchutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import NullFormatter
import matplotlib.collections as mcoll
from matplotlib.backends.backend_pdf import PdfPages

import networkx as nx
from community import community_louvain

def linechart_for_total_number(obs, fitted, dates, modelname, time_mean = True, figsize = (15, 10)):
    fig = plt.figure(figsize = figsize)
    ax = plt.subplot(111)
    x = np.arange(obs.shape[1])
    ground = np.zeros(obs.shape[1])
    
    ax.fill_between(x, obs.sum(0), ground, where=obs.sum(0) >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Real")
    line1, = ax.plot(range(obs.shape[1]), obs.sum(0), color = '#AE262A', alpha=.6, linestyle="--")
    line2, = ax.plot(range(fitted.shape[1]), fitted.sum(0), alpha=1, linestyle="-")
    ax.legend((line1, line2), ('Observfation', 'Fitted lambda'), fontsize = 10)
    if time_mean:
        ax.set_xticks(np.arange(obs.shape[1]))
        ax.set_xticklabels(dates, rotation=80)
    else:
        ax.set_xticks(np.arange(1, obs.shape[1], 5))
        ax.set_xticklabels(dates[np.arange(1, obs.shape[1], 5)], rotation=80)
    ax.set_xlabel("Dates", size = 10)
    ax.set_ylabel("Number of cases", size = 10)
    ax.set_title("Observation and Fitted", size = 15)
    plt.show()
    
    pdf = PdfPages("%s_linechart_for_total_number.pdf" % modelname)
    pdf.savefig(fig)
    pdf.close()
    

def linechart_for_community(obs, fitted, dates, modelname, time_mean = True, figsize = (16, 16)):
    
    x = np.arange(obs.shape[1])
    ground = np.zeros(obs.shape[1])

    fig = plt.figure(figsize = figsize)
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    for i in range(22):
        ax = plt.subplot(6, 4, i+1)
        fill1 = ax.fill_between(x, obs[i, ], ground, where=obs[i, ] >= ground, facecolor='green', alpha=0.2, interpolate=True, label="Real")
        line1, = ax.plot(x, obs[i, ], linewidth=1, color="green", alpha=1, label="Obsv")
        line2, = ax.plot(x, fitted[i, ], linewidth=2, color="black", alpha=1, linestyle="--", label="Fitted")
        if time_mean: intv = 5
        else: intv = 30
        ax.set_xticks(np.arange(1, obs.shape[1], intv))
        ax.set_xticklabels(dates[np.arange(1, obs.shape[1], intv)], rotation=80)
        ax.set_title("Obsv and Fitted, community" + str(i))

#    plt.xticks(np.arange(len(dates)), dates, rotation=90)
#    plt.xlabel(r"Dates")
#    plt.ylabel(r"Numbers per week")
    fig.legend((fill1, line1, line2), ("Real", "Obsv", "Fitted"), loc=(0.7, 0.08), fontsize=15)
    
    pdf = PdfPages("%s_linechart_for_community.pdf" % modelname)
    pdf.savefig(fig)
    pdf.close()
#    ax.legend(fontsize=5, bbox_transform=plt.gcf().transFigure, loc = 'lower right') # loc='upper left'
#    plt.title("Confirmed cases in %s" % statename)



def find_nonzero(alphas):
    x, y = np.where(alphas != 0)
    values = alphas[x, y]
    return x, y, values

def construct_graph(alphas):
    x, y, values = find_nonzero(alphas)
    G = nx.Graph()
    G.add_nodes_from(np.arange(alphas.shape[0]))
    edges = list(zip(x, y))
    G.add_edges_from(edges)
    for i in np.arange(len(edges)):
        edge = edges[i]
        G[edge[0]][edge[1]]['weight'] = values[i]
    
    return G

def community_detection(alphas):
    G = construct_graph(alphas)
    partition = community_louvain.best_partition(G)
    partition_sorted = [item[0] for item in sorted(partition.items(), key=lambda x: x[1])]
    return partition, partition_sorted

def heatmap_for_alpha(alphas, modelname, figsize = (8, 8)):
    partition, order = community_detection(alphas)
    counts = pd.DataFrame(list(partition.values())).value_counts()[sorted(set(partition.values()))]
    idx = list(np.cumsum(counts))
    idx.insert(0, 0)
    
    get_colors = lambda n: list(map(lambda i:"#" +"%06x" % random.randint(0xF00000, 0xFFFFFF),range(n)))
    colors = get_colors(len(set(partition.values())))
    
    left, width       = 0.06, 0.8
    bottom, height    = 0, 0.8
    left_h = left + width + 0.01
    bottom_h = bottom + height + 0.01
    wid_com_label = 0.05
    
    rect_heatmap = [left, bottom, width, height]
    rect_com_label_above = [left, bottom_h, width, wid_com_label]
    rect_com_label_left = [0, 0, wid_com_label, height]
    
    fig = plt.figure(figsize=figsize)
    ax_hm = plt.axes(rect_heatmap)
    hm = ax_hm.imshow(alphas[order, :][:, order], cmap="YlGn")
    #hm = sns.heatmap(alphas[order, :][:, order], cmap="YlGnBu", cbar="Alpha")
    
    ax_hm.set_xticks(np.arange(alphas.shape[0]))
    ax_hm.set_yticks(np.arange(alphas.shape[0]))
    ax_hm.yaxis.tick_right()
    ax_hm.set_xticklabels(np.array(order) + 1)
    ax_hm.set_yticklabels(np.array(order) + 1)
    for i in range(0, len(idx)-2):
        ax_hm.plot(np.arange(idx[i], idx[i+1] + 1, 1) - 0.5, np.ones(idx[i+1] + 1 - idx[i]) * idx[i] - 0.5, color = "r", alpha = 0.3, linestyle = "--")
        ax_hm.plot(np.arange(idx[i], idx[i+1] + 1, 1) - 0.5, np.ones(idx[i+1] + 1 - idx[i]) * idx[i+1] - 0.5, color = "r", alpha = 0.3, linestyle = "--")
        ax_hm.plot(np.ones(idx[i+1] + 1 - idx[i]) * idx[i] - 0.5, np.arange(idx[i], idx[i+1] + 1, 1) - 0.5, color = "r", alpha = 0.3, linestyle = "--")
        ax_hm.plot(np.ones(idx[i+1] + 1 - idx[i]) * idx[i+1] - 0.5, np.arange(idx[i], idx[i+1] + 1, 1) - 0.5, color = "r", alpha = 0.3, linestyle = "--")
    ax_hm.plot(np.arange(idx[-2], idx[-1] + 1, 1) - 0.5, np.ones(idx[-1] + 1 - idx[-2]) * idx[-2] - 0.5, color = "r", alpha = 0.3, linestyle = "--")
    ax_hm.plot(np.ones(idx[-1] + 1 - idx[-2]) * idx[-2] - 0.5, np.arange(idx[-2], idx[-1] + 1, 1) - 0.5, color = "r", alpha = 0.3, linestyle = "--")
    
    lba_axes = fig.add_axes(rect_com_label_above)
    lba_axes.get_xaxis().set_ticks([])
    lba_axes.get_yaxis().set_ticks([])
    lba_axes.patch.set_visible(False)
    for edge, spine in lba_axes.spines.items():
        spine.set_visible(False)
    lba_axes.set_xlim(0, alphas.shape[0])
    lba_axes.set_ylim(0, 1)
    
    for c in range(len(idx) - 1): 
        lba_axes.fill_between(np.arange(idx[c], idx[c+1]+1), np.ones(idx[c+1]+1 - idx[c]), np.zeros(idx[c+1]+1 - idx[c]), facecolor=colors[c], alpha=0.5, interpolate=True)

        
    lbl_axes = fig.add_axes(rect_com_label_left)
    lbl_axes.get_xaxis().set_ticks([])
    lbl_axes.get_yaxis().set_ticks([])
    lbl_axes.patch.set_visible(False)
    for edge, spine in lbl_axes.spines.items():
        spine.set_visible(False)
    lbl_axes.set_ylim(0, alphas.shape[0])
    lbl_axes.set_xlim(0, 1)
    
    for c in range(len(idx) - 1): 
        lbl_axes.fill_between(np.arange(0, 2), alphas.shape[0] - np.ones(2) * (idx[c+1]), alphas.shape[0] - np.ones(2) * idx[c], facecolor=colors[c], alpha=0.5, interpolate=True)
        #     cbar = fig.colorbar(img, cax=cbaxes)
#     cbar.set_ticks([
#         np.log(error_mat + 1e-5).min(), 
#         np.log(error_mat + 1e-5).max()
#     ])
#     cbar.set_ticklabels([
#         0, # "%.2e" % error_mat.min(), 
#         "%.2f" % error_mat.max()
#     ])
#    cbar.ax.set_ylabel('MAE', rotation=270, labelpad=-5)
    
    cbaxes = fig.add_axes([left_h +0.03, 0, 0.06, height])
    cbar = fig.colorbar(hm, cax=cbaxes)
    cbar.ax.set_ylabel("Alpha", fontsize=15)#, rotation=-90, va="bottom")
    
    plt.show()
    
    pdf = PdfPages("%s_heatmap_of_alpha.pdf" % modelname)
    pdf.savefig(fig, bbox_inches = 'tight')
    pdf.close()
    
    return partition, order, colors
    
    
def heatmap_for_MAE(obs, fitted, comuna, dates, modelname, time_mean = True):

#    dates = dates[-nweeks:]
    ndates = len(dates)
    ncomuna = len(comuna)

    error_mat = np.abs(obs - fitted)
    MAE_mean = error_mat.mean(1)
    com_order = np.argsort(MAE_mean)
    
#     print(error_mat.shape)
#     print(len(dates))

    comuna_order = [ comuna[ind] for ind in com_order ]
    error_mat    = error_mat[comuna_order, :]
    # error_mat1   = error_mat1[states_order, :]
    # error_mat2   = error_mat2[states_order, :]
    rev_comuna_order = np.flip(comuna_order)
    error_comuna  = MAE_mean[rev_comuna_order]
    error_date = error_mat.mean(0)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width       = 0.15, 0.65
    bottom, height    = 0.15, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_imshow = [left, bottom, width, height]
    rect_week   = [left, bottom_h, width, 0.12]
    rect_state  = [left_h, bottom, 0.12, height]

    for i in range(1):
        # start with a rectangular Figure
        fig = plt.figure(1, figsize=(8, 8))

        ax_imshow = plt.axes(rect_imshow)
        ax_state  = plt.axes(rect_state)
        ax_week   = plt.axes(rect_week)

        # no labels
        ax_state.xaxis.set_major_formatter(nullfmt)
        ax_week.yaxis.set_major_formatter(nullfmt)

        # the error matrix for counties in states:
        cmap = matplotlib.cm.get_cmap('magma')
        img  = ax_imshow.imshow(np.log(error_mat + 1e-5), cmap=cmap, extent=[0,ndates,0,ncomuna], aspect=float(ndates) / float(ncomuna))
        ax_imshow.set_yticks(np.arange(ncomuna))
        ax_imshow.set_yticklabels(rev_comuna_order)
        
        if time_mean:
            ax_imshow.set_xticks(np.arange(ndates))
            ax_imshow.set_xticklabels(dates, rotation=90)
        else:
            ax_imshow.set_xticks(np.arange(1, ndates, 5))
            ax_imshow.set_xticklabels(dates[np.arange(1, ndates, 5)], rotation=90)

        # the error vector for states and weeks
        ax_state.plot(np.log(error_comuna + 1e-5), np.arange(len(comuna)), "b.-", linewidth=2, label="MAE", alpha=.5)
        # ax_state.legend(loc="upper right")
        ax_week.plot(np.log(error_date + 1e-5), "b.-", linewidth=2, label="MAE", alpha=.5)

        ax_state.get_yaxis().set_ticks([])
        ax_state.get_xaxis().set_ticks([])
        ax_state.set_xlabel("Avg MAE")
        ax_state.set_ylim(0, len(comuna))
        ax_week.get_xaxis().set_ticks([])
        ax_week.get_yaxis().set_ticks([])
        ax_week.set_ylabel("Avg MAE")
        ax_week.set_xlim(0, ndates)
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.92, 0.133, '%.2f' % max(error_comuna))
        plt.figtext(0.135, 0.81, '0')
        # plt.figtext(0.105, 0.915, '%.2f' % max(max(error_week), max(error_week1)))
        plt.figtext(0.095, 0.915, '%.2f' % max(error_date))
        plt.legend(loc="upper right")

        cbaxes = fig.add_axes([left_h, height + left + 0.01, .03, .12])
        cbaxes.get_xaxis().set_ticks([])
        cbaxes.get_yaxis().set_ticks([])
        cbaxes.patch.set_visible(False)
        cbar = fig.colorbar(img, cax=cbaxes)
        cbar.set_ticks([
            np.log(error_mat + 1e-5).min(), 
            np.log(error_mat + 1e-5).max()
        ])
        cbar.set_ticklabels([
            0, # "%.2e" % error_mat.min(), 
            "%.2f" % error_mat.max()
        ])
        cbar.ax.set_ylabel('MAE', rotation=270, labelpad=-5)

        fig.tight_layout()
        
        pdf = PdfPages("%s_heatmap_of_MAE.pdf" % modelname)
        pdf.savefig(fig, bbox_inches = 'tight')
        pdf.close()
        
def draw_polygon_folium(Polygons, partition, colors, m):
    
    for i in range(len(Polygons)):
        folium.Choropleth(
            geo_data=Polygons[i],
            fill_color=colors[partition[i]],
            fill_opacity=0.6,
            line_opacity=0.2,
        ).add_to(m)

def draw_center_point(ctps, mus, m):
    for i in range(len(mus)):
        folium.CircleMarker(location=(ctps[i, 0], ctps[i, 1]),
                        radius= 10 * mus[i] / max(mus),
                        color="navy",
                        fillcolor="navy",
                        fill=True).add_to(m)

def draw_arrow_folium(alphas, m):
    max_alpha = alphas.max()
    
    for i in np.arange(alphas.shape[1]):
        idx_non0 = np.where(alphas[:, i] != 0)[0]
        start_p = ctps[i, :]
        for j in idx_non0:
            if j == i: continue
            end_p = ctps[j, :]
            wth = alphas[j, i]
            coordinates=[list(start_p), list(end_p)]
            xdelta = (end_p[1] - start_p[1])
            ydelta = (end_p[0] - start_p[0])
            angle = np.arctan(ydelta / xdelta) / np.pi * 180 + 180 * (xdelta < 0)
            m.add_child(folium.PolyLine(locations=coordinates, weight= 10 * float(wth / max_alpha),
                                               color = '#6B8E23'))
            folium.RegularPolygonMarker(location=list(end_p),
                                        color='#6B8E23', fillcolor='#6B8E23',
                                        number_of_sides=3, radius=20 * float(wth / max_alpha),
                                        rotation=360 - angle).add_to(m)

def alpha_visualization_with_arrow(alphas, mus, ctps, Polygons, partition, colors):
    center = [3.42, -76.51]
    map_cali = folium.Map(location=center, zoom_start=12, tiles='cartodbpositron', zoom_control=False, scrollWheelZoom=False)
    
    draw_polygon_folium(Polygons, partition, colors, map_cali)
    draw_center_point(ctps, mus, map_cali)
    draw_arrow_folium(alphas, map_cali)
    
    return map_cali