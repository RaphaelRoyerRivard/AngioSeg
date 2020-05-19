import numpy as np
from scipy import ndimage
from scipy import misc
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os
import os.path
import sys, getopt
import cv2
import ctypes
from ctypes import *
import numpy as np
import cv2
import os.path
import math
import networkx as nx
import random
import argparse
import time
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

_DIRNAME = os.path.dirname(__file__)
toolsdll=cdll.LoadLibrary(os.path.join(_DIRNAME,'VesselAnalysis/x64/Release/VesselAnalysis.dll'))
toolsdllcuda=cdll.LoadLibrary(os.path.join(_DIRNAME,'CUDASkel/x64/Release/CUDASkel.dll'))


def va_getskeleton(image):
    skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskel = skeleton.tostring()
    strseg = image.tostring()

    toolsdll.vesselanalysis_getskeleton(c_char_p(strseg), image.shape[0], image.shape[1], c_char_p(strskel))

    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))

    return cvskel

def va_getskeleton_cuda(image):
    skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskel = skeleton.tostring()
    strseg = image.tostring()

    toolsdllcuda.skeletonize(c_char_p(strseg), c_char_p(strskel), image.shape[1], image.shape[0], 512, 512)

    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))

    return cvskel

def va_getskeletoncomponents_cuda(image):
    skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskel = skeleton.tostring()
    strseg = image.tostring()

    toolsdllcuda.skeletonize(c_char_p(strseg), c_char_p(strskel), image.shape[1], image.shape[0], 512, 512)

    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))

    cc = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    strcc = cc.tostring()
    nbcomponents=c_int(0)
    toolsdll.vesselanalysis_getcomponents(c_char_p(strskel), image.shape[0], image.shape[1], byref(nbcomponents), c_char_p(strcc))

    cvdt = np.fromstring(strcc, np.int32)
    cvdt = np.reshape(cvdt,(image.shape[0] ,image.shape[1]))

    return cvskel, cvdt

def va_getskeletondt_cuda(image, gradient):
    skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskel = skeleton.tostring()
    strseg = image.tostring()
    strgrad = gradient.tostring()

    toolsdllcuda.skeletonize(c_char_p(strseg), c_char_p(strskel), image.shape[1], image.shape[0], 512, 512)

    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))

    dft = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    strdft = dft.tostring()
    toolsdllcuda.distancetransform(c_char_p(strgrad), c_char_p(strdft),image.shape[0], image.shape[1])

    cvdt = np.fromstring(strdft, np.uint8)
    cvdt = np.reshape(cvdt,(image.shape[0] ,image.shape[1]))

    return cvskel, cvdt


def display_voronoi(voronoi, skel, image, displayupsc):
    voronoidisp = np.zeros((image.shape[0]*displayupsc, image.shape[1]*displayupsc, 3), dtype=np.uint8)
    if not displayupsc==1:
        imagerescale = cv2.resize(image, (image.shape[0] * displayupsc, image.shape[1] * displayupsc))
    else:
        imagerescale = image
    voronoidisp[:, :, :] = imagerescale
    xnonzeros = np.nonzero(skel)
    for i in range(1, len(xnonzeros[0]), 1):
        (x, y) = (xnonzeros[0][i], xnonzeros[1][i])
        xclose = voronoi[x, y, 1]*displayupsc
        yclose = voronoi[x, y, 0]*displayupsc
        xdeb = (2 * x *displayupsc - xclose)
        ydeb = (2 * y *displayupsc - yclose)
        cv2.line(voronoidisp, (ydeb, xdeb), (yclose, xclose), (0, 255, 0), 1)

    return voronoidisp


def display_natural_graph(G, image, color=None, treeori=None):
    displayupsc = 5
    imagegraph = np.zeros((image.shape[0] * displayupsc, image.shape[1] * displayupsc, 3), dtype=np.uint8)
    imagerescale = cv2.resize(image[:, :, 1], (image.shape[0] * displayupsc, image.shape[1] * displayupsc))
    imagegraph[:, :, 0] = imagerescale
    imagegraph[:, :, 1] = imagerescale
    imagegraph[:, :, 2] = imagerescale

    if color is None:
        inputcolor = (0, 255, 0)
    else:
        inputcolor = color

    for n in G.edges():
        weight = G.edges[n]['weight']
        line1 = G.nodes[n[0]]['pos']
        line2 = G.nodes[n[1]]['pos']

        linex1 = int(line1[1] * displayupsc)
        liney1 = int(line1[0] * displayupsc)
        linex2 = int(line2[1] * displayupsc)
        liney2 = int(line2[0] * displayupsc)

        listposedge = G.edges[n]['listnodes']

        inputcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        if len(listposedge) > 1:

            for i in range(len(listposedge)):
                n1 = listposedge[i][0]
                n2 = listposedge[i][1]
                line1 = treeori.nodes[n1]['pos']
                line2 = treeori.nodes[n2]['pos']
                diameterc = (0.5 * treeori.nodes[n1]['diameter'] + 0.5 * treeori.nodes[n2]['diameter'])*displayupsc

                linex1c = int(line1[1] * displayupsc)
                liney1c = int(line1[0] * displayupsc)
                linex2c = int(line2[1] * displayupsc)
                liney2c = int(line2[0] * displayupsc)
                cv2.line(imagegraph, (linex1c, liney1c), (linex2c, liney2c), inputcolor, int(diameterc))
                cv2.circle(imagegraph, (linex1c, liney1c), 3, (0, 0, 0), -1)
                cv2.circle(imagegraph, (linex2c, liney2c), 3, (0, 0, 0), -1)

                inputcolor = (inputcolor[0]+10, inputcolor[1]+10, inputcolor[2]+10)

        else:
            diameter = int(G.edges[n]['diameter'] * displayupsc)
            if color is None:
                if weight == 0:
                    inputcolor = (0, 255, 0)
                else:
                    inputcolor = (weight * 5, weight * 5, weight * 5)
                    diameter=displayupsc*2

            cv2.line(imagegraph, (linex1, liney1), (linex2, liney2), inputcolor, diameter)

        cv2.circle(imagegraph, (linex1, liney1), 10, (255, 255, 255), -1)
        cv2.circle(imagegraph, (linex2, liney2), 10, (255, 255, 255), -1)
        cv2.putText(imagegraph, str(n[0]), (linex1 - 5, liney1 + 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(imagegraph, str(n[1]), (linex2 - 5, liney2 + 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


# for n, nbrs in G.adj.items():
#     for nbr, eattr in nbrs.items():
#         wt = eattr['weight']
#         if wt < 10: print('(%d, %d, %.3f)' % (n, nbr, wt))

    return imagegraph

def va_getskeletondtcomponents_cuda(ori, image, gradient, num):
    skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskel = skeleton.tostring()
    strseg = image.tostring()
    strgrad = gradient.tostring()

    toolsdllcuda.skeletonize(c_char_p(strseg), c_char_p(strskel), image.shape[1], image.shape[0], 512, 512)

    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))

    dft = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    voronoi = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.int16)
    strdft = dft.tostring()
    strvoronoi = voronoi.tostring()
    toolsdllcuda.distancetransform(c_char_p(strgrad), c_char_p(strdft),c_char_p(strvoronoi), image.shape[0], image.shape[1])

    cvdt = np.fromstring(strdft, np.float32)
    cvdt = np.reshape(cvdt,(image.shape[0] ,image.shape[1]))

    cvvoronoi = np.fromstring(strvoronoi, np.int16)
    cvvoronoi = np.reshape(cvvoronoi,(image.shape[0] ,image.shape[1], 2))

    #voronoi_disp = display_voronoi(cvvoronoi, cvskel, ori)


    cc = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    strcc = cc.tostring()
    nbcomponents=c_int(0)
    toolsdllcuda.getconnectedcomponents(c_char_p(strskel), c_char_p(strcc), image.shape[0], image.shape[1], 8,  byref(nbcomponents))
    cvcomp = np.fromstring(strcc, np.int32)
    cvcomp = np.reshape(cvcomp,(image.shape[0] ,image.shape[1]))


    #cv2.imwrite('testcomp' + str(num) + 'skel.png', (cvskel * 255).astype(np.uint8))
    #cv2.imwrite('testcomp' + str(num) + 'voronoi.png', voronoi_disp)




    # plt.subplot(121)
    # plt.imshow(imagegraph)
    # plt.subplot(122)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()


    #for i in

    #cv2.imwrite('testcomp'+str(num)+'.png', (cvskel*255).astype(np.uint8))



    return cvskel, cvdt, cvvoronoi,  cvcomp, nbcomponents.value #, cvbuf


def takeSecond(elem):
    return elem[1]

def va_creategraph(ori, cc, dft, skel, nbcomponents, diammin):


    buffer = np.zeros((nbcomponents, 6), dtype=np.float32)
    strcc = cc.tostring()
    strdft = dft.tostring()
    strskel = skel.tostring()
    strbuf = buffer.tostring()
    toolsdll.vesselanalysis_getstats(c_char_p(strcc),c_char_p(strdft), c_char_p(strskel),ori.shape[0], ori.shape[1], 6, nbcomponents, c_char_p(strbuf))
    cvbuf = np.fromstring(strbuf, np.float32)
    cvbuf = np.reshape(cvbuf,(nbcomponents, 6))

    for i in cvbuf:
        if i[5]>0:
            i[4] = i[4]/i[5]
    #np.sort(cvbuf, axis=5)

    distanceposition = np.ones((nbcomponents*2, nbcomponents*2))*30
    for i in range(nbcomponents):
        if cvbuf[i,5]>4 and cvbuf[i,4]>diammin:
            for j in range(nbcomponents):
                if cvbuf[j, 5] > 4 and cvbuf[j,4]>diammin:
                    if i<j:
                        distanceposition[i*2,j*2] = min(30, 6*min(cvbuf[i,4], cvbuf[j,4]))
                        distanceposition[i * 2, j * 2+1] =min(30,  6 * min(cvbuf[i, 4], cvbuf[j, 4]))
                        distanceposition[i * 2+1, j * 2] = min(30, 6 * min(cvbuf[i, 4], cvbuf[j, 4]))
                        distanceposition[i * 2+1, j * 2+1] =min(30,  6 * min(cvbuf[i, 4], cvbuf[j, 4]))
                        if (cvbuf[i,0]!=0 or cvbuf[i,1]!=0) and (cvbuf[j,0]!=0 or cvbuf[j,1]!=0):
                            distanceposition[i*2,j*2] = min(distanceposition[i*2,j*2], math.sqrt((cvbuf[i,1]-cvbuf[j,1])*(cvbuf[i,1]-cvbuf[j,1])+
                                                                                           (cvbuf[i,0] - cvbuf[j,0])*(cvbuf[i,0]-cvbuf[j,0])))
                        if (cvbuf[i,0]!=0 or cvbuf[i,1]!=0) and (cvbuf[j,2]!=0 or cvbuf[j,3]!=0):
                            distanceposition[i*2,j*2+1] = min(distanceposition[i*2,j*2+1], math.sqrt((cvbuf[i,1]-cvbuf[j,3])*(cvbuf[i,1]-cvbuf[j,3])+
                                                                                           (cvbuf[i,0] - cvbuf[j,2])*(cvbuf[i,0]-cvbuf[j,2])))
                        if (cvbuf[i,2]!=0 or cvbuf[i,3]!=0) and (cvbuf[j,0]!=0 or cvbuf[j,1]!=0):
                            distanceposition[i*2+1,j*2] = min(distanceposition[i*2+1,j*2], math.sqrt((cvbuf[i,3]-cvbuf[j,1])*(cvbuf[i,3]-cvbuf[j,1])+
                                                                                           (cvbuf[i,2] - cvbuf[j,0])*(cvbuf[i,2]-cvbuf[j,0])))
                        if (cvbuf[i,2]!=0 or cvbuf[i,3]!=0) and (cvbuf[j,2]!=0 or cvbuf[j,3]!=0):
                            distanceposition[i*2+1,j*2+1] = min(distanceposition[i*2+1,j*2+1], math.sqrt((cvbuf[i,3]-cvbuf[j,3])*(cvbuf[i,3]-cvbuf[j,3])+
                                                                                           (cvbuf[i,2] - cvbuf[j,2])*(cvbuf[i,2]-cvbuf[j,2])))

    G = nx.Graph()

    for i in range(nbcomponents):
        if cvbuf[i, 5] > 4 and cvbuf[i,4]>diammin:
            G.add_node(i*2, pos=(cvbuf[i,0], cvbuf[i,1]), diameter = cvbuf[i,4])
            G.add_node(i*2+1, pos=(cvbuf[i,2], cvbuf[i,3]), diameter = cvbuf[i,4])
            G.add_edge(i*2, i*2+1, weight=0, diameter = cvbuf[i,4])
    for i in range(nbcomponents*2):
        for j in range(nbcomponents*2):
            if (distanceposition[i,j] < min(30,6*min(cvbuf[int(i/2),4], cvbuf[int(j/2),4]))):
                i2 = int(i/2)
                j2 = int(j/2)
                y1 = cvbuf[i2,2] - cvbuf[i2,0]
                x1 = cvbuf[i2, 3] - cvbuf[i2, 1]
                y2 = cvbuf[j2,2] - cvbuf[j2,0]
                x2 = cvbuf[j2, 3] - cvbuf[j2, 1]
                angle1 = math.atan(y1/x1)
                angle2 = math.atan(y2/x2)
                da = abs(angle2-angle1)
                if da>math.pi/2:
                    da -=math.pi
                da = abs(da)
                G.add_edge(i,j, weight=distanceposition[i,j]+abs(cvbuf[int(i/2),4]-cvbuf[int(j/2),4])+da*5, diameter = 0.5*cvbuf[int(i/2),4]+0.5*cvbuf[int(j/2),4])

    T = nx.minimum_spanning_tree(G)

    root = sorted(list(T.nodes('diameter')), key=takeSecond, reverse=True)[0][0]
    if root%2==1:
        root = root-1

    nx.set_node_attributes(T, 0, name='fetch')

    for n in T.edges():
        T.edges[n]['listnodes'] = []

    nx.set_edge_attributes(G, [], name='listnodes')

    Tmerged = T.copy()


    # on merge les noeuds qui sont seuls
    for n in nx.dfs_preorder_nodes(T, source=root):
        #line0 = T.nodes[n]['posline']
        for nbr in T[n]:
            if (n==root or not T.nodes[nbr]['fetch']):
                Tmerged.edges[(n, nbr)]['listnodes'] = [(n, nbr)]

            if not T.nodes[nbr]['fetch'] and T.degree[n]<3 and n!=root and nbr!=root:
                Tmerged.edges[(n, nbr)]['listnodes'] = [(n, nbr)]

                Tmerged = nx.contracted_nodes(Tmerged, nbr, n, self_loops=False)

                # apres contraction on doit pouvoir updater l edge
                for nbr2 in Tmerged[nbr]:
                    if T.nodes[nbr2]['fetch'] :
                        # alors c est cet edge
                        Tmerged.edges[nbr, nbr2]['listnodes'].append((n,nbr))

        T.nodes[n]['fetch']=1



    imagegraph=display_natural_graph(G, ori)

    imagetree = display_natural_graph(Tmerged, ori, color=(0,255,0), treeori=T)



    return imagegraph, imagetree, T, Tmerged


def fill_missing_pixels(image):
    filter = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    res = convolve2d(image, filter, mode="same")
    image[np.where(res >= res.max() * 0.2)] = 255


def find_ostia(skeleton_distances, bifurcations):
    five_percent = int(skeleton_distances.shape[1] * 0.05)
    potential_catheters = np.where(skeleton_distances[five_percent, :] > 0)
    print(potential_catheters)
    # plt.imshow(skeleton_distances)
    # plt.show()
    for pixel_x in potential_catheters[0]:
        if skeleton_distances[five_percent+1, pixel_x] > 0:  # Bottom
            starting_point = (pixel_x, five_percent+1)
        elif skeleton_distances[five_percent+1, pixel_x-1] > 0:  # Bottom left
            starting_point = (pixel_x-1, five_percent+1)
        elif skeleton_distances[five_percent+1, pixel_x+1] > 0:  # Bottom right
            starting_point = (pixel_x+1, five_percent+1)
        else:  # No valid points to do the pathing
            print(f"No valid points to do the pathing for ({pixel_x}, {five_percent})")
            continue
        nodes, vessel_width = follow_path(skeleton_distances, bifurcations, starting_point, (pixel_x, five_percent))
        widening_location = get_location_of_widening(nodes, vessel_width)
        if widening_location is not None:
            return widening_location

    return None


class PathNode:

    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def follow_path(skeleton_distances, bifurcations, starting_point, parent=None, crossing_regression_model=None):
    """
    Follows a path on the skeleton until it reaches a big change in width or a bifurcation.
    :param skeleton_distances: A 2D ndarray containing the skeleton with distances.
    :param bifurcations: A 2D ndarray containing the points of bifurcations on the skeleton.
    :param starting_point: A tuple of indices (x, y) representing the position in the skeleton on which we want to start pathing.
    :param parent: A tuple of indices (x, y) representing the position in the skeleton from which the starting point was chosen.
    It basically allows the algorithm to know in what direction to start pathing.
    :param crossing_regression_model: Regression model computed with sklearn on the pixels of the vessel skeleton we want to find the continuity.
    :return: Returns a list of nodes representing the path taken and the vessel width along the path.
    """
    nodes_in_path = 0
    parent_node = None if parent is None else PathNode(parent[0], parent[1])
    current_node = PathNode(starting_point[0], starting_point[1], parent_node)
    while True:
        nodes_in_path += 1
        patch = skeleton_distances[current_node.y-1:current_node.y+2, current_node.x-1:current_node.x+2]  # 3x3 around the current point
        potential_points = []
        for y in range(patch.shape[0]):
            for x in range(patch.shape[1]):
                if patch[y, x] > 0:  # if pixel is lit up
                    if x == 1 and y == 1:
                        continue  # skip current point
                    if current_node.parent is not None:
                        if current_node.x + x - 1 == current_node.parent.x and current_node.y + y - 1 == current_node.parent.y:
                            continue  # skip parent point
                        if current_node.parent.parent is not None and current_node.x + x - 1 == current_node.parent.parent.x and current_node.y + y - 1 == current_node.parent.parent.y:
                            continue  # skip parent's parent point
                    potential_points.append((x, y))

        if len(potential_points) == 0:
            break  # finished on a dead end

        next_point = potential_points[0]
        if len(potential_points) > 1:  # there was 2 new pixels next to the current point
            # we find the closest one (the one that is not in diagonal)
            for potential_point in potential_points:
                if abs(potential_point[0]) + abs(potential_point[1]) == 1:
                    next_point = potential_point
                    break

        current_node = PathNode(current_node.x + next_point[0] - 1, current_node.y + next_point[1] - 1, current_node)

        is_bifurcation = bifurcations[current_node.y, current_node.x] > 0
        check_for_crossing = crossing_regression_model is not None and (nodes_in_path == 20 or is_bifurcation)
        if is_bifurcation or check_for_crossing:
            # compute linear regression on the last few points to get the vessel angle
            reg = get_regression_model(current_node)
            print("reg", reg.coef_, reg.intercept_)
            if check_for_crossing:
                print("crossing_regression_model", crossing_regression_model.coef_, crossing_regression_model.intercept_)
            if is_bifurcation:
                print("bifurcation")
                # start a new path following on branches
                _, branches = detect_bifurcation((current_node.y, current_node.x), skeleton_distances)
                for branch_y, branch_x in branches:
                    branch_point = (current_node.x + branch_x, current_node.y + branch_y)
                    if branch_point[0] != current_node.parent.x or branch_point[1] != current_node.parent.y:
                        print(f"new branch ({branch_x}, {branch_y})")
                        regression_model_param = crossing_regression_model if crossing_regression_model is not None else reg
                        follow_path(skeleton_distances, bifurcations, branch_point, parent=(current_node.x, current_node.y), crossing_regression_model=regression_model_param)
                break  # finished on a bifurcation

    nodes, vessel_width = get_nodes_and_vessel_width(current_node, skeleton_distances)
    return nodes, vessel_width


def get_regression_model(end_node):
    points = []
    backtrack_node = end_node.parent
    while backtrack_node is not None and len(points) < 20:
        points.append([backtrack_node.x, backtrack_node.y])
        backtrack_node = backtrack_node.parent
    points = np.array(points)
    x = points[:, np.newaxis,  0]
    y = points[:, np.newaxis, 1]
    return LinearRegression().fit(x, y)


def get_nodes_and_vessel_width(end_node, skeleton_distances):
    current_node = end_node
    nodes = [current_node]
    distances = []
    while current_node is not None:
        distances.append(skeleton_distances[current_node.y, current_node.x])
        current_node = current_node.parent
        nodes.append(current_node)
    nodes.pop()
    nodes = list(reversed(nodes))

    vessel_width = np.array(list(reversed(distances)))
    vessel_width = gaussian_filter1d(vessel_width, 6)

    return nodes, vessel_width


def get_location_of_widening(nodes, vessel_width):
    plt.title("Vessel width for pixels of skeleton segment")
    plt.plot(vessel_width)
    plt.xlabel("Segment pixel #")
    plt.ylabel("Vessel width in pixels")
    plt.show()

    for i in range(10, len(vessel_width)):
        if vessel_width[i-10] > 0:
            ratio = vessel_width[i] / vessel_width[i-10]
            if ratio >= 1.5:
                return nodes[i].x, nodes[i].y

    return None


def vesselanalysis(image, out_name, use_scikit=True):
    print(out_name)

    # 1 get skeleton
    start = time.time()
    if use_scikit:
        skeleton, distance = medial_axis(image, return_distance=True)
        # skeleton = (skeletonize(image, method='lee') / 255).astype(np.uint8)
        # skeleton2, distance = medial_axis(image, return_distance=True)
        # skeletons = np.array([skeleton, skeleton2, np.zeros_like(image)])
        # skeletons = np.moveaxis(skeletons, 0, 2)
        # print(skeletons.min(), skeletons.max(), skeletons.mean())
        # plt.imshow(skeleton, vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(skeleton2, vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(skeletons*255)
        # plt.show()
        strskel = skeleton.tostring()
    else:
        # 1 get skeleton
        distance = np.ones_like(image)
        start = time.time()
        skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        strskel = skeleton.tostring()
        strseg = image.tostring()
        toolsdll.vesselanalysis_getskeleton(c_char_p(strseg), image.shape[0], image.shape[1], c_char_p(strskel))
        # cvskel = np.fromstring(strskel, np.uint8)
        # cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))
        # cv2.imwrite(f'{out_path}\\test.png', cvskel*255)
    print(f"1. skeleton took {time.time() - start}s")

    # 2b get rid of spur
    start = time.time()
    skeletonspur = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    strskelspur = skeletonspur.tostring()
    toolsdll.va_spur_pruning(c_char_p(strskel), image.shape[0], image.shape[1], 20, c_char_p(strskelspur))
    cvskelspur = np.fromstring(strskelspur, np.uint8)
    cvskelspur = np.reshape(cvskelspur, (image.shape[0], image.shape[1]))
    print(f"2. spur took {time.time() - start}s")
    cv2.imwrite(out_name + ".png", cvskelspur*255)

    # 3 get components (separate branches)
    # start = time.time()
    # cc = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # strcc = cc.tostring()
    # nbcomponents = np.zeros(1, dtype=np.uint8)
    # strnbcomponents = nbcomponents.tostring()
    # toolsdll.vesselanalysis_getcomponents(c_char_p(strskelspur), image.shape[0], image.shape[1], strnbcomponents, c_char_p(strcc))
    # nbcomponents = np.fromstring(strnbcomponents, np.uint8)[0]
    # components = np.fromstring(strcc, np.uint8)
    # print(f"3. components took {time.time() - start}s")
    # start = time.time()
    # component_colors = np.random.rand(nbcomponents, 3) * 255
    # components = np.repeat(components[:, np.newaxis], 3, axis=1)
    # for i in range(components.shape[0]):
    #     if components[i, 0] > 0:
    #         components[i, :] = component_colors[components[i, 0] - 1]
    # components = np.reshape(components, (image.shape[0], image.shape[1], 3))
    # print(f"4. colors took {time.time() - start}s")
    # cv2.imwrite(out_name + "_components.png", components)

    # 3 get components (separate branches)
    start = time.time()
    bifurcation_pixels_x = []
    bifurcation_pixels_y = []
    skel_x, skel_y = np.where(cvskelspur > 0)
    for pixel in range(len(skel_x)):
        x = skel_x[pixel]
        y = skel_y[pixel]
        bifurcation, _ = detect_bifurcation((x, y), cvskelspur)
        if bifurcation:
            bifurcation_pixels_x.append(x)
            bifurcation_pixels_y.append(y)
    print(len(bifurcation_pixels_x))
    bifurcations_layer = np.zeros(cvskelspur.shape, dtype=np.uint8)
    bifurcations_layer[np.array(bifurcation_pixels_x, dtype=np.int), np.array(bifurcation_pixels_y, dtype=np.int)] = 1
    bifurcations = np.repeat(cvskelspur[:, :, np.newaxis], 3, axis=2)
    bifurcations[:, :, 1] -= bifurcations_layer
    print(f"3. bifurcations took {time.time() - start}s")
    cv2.imwrite(out_name + "_bifurcations.png", bifurcations*255)

    skeleton_distance = cvskelspur * distance
    return skeleton_distance, bifurcations_layer


def detect_bifurcation(point, skeleton):
    """
    Check the branches around a point on a skeleton to determine if it is a bifurcation.
    :param point: Tuple (x, y)
    :param skeleton: 2D ndarray representing the skeleton.
    :return: A boolean revealing if the point is a bifurcation and a list containing the starting point of the branches.
    """
    branches = []
    x = point[0]
    y = point[1]
    patch = skeleton[x-1:x+2, y-1:y+2]
    # The 3x3 patch needs at least 4 pixels to be a bifurcation, otherwise it's a single branch
    if patch.sum() < 4:
        return False, branches
    # We need to check the pixels around the center pixel to count the branches
    for patch_x, patch_y in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        if patch[patch_x, patch_y] > 0:
            branches.append((patch_x, patch_y))
    # We also check the pixels in diagonal, but if they were next to an already identified branch, they count for the same branch
    for patch_x, patch_y in [(0, 0), (2, 0), (0, 2), (2, 2)]:
        if patch[patch_x, patch_y] > 0:
            same_branch = False
            for branch_x, branch_y in branches:
                if abs(branch_x - patch_x) + abs(branch_y - patch_y) == 1:
                    same_branch = True
                    break
            if not same_branch:
                branches.append((patch_x, patch_y))
    adjusted_branches = []
    for branch in branches:
        adjusted_branches.append((branch[0]-1, branch[1]-1))
    return True, adjusted_branches


def overlap(image, skeleton, out_name):
    # Overlapping skeleton and original image
    rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    skeleton_indices = skeleton.nonzero()
    if len(skeleton.shape) == 2:
        rgb_image[skeleton_indices[0], skeleton_indices[1], :] = 0
        rgb_image[skeleton_indices[0], skeleton_indices[1], 1] = 255
    else:
        rgb_image[skeleton_indices[0], skeleton_indices[1], :] = skeleton[skeleton_indices[0], skeleton_indices[1], :]
    cv2.imwrite(out_name, rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='input file or folder in which to search recursively for images of segmented frames to skeletonize')
    parser.add_argument('--file_type', default='png', help='type of images to skeletonize (png, jpg, tif, etc)')
    parser.add_argument('--contains', default='', help='filters out files that do not contain that string (i.e. "_seg")')
    parser.add_argument('--visualize', default=False, help='mode where it saves overlapped images of the skeleton over the original image')
    args = parser.parse_args()

    input_path = args.input
    file_type = args.file_type
    contains = args.contains
    visualize = args.visualize

    if os.path.isfile(input_path):
        print("Input is a file")
        file_type = input_path.split('.')[-1]
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        file_name = input_path.split(f".{file_type}")[0]
        fill_missing_pixels(image)
        skeleton, bifurcations = vesselanalysis(image, f"{file_name}_skeleton")
        ostia = find_ostia(skeleton, bifurcations)
        if ostia is not None:
            skeleton[skeleton > 0] = 1
            visualization = np.repeat(skeleton[:, :, np.newaxis], 3, 2)
            visualization[:, :, 1] -= bifurcations
            visualization[ostia[1], ostia[0], 0] -= 1
            plt.imshow(visualization)
            plt.show()
        else:
            print("Ostia not found")
    else:
        print("Input is a folder")
        for path, subfolders, files in os.walk(input_path):
            print(path)

            if visualize:
                # if "segmented" not in subfolders:
                #     continue
                # output_folder = f'{path}\\segmented\\overlapped'
                output_folder = f'{path}'
            else:
                # if path.split("\\")[-1] != "segmented":
                #     continue
                # output_folder = f'{path}\\skeletonized'
                output_folder = f'{path}'

            for file in files:
                image_type = f".{file_type}"
                if not file.endswith(image_type):
                    continue

                if contains != "" and contains not in file:
                    continue

                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                image_path = f'{path}\\{file}'
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                fill_missing_pixels(image)

                if visualize:
                    # file_name = file.split(".jpg")[0]
                    # skeleton_file_name = f'{path}\\segmented\\skeletonized\\{file_name}_skeletonized'
                    # if os.path.exists(f'{skeleton_file_name}_components.png'):
                    #     skeleton = cv2.imread(f'{skeleton_file_name}_components.png', cv2.IMREAD_COLOR)
                    # else:
                    #     skeleton = cv2.imread(f'{skeleton_file_name}.png', cv2.IMREAD_GRAYSCALE)
                    file_name = file.split("seg")[0]
                    output = f'{output_folder}\\{file_name}skeletonized'
                    skeleton, components = vesselanalysis(image, output)
                    output = f'{output_folder}\\{file_name}overlap.png'
                    image_path = f'{path}\\{file_name[:-1]}{image_type}'
                    print(image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    overlap(image, skeleton, output)
                else:
                    # file_name = file.split("segmented")[0]
                    file_name = file.split("seg")[0]
                    output = f'{output_folder}\\{file_name}skeletonized'
                    cv2.imwrite(f'{output_folder}\\{file_name}seg_filled.png', image)
                    vesselanalysis(image, output)
