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
import cv2
import os.path
import math
import networkx as nx
import random
import argparse
import time
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import gaussian_filter1d
import heapq
import operator

_DIRNAME = os.path.dirname(__file__)
toolsdll=cdll.LoadLibrary(os.path.join(_DIRNAME,'VesselAnalysis/x64/Release/VesselAnalysis.dll'))
toolsdllcuda=cdll.LoadLibrary(os.path.join(_DIRNAME,'CUDASkel/x64/Release/CUDASkel.dll'))


########################################################################################################################
##################################################  LEGACY CODE  #######################################################
########################################################################################################################


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


########################################################################################################################
##############################################  END OF LEGACY CODE  ####################################################
########################################################################################################################


def fill_missing_pixels(image):
    # Add a 1 pixel border to everything in the image and saturate values
    filter = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    res = convolve2d(image, filter, mode="same")
    image[np.where(res >= res.max() * 0.2)] = 255

    # Fill small holes
    image[np.where(image > 0)] = 1
    image = image.astype(np.int8)
    image[np.where(image <= 0)] = -1
    filter_center = np.array([[1, 1, 1],
                              [1,-1, 1],
                              [1, 1, 1]])
    res = convolve2d(image, filter_center, mode="same")
    image[np.where(res == 9)] = 1
    for i in range(2):
        for j in range(2):
            filter_corner = np.copy(filter_center)
            filter_corner[i*2, j*2] = -1
            res = convolve2d(image, filter_corner, mode="same")
            image[np.where(res == 9)] = 1
    image[np.where(image <= 0)] = 0
    
    return image.astype(np.uint8) * 255


def find_ostium(raw_image, skeleton_distances, bifurcations, search_best_crossing=True, debug=False):
    """
    This method searches for catheters on top of the image to detect the ostium.
    The ostium is detected by a big change in width, a sudden change in angle or the first encountered bifurcation.
    If the catheter is not connected to the coronary tree, we find the closest skeleton point and identify it as the ostium.
    :param raw_image: The 2D numpy array of pixel intensities.
    :param skeleton_distances: Skeleton distances 2D numpy array.
    :param bifurcations: The bifurcations 2D numpy array.
    :param search_best_crossing: Used when following paths. True to keep exploring all valid bifurcations to make sure we choose the best crossing, otherwise we take the first one with detect.
    :param debug: True to show debug logs.
    :return: The ostium location (x, y) and the parent point (x, y) if any to know in which direction to explore after.
    """
    percent_with_one_possible_catheter = -1
    percent_with_two_possible_catheters = -1
    potential_catheters = None
    for i in range(50):
        if percent_with_one_possible_catheter >= 0 and i - percent_with_one_possible_catheter > 25:
            break  # do not check too far after we found the first catheter, but enough so we can check farther than the superposition of the two branches of the catheter
        image_percent = int(skeleton_distances.shape[1] * i / 100)
        current_potential_catheters = np.where(skeleton_distances[image_percent, :] > 0)[0]
        if len(current_potential_catheters) > 1:
            # Remove potential catheters that are the same one (2 consecutive x)
            current_potential_catheters_diffs = [current_potential_catheters[i] - current_potential_catheters[i-1] for i in range(1, len(current_potential_catheters))]
            current_potential_catheters_indices_to_remove = np.array([i for i in range(len(current_potential_catheters_diffs)) if current_potential_catheters_diffs[i] == 1])
            if len(current_potential_catheters_indices_to_remove) > 0:
                mask = np.ones_like(current_potential_catheters, dtype=bool)
                mask[current_potential_catheters_indices_to_remove] = False
                current_potential_catheters = current_potential_catheters[mask, ...]
        if len(current_potential_catheters) > 1:
            if len(current_potential_catheters) > 2:
                potential_catheters_intensity = {}
                for potential_catheter_x in current_potential_catheters:
                    # follow the potential catheters to get their average width and variance
                    starting_point = get_catheter_starting_point(potential_catheter_x, image_percent, skeleton_distances)
                    result_dict = follow_path_bfs(skeleton_distances, bifurcations, starting_point, get_branch_points=True, parent=(potential_catheter_x, image_percent), search_best_crossing=search_best_crossing, debug=debug)
                    path_points = result_dict["branch_points"]
                    ends_on_dead_end = result_dict["dead_end"]
                    path_points = np.array(path_points)
                    if not ends_on_dead_end:
                        last_width = int(round(skeleton_distances[path_points[-1, 1], path_points[-1, 0]]))
                        path_points = path_points[:-last_width]
                    if len(path_points) > 0:
                        intensities = raw_image[path_points[:, 1], path_points[:, 0]]
                        potential_catheters_intensity[potential_catheter_x] = intensities.mean()
                print("intensity", potential_catheters_intensity)
                potential_catheters_intensity = sorted(potential_catheters_intensity.items(), key=operator.itemgetter(1))
                # keep the two potential catheters that have the lower pixel intensity (darker)
                current_potential_catheters = potential_catheters_intensity[0:2]
                current_potential_catheters = [t[0] for t in current_potential_catheters]
                current_potential_catheters.sort()  # put them back in left to right order
            if percent_with_one_possible_catheter >= 0 and i - percent_with_one_possible_catheter > 2 and abs(current_potential_catheters[0] - current_potential_catheters[1]) > 10:
                break  # if the two catheters segments are distanced vertically and horizontally, one of those is probably a vessel
            potential_catheters = current_potential_catheters
            percent_with_two_possible_catheters = i
            break
        if len(current_potential_catheters) == 1 and percent_with_one_possible_catheter < 0:
            percent_with_one_possible_catheter = i
            potential_catheters = current_potential_catheters
    if percent_with_two_possible_catheters >= 0:
        percent = percent_with_two_possible_catheters
    elif percent_with_one_possible_catheter >= 0:
        percent = percent_with_one_possible_catheter
    else:
        print("Cannot find a possible catheter")
        return None, None
    percent = int(skeleton_distances.shape[1] * percent / 100)

    print("potential catheters", potential_catheters)
    ostium_location = None
    ostium_location_parent = None
    catheter_points = []
    catheter_tips = []
    for pixel_x in potential_catheters:
        print(f"\nTrying for potential catheter at ({pixel_x}, {percent})", "only to remove catheter segments" if ostium_location is not None else "")
        starting_point = get_catheter_starting_point(pixel_x, percent, skeleton_distances)
        if starting_point is None:
            print(f"Skipping {pixel_x}")
            continue
        # Get the catheter points to find the tip
        start_time = time.time()
        result_dict = follow_path_bfs(skeleton_distances, bifurcations, starting_point, search_for_ostium=True, get_branch_points=True, allow_crossings=True, parent=(pixel_x, percent), search_best_crossing=search_best_crossing, debug=debug)
        path_points = result_dict["branch_points"]
        catheter_tip = path_points[-1]
        catheter_tips.append((catheter_tip[1], catheter_tip[0]))  # from (x, y) to (y, x)
        if ostium_location is None:
            ostium_location = result_dict["ostium_location"]
            ostium_location_parent = result_dict["ostium_location_parent"]
            if ostium_location is not None:
                path_points = path_points[:-1]  # We don't want to remove ostium point
        catheter_points += path_points
        print(f"follow_path_bfs took {time.time() - start_time}s")
    catheter_points = np.array(catheter_points, dtype=np.int16)

    if ostium_location is not None:
        remove_catheter_points(catheter_points, skeleton_distances, bifurcations)
        return ostium_location, ostium_location_parent

    print("\nCould not find the ostium connected to the catheter, now trying to find a disconnected catheter to deduce the ostium position")
    start_time = time.time()
    # Find the catheter tip that is the closest to the center of the image
    centermost_catheter_tip = ()
    centermost_catheter_tip_percent = ()
    for catheter_tip in catheter_tips:
        dead_end_percent_x = catheter_tip[1] / skeleton_distances.shape[1]
        dead_end_percent_y = catheter_tip[0] / skeleton_distances.shape[0]
        # If the tip of the catheter is closer to the center
        if centermost_catheter_tip == () or abs(dead_end_percent_x - 0.5) + abs(dead_end_percent_y - 0.5) < abs(centermost_catheter_tip_percent[0] - 0.5) + abs(centermost_catheter_tip_percent[1] - 0.5):
            centermost_catheter_tip = catheter_tip
            centermost_catheter_tip_percent = (dead_end_percent_x, dead_end_percent_y)

    if centermost_catheter_tip_percent[1] < 0.6:
        remove_catheter_points(catheter_points, skeleton_distances, bifurcations)
        skeleton_points = np.where(skeleton_distances > 0)
        skeleton_points = np.array([skeleton_points[0], skeleton_points[1]])
        centermost_catheter_tip = np.array(centermost_catheter_tip)
        distances = np.linalg.norm(skeleton_points - centermost_catheter_tip[:, None], axis=0)
        adjusted_distances = distances - skeleton_distances[skeleton_points[0, :], skeleton_points[1, :]] * 2  # Adjusting the distances by removing twice the vessel width
        min_adjusted_distance_idx = adjusted_distances.argmin()
        closest_point = skeleton_points[:, min_adjusted_distance_idx]
        ostium_location = (closest_point[1], closest_point[0])  # (x, y)
        print(f"Finding ostium location from catheter tips took {time.time() - start_time}s")
        return ostium_location, None

    print("Tip of catheter is at the bottom of the image, so we fall back at using the first bifurcation we can find")
    first_bifurcations = []
    for pixel_x in potential_catheters:
        print(f"\nTrying for potential catheter at ({pixel_x}, {percent})")
        starting_point = get_catheter_starting_point(pixel_x, percent, skeleton_distances)
        if starting_point is None:
            print(f"Skipping {pixel_x}")
            continue
        start_time = time.time()
        result_dict = follow_path_bfs(skeleton_distances, bifurcations, starting_point, search_for_ostium=True, get_branch_points=True, allow_crossings=False, parent=(pixel_x, percent), search_best_crossing=search_best_crossing, debug=debug)
        path_points = result_dict["branch_points"]
        dead_end = result_dict["dead_end"]
        if not dead_end and len(path_points) > 0:
            parent = path_points[-2] if len(path_points) > 1 else None
            first_bifurcations.append((path_points[-1], parent))
        print(f"follow_path_bfs took {time.time() - start_time}s")
    if len(first_bifurcations) > 0:
        # We find the first bifurcation that is the closest to the center of the image vertically
        closest_to_center = -1
        closest_distance_to_center = 0
        for i, first_bifurcation in enumerate(first_bifurcations):
            distance_to_center = abs(first_bifurcation[0][1] - skeleton_distances.shape[0] / 2)
            if closest_to_center < 0 or distance_to_center < closest_distance_to_center:
                closest_to_center = i
                closest_distance_to_center = distance_to_center
        remove_catheter_points(catheter_points, skeleton_distances, bifurcations)
        return first_bifurcations[closest_to_center]

    print("ERROR: Could not find the ostium")
    return None, None


def extract_graph_from_skeleton(raw_image, segmented_image, skeleton_distances, bifurcations, ostium, ostium_parent, oriented=False, search_best_crossing=True, show_final_graph=False, debug=False):
    """
    Extracts the skeleton graph by following the skeleton segments starting from the ostium.
    :param raw_image: The 2D array containing the x-ray image (y, x).
    :param segmented_image: The 2D array containing the binarized segmentation (y, x).
    :param skeleton_distances: The 2D array containing the skeleton distances (y, x).
    :param bifurcations: The 2D array containing the bifurcations (y, x).
    :param ostium: The location of the ostium point (x, y).
    :param ostium_parent: The location of the point parent of the ostium (x, y).
    :param oriented: True to extract the graph in an oriented fashion, starting from the ostium.
    :param search_best_crossing: Used when following paths. True to keep exploring all valid bifurcations to make sure we choose the best crossing, otherwise we take the first one with detect.
    :param show_final_graph: True to show final graph with lines on the segmented image.
    :param debug: True to show debug logs.
    :return: The adjacent nodes (dict composed of node ids as keys and lists of node ids as values),
             the node features (dict composed of node ids as keys and an array of [position_x, position_y, vessel_width, pixel_intensity, is_ostium] as values),
             and the edge features (dict composed of tuples of (id of node A, id of node B) as keys and an array of [vessel_length, average_vessel_width, average_pixel_intensity] as values).
    """
    start_time = time.time()

    # Creating background image for visualization
    binarized_skeleton = skeleton_distances.copy()
    binarized_skeleton[binarized_skeleton > 0] = 255
    background_image = segmented_image + binarized_skeleton

    nodes = [ostium]
    adjacent_nodes = {0: []}
    all_edge_features = {}  # (node_a, node_b): (vessel_length, average_vessel_width, average_pixel_intensity)
    all_node_features = {}  # node: (position_x, position_y, vessel_width, pixel_intensity, is_ostium)
    unexplored_paths = []  # heap queue containing tuples (-bifurcation_width, path_start_point, parent_point, parent_node_index)
    explored_paths = []  # a list of points representing the first and penultimate points of explored paths
    all_node_features[0] = get_node_features(ostium, 0, skeleton_distances, raw_image)

    # Adding branches that start from the ostium
    branches = get_branches(ostium[::-1], skeleton_distances, debug=False)
    for branch in branches:
        branch_point = (ostium[0] + branch[1], ostium[1] + branch[0])
        if ostium_parent != branch_point:
            ostium_branch_width = skeleton_distances[branch_point[1], branch_point[0]]
            heapq.heappush(unexplored_paths, (-ostium_branch_width, branch_point, ostium, 0))

    step = 0
    # Keep exploring while we still have branches that we didn't go through
    while len(unexplored_paths) > 0:
        _, start_point, parent_point, parent_node_index = heapq.heappop(unexplored_paths)
        if oriented and start_point in explored_paths:
            # print("----- PATH ALREADY EXPLORED!!! (1)")
            continue
        # print("following path", start_point, "from parent point", parent_point, "of node", parent_node_index)
        result_dict = follow_path_bfs(skeleton_distances, bifurcations, start_point, parent=parent_point, get_branch_points=True, allow_crossings=oriented, search_best_crossing=search_best_crossing, debug=debug)
        path_points = result_dict["branch_points"]
        end_point = path_points[-1]
        end_point_parent = path_points[-2]
        # print("end_point", end_point)
        if oriented and end_point_parent in explored_paths:
            # print("----- PATH ALREADY EXPLORED!!! (2)")
            continue

        # If there is already an edge between those two nodes and we are creating an oriented graph, we want to skip it
        already_explored_node = end_point in nodes
        already_existing_connection = False
        if oriented and already_explored_node:
            parent_node_indices = [i for i, x in enumerate(nodes) if x == parent_point]
            end_node_indices = [i for i, x in enumerate(nodes) if x == end_point]
            for parent_node_id in parent_node_indices:
                for end_node_id in end_node_indices:
                    if end_node_id in adjacent_nodes[parent_node_id] or parent_node_id in adjacent_nodes[end_node_id]:
                        # print("----- THERE IS ALREADY A CONNECTION!!!")
                        already_existing_connection = True
                        break
                if already_existing_connection:
                    break
            if oriented and already_existing_connection:
                continue

        # Save the new connection to prevent other paths from ending on it
        explored_paths.append(start_point)
        explored_paths.append(end_point_parent)
        # If we are creating an oriented graph or if the node is new, we want to add it to the list of nodes
        if oriented or not already_explored_node:
            nodes.append(end_point)
            end_point_index = len(nodes) - 1
            adjacent_nodes[end_point_index] = []
            all_node_features[end_point_index] = get_node_features(end_point, end_point_index, skeleton_distances, raw_image)
        # If the branch ended on an existing point and we are creating a non oriented graph, we connect the two nodes together
        else:
            end_point_index = nodes.index(end_point)
            adjacent_nodes[end_point_index].append(parent_node_index)
        adjacent_nodes[parent_node_index].append(end_point_index)

        edge_features = get_edge_features(path_points, skeleton_distances, raw_image)
        all_edge_features[(parent_node_index, end_point_index)] = edge_features
        if not oriented:
            all_edge_features[(end_point_index, parent_node_index)] = edge_features

        is_bifurcation = bifurcations[end_point[1], end_point[0]] > 0
        # If the new node is a bifurcation, we need to add the other branches to the list of unexplored paths
        if not already_explored_node and is_bifurcation:
            branches = get_branches(end_point[::-1], skeleton_distances, debug=False)
            for branch in branches:
                branch_point = (end_point[0] + branch[1], end_point[1] + branch[0])
                if end_point_parent != branch_point:
                    heapq.heappush(unexplored_paths, (-skeleton_distances[branch_point[1], branch_point[0]], branch_point, end_point, nodes.index(end_point)))
                    # print("Adding branch", branch)

        if debug:
            adjacency_matrix = np.zeros((len(nodes), len(nodes)))
            plot_title = ("" if oriented else "Non-") + "Oriented Graph Reconstruction step " + str(step)
            plt.title(plot_title)
            plt.imshow(background_image)
            for edge_index, adjacent_edges_list in adjacent_nodes.items():
                adjacency_matrix[edge_index, adjacent_edges_list] = 1
                if not oriented:
                    adjacency_matrix[adjacent_edges_list, edge_index] = 1
                for adjacent_edge in adjacent_edges_list:
                    plt.plot([nodes[edge_index][0], nodes[adjacent_edge][0]], [nodes[edge_index][1], nodes[adjacent_edge][1]])
                    if oriented:
                        point_position_x = nodes[adjacent_edge][0] - (nodes[adjacent_edge][0] - nodes[edge_index][0]) * 0.05
                        point_position_y = nodes[adjacent_edge][1] - (nodes[adjacent_edge][1] - nodes[edge_index][1]) * 0.05
                        plt.scatter(point_position_x, point_position_y, s=10)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()

        step += 1

    if debug:
        print(f"follow_path_bfs took {time.time() - start_time}s")
        print("nodes", nodes)
        print("adjacent_nodes", adjacent_nodes)
        print("all_node_features", all_node_features)
        print("all_edge_features", all_edge_features)

    if show_final_graph:
        plot_title = ("" if oriented else "Non-") + "Oriented Graph Reconstruction"
        plt.title(plot_title)
        plt.imshow(background_image)

    # Create the adjacency matrix
    # adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    # for edge_index, adjacent_edges_list in adjacent_nodes.items():
    #     adjacency_matrix[edge_index, adjacent_edges_list] = 1
    #     if not oriented:
    #         adjacency_matrix[adjacent_edges_list, edge_index] = 1
    #
    #     if show_final_graph:
    #         # Plot the edges
    #         for adjacent_edge in adjacent_edges_list:
    #             plt.plot([nodes[edge_index][0], nodes[adjacent_edge][0]], [nodes[edge_index][1], nodes[adjacent_edge][1]])
    #             if oriented:
    #                 point_position_x = nodes[adjacent_edge][0] - (nodes[adjacent_edge][0] - nodes[edge_index][0]) * 0.05
    #                 point_position_y = nodes[adjacent_edge][1] - (nodes[adjacent_edge][1] - nodes[edge_index][1]) * 0.05
    #                 plt.scatter(point_position_x, point_position_y, s=10)

    if show_final_graph:
        plt.show()

        # plt.title("Adjacency matrix")
        # plt.imshow(adjacency_matrix)
        # plt.show()

    return adjacent_nodes, all_node_features, all_edge_features


def get_node_features(end_point, end_point_index, skeleton_distances, raw_image):
    position_x = end_point[0] / skeleton_distances.shape[1]
    position_y = end_point[1] / skeleton_distances.shape[0]
    vessel_width = skeleton_distances[end_point[1], end_point[0]]
    pixel_intensity = raw_image[end_point[1], end_point[0]] / 255
    is_ostium = (end_point_index == 0) * 1
    return np.array([position_x, position_y, vessel_width, pixel_intensity, is_ostium])


def get_edge_features(path_points, skeleton_distances, raw_image):
    vessel_length = len(path_points)  # TODO make sure all path points are in that list
    average_vessel_width = get_average_vessel_width(path_points, skeleton_distances, real_average=True, debug=False)
    path_points = np.array(path_points)
    intensities = raw_image[path_points[:, 1], path_points[:, 0]]
    average_pixel_intensity = intensities.mean() / 255
    return np.array([vessel_length, average_vessel_width, average_pixel_intensity])


def remove_catheter_points(catheter_points, skeleton, bifurcations):
    # We find the bifurcations from the catheter points
    bifurcation_catheter_points = catheter_points[bifurcations[catheter_points[:, 1], catheter_points[:, 0]] > 0]
    # We remove bifurcation points from catheter_points
    catheter_points = catheter_points[np.where(bifurcations[catheter_points[:, 1], catheter_points[:, 0]] == 0)]
    # We remove bifurcation points from bifurcations
    bifurcations[bifurcation_catheter_points[:, 1], bifurcation_catheter_points[:, 0]] = 0
    # We remove the catheter points from the skeleton
    skeleton[catheter_points[:, 1], catheter_points[:, 0]] = 0


def get_catheter_starting_point(x, y, skeleton_distances):
    if skeleton_distances[y, x] > 0:  # Bottom
        starting_point = (x, y+1)
    elif skeleton_distances[y+1, x-1] > 0:  # Bottom left
        starting_point = (x-1, y+1)
    elif skeleton_distances[y+1, x+1] > 0:  # Bottom right
        starting_point = (x+1, y+1)
    elif skeleton_distances[y, x-1] > 0:  # Left
        starting_point = (x-1, y)
    elif skeleton_distances[y, x+1] > 0:  # Right
        starting_point = (x+1, y)
    else:  # No valid points to do the pathing
        print(f"No valid points to do the pathing for ({x}, {y})")
        return None
    return starting_point


class PathNode:

    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


class CrossingParams:
    """
    This class contains the necessary information to deduce a crossing in the coronary tree skeleton from bifurcations.
    """
    def __init__(self, bifurcation_node=None, angle=0.0, vessel_width=0.0, image_size=1024, crossing_params=None):
        """
        Constructor of the CrossingParams class.
        :param bifurcation_node: The node from which the crossing would start.
        :param angle: The angle in degrees of the vessel (0 degree is when the vessel points completely towards the right).
        :param vessel_width: Size of the vessel to help with the crossing identification.
        :param image_size: Height of the image. Used to scale the constants.
        :param crossing_params: A CrossingParams object to copy. If set, will ignore the other parameters.
        """
        if crossing_params is None:
            self.bifurcation_node = bifurcation_node
            self.angle = angle
            self.vessel_width = vessel_width
            self.image_size = image_size
            self.explored_bifurcations = []
        else:
            self.bifurcation_node = crossing_params.bifurcation_node
            self.angle = crossing_params.angle
            self.vessel_width = crossing_params.vessel_width
            self.image_size = crossing_params.image_size
            self.explored_bifurcations = crossing_params.explored_bifurcations.copy()

    def get_distance_and_angle_diff(self, bifurcation):
        vector = np.array([bifurcation[1] - self.bifurcation_node.x, bifurcation[0] - self.bifurcation_node.y])  # [x, y]
        distance = np.sqrt(np.square(vector).sum())
        angle = get_angle_from_vector(vector, previous_angle=self.angle)
        angle_diff = abs(angle - self.angle)
        return distance, angle_diff

    def is_bifurcation_valid(self, bifurcation, indentation_level, debug):
        """
        Checks if a bifurcation is valid for these crossing params depending on the distance and angle of the bifurcation.
        :param bifurcation: A point (y, x).
        :param indentation_level: The level of indentation to improve logs readability.
        :param debug: This method will print logs only if that parameter is True.
        :return: True if the bifurcation is within 45 degrees and 100 pixels.
        """
        MAX_CROSSING_DISTANCE = get_max_crossing_distance(self.image_size)
        # Invalid if already explored
        if bifurcation in self.explored_bifurcations:
            print_indented_log(indentation_level, f"Bifurcation {bifurcation} already explored {self.explored_bifurcations})", debug)
            return False
        # Invalid if too far
        vector = np.array([bifurcation[1] - self.bifurcation_node.x, bifurcation[0] - self.bifurcation_node.y])  # [x, y]
        distance = np.sqrt(np.square(vector).sum())
        print_indented_log(indentation_level, f"Bifurcation {bifurcation} has a vector {vector} and distance of {distance} pixels", debug)
        if distance > MAX_CROSSING_DISTANCE:
            print_indented_log(indentation_level, f"Bifurcation {bifurcation} is too far ({distance} pixels)", debug)
            return False
        # Invalid if not in the right direction
        angle = get_angle_from_vector(vector, previous_angle=self.angle)
        maximum_angle = 90 - 75 * distance / MAX_CROSSING_DISTANCE
        print_indented_log(indentation_level, f"Maximum angle = {maximum_angle}, current = {abs(angle - self.angle)}", debug)
        if abs(angle - self.angle) > maximum_angle:
            print_indented_log(indentation_level, f"Bifurcation {bifurcation} is not in the right direction (at {angle} degrees for a {abs(angle - self.angle)} degrees difference)", debug)
            return False
        # Valid otherwise
        return True


def get_max_crossing_distance(image_size):
    return 100 * image_size / 1024


def follow_path_bfs(skeleton_distances, bifurcations, starting_point, parent=None, search_for_ostium=False, search_for_crossing=False, get_branch_points=False, allow_crossings=False, allow_recursion=True, search_best_crossing=True, default_log_indentation=0, debug=False):
    """
    Follows a path on the skeleton until it reaches a bifurcation or dead-end (or sudden change in vessel width or angle when searching for the ostium).
    :param skeleton_distances: A 2D ndarray containing the skeleton with distances.
    :param bifurcations: A 2D ndarray containing the points of bifurcations on the skeleton.
    :param starting_point: A tuple of indices (x, y) representing the position in the skeleton on which we want to start pathing.
    :param parent: A tuple of indices (x, y) representing the position in the skeleton from which the starting point was chosen.
    It basically allows the algorithm to know in what direction to start pathing.
    :param search_for_ostium: A bool representing if we are searching for the ostium (by analysing the vessel width and angle).
    :param search_for_crossing: A bool representing if we are searching for a crossing (will stop as soon as the first crossing is found).
    :param get_branch_points: A bool representing if we want to simply follow the path to retrieve the branch points. This mode won't check for crossings unless allow_crossings is True.
    :param allow_crossings: Used only when get_branch_points is also True. It is a bool that defines if we want to stop at the first bifurcation or allow crossings.
    :param allow_recursion: It is a bool that defines if we should check the other branch has a crossing when we detect a crossing on the first bifurcation.
    :param search_best_crossing: True to keep exploring all valid bifurcations to make sure we choose the best crossing, otherwise we take the first one with detect.
    :param default_log_indentation: The default number of indentations when printing logs.
    :param debug: A bool representing if we want to print debug logs or not.
    :return: The return value is different depending on the parameters.
    - If search_for_crossing is True, returns True if the first bifurcation encountered is a crossing, otherwise False.
    - If search_for_ostium or get_branch_points is True, returns a dict with multiple values:
        - "ostium_location": the location (x, y) of the ostium if found, otherwise None. Present only if search_for_ostium is True.
        - "ostium_location_parent": the location (x, y) of the parent of the ostium if found, otherwise None. Present only if search_for_ostium is True.
        - "branch_points": the list of points (x, y) encountered. Present only if get_branch_points is True.
        - "dead_end": a bool that tells if the path ended on a dead-end. Present only if get_branch_points is True.
    """
    check_for_crossings = not get_branch_points or allow_crossings
    parent_node = None if parent is None else PathNode(parent[0], parent[1])
    current_node = PathNode(starting_point[0], starting_point[1], parent_node)
    paths_to_explore = []
    crossing_params = None
    parent_path_points_key = None
    heapq.heappush(paths_to_explore, (0, 0, np.random.ranf(), current_node, crossing_params, parent_path_points_key))  # push items to the priority queue by adding a priority + tie breaker and crossing params alongside the node
    explored_bifurcations = []
    path_points = [(parent_node.x, parent_node.y)]
    all_paths_points = {}  # (bifurcation (y, x), branch): points (x, y)
    parent_paths = {}  # bifurcation (y, x): (parent bifurcation (y, x), branch)
    ends_on_dead_end = False
    if search_best_crossing:
        best_crossing = None  # (crossing_score, current_node, path_points_key, last_segment_path_points)
    else:
        first_bifurcation_crossing_info = None
    image_size = skeleton_distances.shape[0]
    while len(paths_to_explore) > 0:  # We still have skeleton segments to explore
        priority, branch_number, random_value, current_node, crossing_params, parent_path_points_key = heapq.heappop(paths_to_explore)
        print_indented_log(default_log_indentation + (0 if crossing_params is None else 1), f"-> Popping node ({current_node.x}, {current_node.y}) of priority {priority} and branch number {branch_number}, {len(paths_to_explore)} remaining", debug)
        if crossing_params is not None:
            print_indented_log(default_log_indentation + 1, f"crossing params values: angle={crossing_params.angle}, vessel_width={crossing_params.vessel_width}, explored_bifurcations={crossing_params.explored_bifurcations}", debug)
        path_points_key = ((current_node.parent.y, current_node.parent.x), branch_number)
        all_paths_points[path_points_key] = []
        nodes_in_path = 0
        while True:  # Keep advancing on the skeleton segment, we break when reaching a bifurcation or a dead end
            if crossing_params is None:
                if (current_node.x, current_node.y) in path_points:
                    print(f"{(current_node.x, current_node.y)} is already in path_points")
                path_points.append((current_node.x, current_node.y))
            all_paths_points[path_points_key].append((current_node.x, current_node.y))
            nodes_in_path += 1
            indentation_level = default_log_indentation + (0 if crossing_params is None else 1)
            is_bifurcation = bifurcations[current_node.y, current_node.x] > 0
            check_for_crossing = check_for_crossings and crossing_params is not None and (nodes_in_path == 20 or (nodes_in_path < 20 and is_bifurcation))
            if is_bifurcation or check_for_crossing:
                # compute the direction vector on the last few points to get the vessel angle
                bifurcation_index = -1 if is_bifurcation else 0
                bifurcation_width = skeleton_distances[current_node.y, current_node.x] if is_bifurcation else skeleton_distances[path_points_key[0][0], path_points_key[0][1]]
                vector, vessel_width = get_direction_vector_and_vessel_width_from_end_node(current_node, skeleton_distances, bifurcation_index, bifurcation_width)
                previous_angle = float('inf') if crossing_params is None else crossing_params.angle
                angle = get_angle_from_vector(vector, previous_angle, debug=False)
                print_indented_log(indentation_level, f"computed an angle of {angle} degrees from {nodes_in_path} nodes with vector {vector} and a vessel width of {vessel_width}", debug)
                if check_for_crossing:
                    angle_diff = abs(angle - crossing_params.angle)
                    vessel_width_ratio = max(crossing_params.vessel_width, vessel_width) / min(crossing_params.vessel_width, vessel_width)
                    vessel_width_diff = abs(crossing_params.vessel_width - vessel_width)
                    print_indented_log(indentation_level, f"angle difference of {angle_diff} degrees, vessel width {crossing_params.vessel_width} -> {vessel_width} ({int(vessel_width_ratio*1000)/10}%)", debug)
                    max_angle_diff = 30 if search_for_ostium else 25
                    max_vessel_width_ratio = 1.45 if search_for_ostium else 1.35
                    min_vessel_width_diff = 0 * image_size / 1024  # In case we want to allow a big vessel width ratio when the difference is small (ex. vessels of width 2 and 4)
                    if angle_diff <= max_angle_diff and (vessel_width_ratio <= max_vessel_width_ratio or vessel_width_diff <= min_vessel_width_diff):  # Thresholds are higher for the catheter since we don't want to have false negatives
                        is_crossing = True
                        # If the crossing was identified on the first bifurcation, it is likely not a crossing, unless we are following a catheter. We need to check if we can find a crossing on the other branch first.
                        if len(crossing_params.explored_bifurcations) == 1 and not search_for_ostium:
                            # If we are already checking for a crossing in the other branch, we don't want to detect a crossing on the same branch we found the potential crossing, we need to check farther
                            if search_for_crossing:
                                is_crossing = False
                            elif allow_recursion:
                                print_indented_log(indentation_level, "We would have found a crossing, but it is the first bifurcation... Now searching for a crossing from the other bifurcation branch.", debug)
                                # Backtracking to the bifurcation we think might be a crossing
                                backtrack_node = current_node.parent
                                previous_node = current_node
                                while bifurcations[backtrack_node.y, backtrack_node.x] == 0:
                                    previous_node = backtrack_node
                                    backtrack_node = backtrack_node.parent
                                bifurcation = (backtrack_node.y, backtrack_node.x)
                                print_indented_log(indentation_level, f"Backtracked from {(current_node.x, current_node.y)} to {(backtrack_node.x, backtrack_node.y)}", debug)
                                # Get the branches and find the one that was not taken
                                branches = get_branches(bifurcation, skeleton_distances)
                                for b, (branch_y, branch_x) in enumerate(branches):
                                    branch_point = (bifurcation[1] + branch_x, bifurcation[0] + branch_y)
                                    if branch_point[0] == previous_node.x and branch_point[1] == previous_node.y:
                                        continue  # Skip the current branch
                                    if branch_point[0] == crossing_params.bifurcation_node.parent.x and branch_point[1] == crossing_params.bifurcation_node.parent.y:
                                        continue  # Skip the parent branch
                                    # The remaining branch is the branch of the vessel that we want to check if it crosses the one we just followed
                                    # We now follow it to its extremity and starts the path following towards the bifurcation we want to identify as a crossing or bifurcation
                                    branch_node = PathNode(branch_point[0], branch_point[1], parent=backtrack_node)
                                    while bifurcations[branch_node.y, branch_node.x] == 0:
                                        next_point = get_next_path_point(branch_node, skeleton_distances, bifurcations)
                                        if next_point is None:
                                            break
                                        branch_node = PathNode(branch_node.x + next_point[0] - 1, branch_node.y + next_point[1] - 1, parent=branch_node)
                                    print_indented_log(indentation_level, f"Checking on branch with starting point {(branch_node.parent.x, branch_node.parent.y)}", debug)
                                    # Check if the vessel has a crossing
                                    is_crossing = follow_path_bfs(skeleton_distances, bifurcations, starting_point=(branch_node.parent.x, branch_node.parent.y), parent=(branch_node.x, branch_node.y), search_for_crossing=True, allow_recursion=False, search_best_crossing=search_best_crossing, default_log_indentation=default_log_indentation+2, debug=debug)
                                    print_indented_log(indentation_level, f"Is first the bifurcation a crossing? {is_crossing}", debug)
                        if is_crossing:
                            if search_for_crossing:
                                return True  # We only wanted to check if we could find a crossing, so we can return already
                            indentation_level -= 1
                            if search_best_crossing:
                                path_start_point = path_points_key[0]  # (y, x)
                                crossing_distance = np.sqrt((path_start_point[1] - crossing_params.bifurcation_node.x) ** 2 + (path_start_point[0] - crossing_params.bifurcation_node.y) ** 2)
                                max_crossing_distance = get_max_crossing_distance(image_size)
                                angle_score = 1 - angle_diff / max_angle_diff
                                vessel_width_ratio_score = max(1 - vessel_width_ratio / max_vessel_width_ratio, 0)
                                crossing_distance_score = 1 - crossing_distance / max_crossing_distance
                                crossing_score = angle_score + vessel_width_ratio_score + crossing_distance_score
                                if best_crossing is None or crossing_score > best_crossing[0]:
                                    print_indented_log(indentation_level, f"Crossing found at ({path_start_point[1]}, {path_start_point[0]}) with a score of {angle_score} + {vessel_width_ratio_score} + {crossing_distance_score} = {crossing_score}" + (" and beats the previous one" if best_crossing is not None else ""), debug)
                                    best_crossing = (crossing_score, current_node, path_points_key, all_paths_points[path_points_key].copy())
                            else:
                                if len(crossing_params.explored_bifurcations) == 1:
                                    print_indented_log(indentation_level, f"Saving current node, path points key and explored bifurcations in case we find no crossing after more than 1 bifurcation", debug)
                                    first_bifurcation_crossing_info = (current_node, path_points_key, explored_bifurcations.copy())
                                else:
                                    print_indented_log(indentation_level, "We have found a crossing, now clearing the priority queue", debug)
                                    if first_bifurcation_crossing_info is not None:
                                        print_indented_log(indentation_level, "Also clearing the first bifurcation crossing info", debug)
                                        first_bifurcation_crossing_info = None
                                    # Add the path points of the previously explored paths based on the bifurcations
                                    parent_bifurcation, branch = path_points_key
                                    # crossing_path_points = list(reversed(all_paths_points[(parent_bifurcation, branch)]))
                                    # print(f"Need to find crossing params bifurcation ({crossing_params.bifurcation_node.y}, {crossing_params.bifurcation_node.x})")
                                    # print(f"all_paths_points: {all_paths_points}")
                                    # print(f"parent_bifurcation and branch: {path_points_key}")
                                    # print(f"parent_paths: {parent_paths}")
                                    # TODO find why this can cause an infinite loop for 2459788_LCA_0_0
                                    # while parent_bifurcation != (crossing_params.bifurcation_node.y, crossing_params.bifurcation_node.x):
                                    #     parent_bifurcation, branch = parent_paths[parent_bifurcation]
                                    #     path_points_to_add = list(reversed(all_paths_points[(parent_bifurcation, branch)]))
                                    #     crossing_path_points += path_points_to_add
                                    # path_points += list(reversed(crossing_path_points))
                                    path_points.append((parent_bifurcation[1], parent_bifurcation[0]))
                                    path_points += all_paths_points[(parent_bifurcation, branch)]
                                    crossing_params = None
                                    paths_to_explore.clear()
                        else:
                            # We followed the other branch of the bifurcation and found no crossing, so we need to stop looking
                            is_bifurcation = True
                            crossing_params.explored_bifurcations.append((current_node.y, current_node.x))
                            paths_to_explore.clear()
                if is_bifurcation:
                    bifurcation = (current_node.y, current_node.x)
                    if bifurcation not in explored_bifurcations:
                        if crossing_params is None:
                            explored_bifurcations.append(bifurcation)
                        parent_paths[bifurcation] = path_points_key
                        if check_for_crossings and (crossing_params is None or crossing_params.is_bifurcation_valid(bifurcation, indentation_level, debug)):
                            if search_for_ostium and crossing_params is None:
                                nodes, vessel_widths = get_nodes_and_vessel_width(current_node, skeleton_distances)
                                sharp_angle_location, parent_location = get_location_of_sharp_angle(nodes, image_size=image_size, show_graph_only_on_detection=debug)
                                if sharp_angle_location is not None:
                                    print_indented_log(indentation_level, f"sharp angle location: {sharp_angle_location}", debug)
                                    return_dict = {"ostium_location": sharp_angle_location, "ostium_location_parent": parent_location}
                                    if get_branch_points:
                                        while path_points[-1] != sharp_angle_location:
                                            path_points.pop(len(path_points) - 1)
                                        return_dict["branch_points"] = path_points
                                        return_dict["dead_end"] = False
                                    return return_dict
                                widening_location, parent_location = get_location_of_widening(nodes, vessel_widths, image_size=image_size, show_graph_only_on_detection=debug)
                                if widening_location is not None:
                                    # Instead of returning already, we must check if there is a crossing. In that case, the widening might just be caused by the crossing
                                    bifurcation_widening_dist = np.sqrt((widening_location[1] - bifurcation[0]) ** 2 + (widening_location[0] - bifurcation[1]) ** 2)
                                    print_indented_log(indentation_level, f"widening location: {widening_location}, bifurcation width is {skeleton_distances[bifurcation]} and dist is {bifurcation_widening_dist}", debug)
                                    if skeleton_distances[bifurcation] < bifurcation_widening_dist:
                                        return_dict = {"ostium_location": widening_location, "ostium_location_parent": parent_location}
                                        if get_branch_points:
                                            while path_points[-1] != widening_location:
                                                path_points.pop(len(path_points) - 1)
                                            return_dict["branch_points"] = path_points
                                            return_dict["dead_end"] = False
                                        return return_dict
                            bifurcation_node = PathNode(bifurcation[1], bifurcation[0])
                            # add bifurcation branches to the priority queue
                            branches = get_branches(bifurcation, skeleton_distances)
                            print_indented_log(indentation_level, f"Adding branches of bifurcation {bifurcation}", debug)
                            for b, (branch_y, branch_x) in enumerate(branches):
                                branch_point = (current_node.x + branch_x, current_node.y + branch_y)
                                if branch_point[0] != current_node.parent.x or branch_point[1] != current_node.parent.y:
                                    if crossing_params is None:
                                        new_crossing_params = CrossingParams(bifurcation_node=current_node, angle=angle, vessel_width=vessel_width, image_size=image_size)
                                        priority = 0
                                    else:
                                        new_crossing_params = CrossingParams(crossing_params=crossing_params)  # Copy
                                        distance, angle_diff = new_crossing_params.get_distance_and_angle_diff(bifurcation)
                                        priority = distance * angle_diff
                                    print_indented_log(indentation_level, f"new branch {b} ({branch_x}, {branch_y}) with priority {priority}", debug)
                                    new_crossing_params.explored_bifurcations.append(bifurcation)
                                    new_node = PathNode(branch_point[0], branch_point[1], parent=bifurcation_node)
                                    heapq.heappush(paths_to_explore, (priority, b, np.random.ranf(), new_node, new_crossing_params, path_points_key))
                                else:
                                    print_indented_log(indentation_level, f"branch {b} ({branch_x}, {branch_y}) is the parent", debug)
                    else:
                        print_indented_log(indentation_level, f"bifurcation {bifurcation} has already been explored", debug)
                    break  # finished on a bifurcation

            next_point = get_next_path_point(current_node, skeleton_distances, bifurcations, debug)

            if next_point is None:
                print_indented_log(indentation_level, f"Dead end", debug)
                ends_on_dead_end = True
                break  # finished on a dead end

            current_node = PathNode(current_node.x + next_point[0] - 1, current_node.y + next_point[1] - 1, current_node)

        if search_best_crossing:
            if len(paths_to_explore) == 0 and best_crossing is not None:
                print_indented_log(indentation_level, "We have found the best crossing, now continuing the path following", debug)
                crossing_params = None
                node = best_crossing[1]
                path_points_key = best_crossing[2]
                explored_segment_path_points = best_crossing[3]
                best_crossing = None
                heapq.heappush(paths_to_explore, (0, 0, np.random.ranf(), node, crossing_params, path_points_key))
                parent_bifurcation, branch = path_points_key
                path_points.append((parent_bifurcation[1], parent_bifurcation[0]))
                path_points += explored_segment_path_points[:-1]
        else:
            # If there are no more paths to explore and we still have a first bifurcation crossing info, that means we found no multi bifurcation crossing, so we accept the first bifurcation crossing
            if len(paths_to_explore) == 0 and first_bifurcation_crossing_info is not None:
                print_indented_log(indentation_level, "No more path to explore, we accept the first bifurcation crossing", debug)
                crossing_params = None
                node = first_bifurcation_crossing_info[0]
                path_points_key = first_bifurcation_crossing_info[1]
                explored_bifurcations = first_bifurcation_crossing_info[2]
                heapq.heappush(paths_to_explore, (0, 0, np.random.ranf(), node, crossing_params, path_points_key))
                first_bifurcation_crossing_info = None

    if get_branch_points or search_for_ostium:
        return_dict = {}
        if search_for_ostium:
            # We haven't found a crossing for a bifurcation. We then suppose the ostium is that bifurcation.
            return_dict["ostium_location"] = (crossing_params.bifurcation_node.x, crossing_params.bifurcation_node.y) if crossing_params is not None else None
            return_dict["ostium_location_parent"] = (crossing_params.bifurcation_node.parent.x, crossing_params.bifurcation_node.parent.y) if crossing_params is not None else None
        if get_branch_points:
            return_dict["branch_points"] = path_points
            return_dict["dead_end"] = ends_on_dead_end  # There is a bug that makes it so that value is the one from the last explored branch instead of the bifurcation if both paths end in dead end
        return return_dict

    if search_for_crossing:
        return False

    return None


def get_next_path_point(current_node, skeleton_distances, bifurcations, debug=False):
    """
    Searches for the points around the current node for the next point in the path.
    :param current_node: The current node of the path finding.
    :param skeleton_distances: The skeleton 2D array with distances.
    :param bifurcations: The 2D array containing the bifurcations.
    :return: A point (x, y) in local space (where 0 <= x <= 2, same for y) indicating where the next point in path is compared to the current point.
    """
    # Find the next potential nodes
    parent_is_bifurcation = current_node.parent is not None and bifurcations[current_node.parent.y, current_node.parent.x] > 0
    pad_top = current_node.y == 0
    pad_bottom = current_node.y == skeleton_distances.shape[0] - 1
    pad_left = current_node.x == 0
    pad_right = current_node.x == skeleton_distances.shape[1] - 1
    patch = np.zeros((3, 3))
    top_index = 1 if pad_top else 0
    bottom_index = 2 if pad_bottom else 3
    left_index = 1 if pad_left else 0
    right_index = 2 if pad_right else 3
    patch[top_index:bottom_index, left_index:right_index] = skeleton_distances[current_node.y-(1-top_index):current_node.y+bottom_index-1, current_node.x-(1-left_index):current_node.x+right_index-1]  # 3x3 around the current point
    # if debug:
    #     plt.title(f"Current node ({current_node.x}, {current_node.y}) of width {skeleton_distances[current_node.y, current_node.x]}")
    #     plt.imshow(patch)
    #     plt.show()
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
                    if parent_is_bifurcation and abs(current_node.x + x - 1 - current_node.parent.x) + abs(current_node.y + y - 1 - current_node.parent.y) == 1:
                        continue  # skip point next to bifurcation parent
                potential_points.append((x, y))

    if len(potential_points) == 0:
        return None

    next_point = potential_points[0]
    if len(potential_points) > 1:  # there was 2 new pixels next to the current point
        # we find the closest one (the one that is not in diagonal)
        # adjacents are: (0,1), (1,0), (1,2), (2,1)
        # diagonals are: (0,0), (2,0), (0,2), (2,2)
        for potential_point in potential_points:
            if abs(potential_point[0] - potential_point[1]) == 1:
                next_point = potential_point
                break

    return next_point


def print_indented_log(indentation_level, text, debug):
    if debug:
        print(f"{'    ' * indentation_level}{text}")


def get_direction_vector_and_vessel_width_from_end_node(end_node, skeleton_distances, bifurcation_index=-1, bifurcation_width=0, max_node_count=20):
    """
    This function returns a direction vector and the vessel width from the end node of the segment by weighting the vectors between each point.
    The weighting is done by giving less importance to vectors between points that are close to the bifurcation (closeness is defined by the bifurcation width).
    :param end_node: The last node we encountered.
    :param skeleton_distances: The skeleton distances matrix.
    :param bifurcation_index: 0 (first) or -1 (last).
    :param bifurcation_width: The size of the bifurcation (helps with defining which vectors should have less importance).
    :param max_node_count: The number of nodes we want to consider before stopping checking farther back.
    :return: The direction vector and the list of points.
    """
    vectors = []
    vector = np.array([0, 0])
    points = []
    backtrack_node = end_node
    while backtrack_node is not None and len(points) < max_node_count:
        point = np.array([backtrack_node.x, backtrack_node.y])
        if len(points) > 0:
            vectors.append(points[-1] - point)
            vector += points[-1] - point
        points.append(point)
        backtrack_node = backtrack_node.parent
    points = np.array(list(reversed(points)))
    vessel_width = get_average_vessel_width(points, skeleton_distances)
    vectors = np.array(list(reversed(vectors)), dtype=np.float)
    index = max(0, int(np.floor(bifurcation_width - vessel_width)))
    if bifurcation_index == 0:
        vectors[:index] *= 0.25
    else:
        vectors[-index:] *= 0.25
    weighted_vector = vectors.sum(axis=0)
    weighted_vector = weighted_vector / np.linalg.norm(weighted_vector)
    vector = vector / np.linalg.norm(vector)
    # return vector, points
    return weighted_vector, vessel_width


def get_average_vessel_width(points, skeleton_distances, real_average=False, debug=False):
    """
    Returns the vessel width that is calculated from the 25th percentile of the skeleton distances of the points, unless the real_average parameter is True.
    :param points: The pixels of the vessel.
    :param skeleton_distances: The skeleton distances.
    :param real_average: Instead of returning the 25th percentile, it returns the mean.
    :param debug: True to print debug logs.
    :return: 25th percentile of vessel width.
    """
    if len(points) == 0:
        return 0
    vessel_widths = []
    vessel_width = 0
    for point in points:
        width = skeleton_distances[point[1], point[0]]
        if debug:
            print("point", point, "has a width of", width)
        vessel_widths.append(width)
        vessel_width += width
    if real_average:
        return vessel_width / len(points)
    vessel_widths = sorted(vessel_widths)
    quarter = round(len(points) / 4)
    return vessel_widths[quarter]


def get_nodes_and_vessel_width(end_node, skeleton_distances):
    """
    This method returns the nodes and their associated gaussian filtered width. Nodes near the bifurcations are ignored.
    :param end_node: The node from which we start to go backwards to get all of them.
    :param skeleton_distances: The skeleton distances matrix.
    :return: The list of nodes (without nodes near the bifurcations) and their associated gaussian filtered width.
    """
    current_node = end_node
    nodes = [current_node]
    distances = []
    while current_node is not None:
        distances.append(skeleton_distances[current_node.y, current_node.x])
        current_node = current_node.parent
        nodes.append(current_node)
    nodes.pop()  # Removing the None
    nodes = list(reversed(nodes))
    vessel_width = np.array(list(reversed(distances)))

    # Ignore nodes near the bifurcations
    # start_width = int(np.floor(vessel_width[0]))
    # end_width = int(np.floor(vessel_width[-1]))
    # if len(nodes) > start_width + end_width:
    #     nodes = nodes[start_width:end_width]
    #     vessel_width = vessel_width[start_width:end_width]

    sigma = int(round(6 * skeleton_distances.shape[0] / 1024))
    vessel_width = gaussian_filter1d(vessel_width, sigma)

    return nodes, vessel_width


def get_location_of_widening(nodes, vessel_width, image_size=1024, show_graph=False, show_graph_only_on_detection=False):
    """
    This method detects a change in the width of the vessel from a list of nodes and a list of vessel widths.
    :param nodes: List of PathNodes.
    :param vessel_width: A list of vessel widths as integers.
    :param image_size: The height of the image.
    :param show_graph: True to show a plot of the vessel widths.
    :param show_graph_only_on_detection: True to show a plot of the angle progression when a sharp angle is found.
    :return: The location (x, y) of the widening alongside of its parent if found, else None twice.
    """
    INTERVAL_SIZE = int(round(30 * image_size / 1024))
    RATIO_THRESHOLD = 1.6
    widening_location = None
    widening_parent_location = None

    for i in range(INTERVAL_SIZE, len(vessel_width)):
        if vessel_width[i-INTERVAL_SIZE] > 0:
            ratio = vessel_width[i] / vessel_width[i-INTERVAL_SIZE]
            if ratio >= RATIO_THRESHOLD:
                print(f"Widening detected with ratio of {ratio}")
                widening_location = (nodes[i].x, nodes[i].y)
                widening_parent_location = (nodes[i-1].x, nodes[i-1].y)
                break

    if show_graph or (show_graph_only_on_detection and widening_location is not None):
        plt.title(f"Vessel width for pixels of skeleton segment from ({nodes[0].x}, {nodes[0].y}) to ({nodes[-1].x}, {nodes[-1].y})")
        plt.plot(vessel_width)
        for i in range(INTERVAL_SIZE, len(vessel_width)):
            if vessel_width[i-INTERVAL_SIZE] > 0:
                ratio = vessel_width[i] / vessel_width[i-INTERVAL_SIZE]
                if ratio >= 1.5:
                    plt.axvline(i)
                    plt.axvspan(i-INTERVAL_SIZE, i, color='y')
                    break
        plt.xlabel("Segment pixel #")
        plt.ylabel("Vessel width in pixels")
        plt.show()

    return widening_location, widening_parent_location


def get_location_of_sharp_angle(nodes, image_size=1024, show_graph=False, show_graph_only_on_detection=False):
    """
    This method detects a sharp angle from a list of nodes by analyzing the gradient of the angle progression between the nodes.
    :param nodes: The nodes (list of PathNode class) of the vessel.
    :param image_size: The height of the image.
    :param show_graph: True to show a plot of the angle progression.
    :param show_graph_only_on_detection: True to show a plot of the angle progression when a sharp angle is found.
    :return: The location (x, y) of the sharp angle alongside of its parent if found, else None twice.
    """
    VECTOR_PIXEL_COUNT = int(round(5 * image_size / 1024))
    MINIMUM_ANGLE_GRADIENT_INTENSITY = 2.75
    MAXIMUM_CURVE_LENGTH = int(round(30 * image_size / 1024))
    angles = []
    for i in range(VECTOR_PIXEL_COUNT-1, len(nodes)):
        points = []
        diff_vectors = []
        for j in reversed(range(VECTOR_PIXEL_COUNT-1)):
            node = nodes[i-j]
            points.append([node.x, -node.y])
            if j < VECTOR_PIXEL_COUNT-1:
                previous_node = nodes[i-j-1]
                diff_vectors.append([node.x - previous_node.x, -node.y + previous_node.y])
        diff_vectors = np.array(diff_vectors)
        sumed_vector = np.sum(diff_vectors, axis=0)
        previous_angle = float('inf') if len(angles) == 0 else angles[-1]
        angle = get_angle_from_vector(sumed_vector, previous_angle)
        angles.append(angle)

    if len(angles) < VECTOR_PIXEL_COUNT:
        return None, None

    angles = np.array(angles)
    angles = gaussian_filter1d(angles, VECTOR_PIXEL_COUNT)
    degrees_gradient = abs(np.gradient(angles))

    if show_graph or (show_graph_only_on_detection and degrees_gradient.max() >= MINIMUM_ANGLE_GRADIENT_INTENSITY):
        plt.title("Angle gradient of direction vector of the skeleton segment")
        plt.axhline(degrees_gradient.mean(), color='g')
        if degrees_gradient.max() >= MINIMUM_ANGLE_GRADIENT_INTENSITY:
            # node = nodes[degrees_gradient.argmax() - int(VECTOR_PIXEL_COUNT/2)]
            # print(node.x, node.y)
            plt.axvline(degrees_gradient.argmax(), color='r')
        plt.plot(degrees_gradient)
        plt.xlabel("Segment pixel #")
        plt.ylabel("Angle gradient of direction vector")
        plt.show()

    if degrees_gradient.max() >= MINIMUM_ANGLE_GRADIENT_INTENSITY:
        start = 0
        end = len(degrees_gradient)
        above_average = degrees_gradient > degrees_gradient.mean()
        for i in range(degrees_gradient.argmax(), 0, -1):
            if not above_average[i]:
                start = i
                break
        for i in range(degrees_gradient.argmax(), len(degrees_gradient)):
            if not above_average[i]:
                end = i
                break
        if end - start <= MAXIMUM_CURVE_LENGTH:  # We don't want the curve to be long, otherwise it's probably not the tip of the catheter
            middle_node_index = degrees_gradient.argmax() + int(VECTOR_PIXEL_COUNT/2)
            node = nodes[middle_node_index]
            parent_node = nodes[middle_node_index - 1]
            if node.y < image_size / 2:
                return (node.x, node.y), (parent_node.x, parent_node.y)
            else:
                print(f"It is very unlikely that the sharp angle we detected at ({node.x}, {node.y}) is in the catheter because it is in the bottom half of the image.")
        elif show_graph or (show_graph_only_on_detection and degrees_gradient.max() >= MINIMUM_ANGLE_GRADIENT_INTENSITY):
            print(f"Sharp angle was discarded because the curve is {end - start} pixels long")
    return None, None


def get_angle_from_vector(vector, previous_angle=float('inf'), debug=False):
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    angle_diff = angle if previous_angle == float('inf') else angle - previous_angle
    if angle_diff > 180:
        if debug:
            print("Subtracting 360 to keep close")
        angle -= 360
    elif angle_diff < -180:
        if debug:
            print("Adding 360 to keep close")
        angle += 360
    return angle


def vesselanalysis(image, out_name, use_scikit=True):
    print(out_name)

    # 1) get skeleton
    start = time.time()
    if use_scikit:
        skeleton, distance = medial_axis(image, return_distance=True)
        strskel = skeleton.tostring()
    else:
        # 1 get skeleton
        distance = np.ones_like(image)
        start = time.time()
        skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        strskel = skeleton.tostring()
        strseg = image.tostring()
        toolsdll.vesselanalysis_getskeleton(c_char_p(strseg), image.shape[0], image.shape[1], c_char_p(strskel))
    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(image.shape[0] ,image.shape[1]))
    # cv2.imwrite(f'{out_path}\\test.png', cvskel*255)
    print(f"1. skeleton took {time.time() - start}s")

    # 2) get bifurcations
    start = time.time()
    bifurcation_pixels_x = []
    bifurcation_pixels_y = []
    skel_y, skel_x = np.where(cvskel > 0)
    for pixel in range(len(skel_x)):
        x = skel_x[pixel]
        y = skel_y[pixel]
        bifurcation = detect_bifurcation((y, x), cvskel)
        if bifurcation:
            bifurcation_pixels_x.append(x)
            bifurcation_pixels_y.append(y)
    print(f"{len(bifurcation_pixels_x)} bifurcations identified")
    bifurcations_layer = np.zeros(cvskel.shape, dtype=np.uint8)
    bifurcations_layer[np.array(bifurcation_pixels_y, dtype=np.int), np.array(bifurcation_pixels_x, dtype=np.int)] = 1
    bifurcations = np.repeat(cvskel[:, :, np.newaxis], 3, axis=2)
    bifurcations[:, :, 1] -= bifurcations_layer
    print(f"2. bifurcations took {time.time() - start}s")
    cv2.imwrite(out_name + "_bifurcations.png", bifurcations*255)

    skeleton_distance = cvskel * distance

    # 3) remove spurs
    start = time.time()
    bifurcation_list = list(zip(bifurcation_pixels_y, bifurcation_pixels_x))
    iteration_count = remove_spurs(bifurcation_list, bifurcations_layer, bifurcations, skeleton_distance, cvskel, distance)
    print(f"3. spur pruning took {time.time() - start}s for {iteration_count} iterations")
    cv2.imwrite(out_name + "_bifurcations_pruned.png", bifurcations*255)

    return skeleton_distance, bifurcations_layer


def remove_spurs(bifurcation_list, bifurcations_layer, bifurcations, skeleton_distance, cvskel, distance):
    iterate = True
    iteration = 0
    while iterate:
        iteration += 1
        # print(f"Iteration {iteration}")
        iterate = False
        pruned_branches = 0
        branch_points_to_remove = []

        # 3.1) Identifying the spurs
        for bifurcation in bifurcation_list:
            # print(f"bifurcation: {bifurcation}")
            branches = get_branches(bifurcation, skeleton_distance)
            for branch in branches:
                start_point = (bifurcation[1] + branch[1], bifurcation[0] + branch[0])
                result_dict = follow_path_bfs(skeleton_distance, bifurcations_layer, starting_point=start_point, parent=(bifurcation[1], bifurcation[0]), get_branch_points=True, debug=False)
                branch_points = result_dict["branch_points"]
                ends_on_dead_end = result_dict["dead_end"]
                if ends_on_dead_end:
                    branch_length = len(branch_points)
                    branch_width = get_average_vessel_width(branch_points, skeleton_distance, real_average=True)
                    width_to_length_ratio = branch_width / branch_length
                    # print(f"branch {branch} has a length of {branch_length} and a width/len ratio of {width_to_length_ratio}")
                    if width_to_length_ratio > 0.1 + 0.01 * branch_length:  # The longer the branch is, the higher the width to length ratio threshold is
                        iterate = True
                        pruned_branches += 1
                        branch_points_to_remove += branch_points
                        # print("Pruning branch")

        # 3.2) Pruning the spurs
        for branch_point in branch_points_to_remove:
            point = branch_point[1], branch_point[0]  # change from (x, y) to (y, x)
            if bifurcations_layer[point] > 0:
                bifurcations_layer[point] = 0
            else:
                cvskel[point] = 0

        # 3.3) Recreating the bifurcations and skeleton distances matrices
        bifurcations[:, :, :] = np.repeat(cvskel[:, :, np.newaxis], 3, axis=2)
        bifurcations[:, :, 1] -= bifurcations_layer
        skeleton_distance[:, :] = cvskel * distance

    return iteration


def get_branches(point, skeleton, debug=False):
    """
    Returns the branches around a point on a skeleton.
    :param point: Tuple (y, x)
    :param skeleton: 2D ndarray representing the skeleton.
    :param debug: True to print debug logs.
    :return: A list of points around the starting point.
    """
    branches = []
    y = point[0]
    x = point[1]
    patch = skeleton[y-1:y+2, x-1:x+2]
    if debug:
        print(patch)
    # If the patch is incomplete, we won't be able to find branches
    if patch.shape[0] * patch.shape[1] < 9:
        return branches
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
    return adjusted_branches


def detect_bifurcation(point, skeleton):
    """
    Check the branches around a point on a skeleton to determine if it is a bifurcation.
    :param point: Tuple (y, x)
    :param skeleton: 2D ndarray representing the skeleton.
    :return: A boolean revealing if the point is a bifurcation and a list containing the starting point of the branches.
    """
    return len(get_branches(point, skeleton)) >= 3


def overlap(image, segmentation_image, skeleton, bifurcations, out_name):
    segmentation_alpha = 0.3
    # Overlapping skeleton and original image
    rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    segmentation_indices = segmentation_image.nonzero()
    skeleton_indices = skeleton.nonzero()
    bifurcations_indices = bifurcations.nonzero()
    rgb_image[segmentation_indices[0], segmentation_indices[1], 1] = (1 - segmentation_alpha) * rgb_image[segmentation_indices[0], segmentation_indices[1], 0] + segmentation_alpha * segmentation_image[segmentation_indices]
    rgb_image[skeleton_indices[0], skeleton_indices[1], :] = 255
    rgb_image[bifurcations_indices[0], bifurcations_indices[1], 1] = 0
    cv2.imwrite(out_name, rgb_image)


def get_graph_data_from_raw_image_and_segmentation(raw_image_path, segmented_image_path, oriented):
    """
    This method loads the raw and segmented images and extract graph data from the skeleton after identifying the ostium.
    :param raw_image_path: Path of the raw image.
    :param segmented_image_path: Path of the segmented image.
    :param oriented: True for getting data of an oriented graph, False for a non oriented graph (where crossings are not identified, thus having many more nodes and shorter edges).
    :return: The list of adjacent nodes, node features, edge features and edge ids.
    """
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    segmented_image = fill_missing_pixels(segmented_image)
    last_point_index = segmented_image_path.rfind(".")
    out_name = segmented_image_path[:last_point_index] + "_skeleton"
    skeleton, bifurcations = vesselanalysis(segmented_image, out_name)
    ostium, ostium_parent = find_ostium(raw_image, skeleton, bifurcations, debug=False)
    if ostium is None:
        raise Exception(f"Failed to detect the ostium from raw image {raw_image_path} and segmented image {segmented_image_path}.")
    adjacency_matrix, node_features, edge_features = extract_graph_from_skeleton(raw_image, segmented_image, skeleton, bifurcations, ostium, ostium_parent, oriented, debug=False)
    edge_ids = {}  # (node_a, node_b): edge_id
    edge_counter = 0
    edges = []
    edge_attr = []
    for node_a, nodes_b in adjacency_matrix.items():
        for node_b in nodes_b:
            edges.append([node_a, node_b])
            edge_attr.append(edge_features[(node_a, node_b)])
            edge_ids[(node_a, node_b)] = edge_counter
            edge_counter += 1
    x = [features for features in node_features.values()]
    return edges, x, edge_attr, edge_ids


if __name__ == '__main__':
    """
    To create the overlap image in a folder that has an angiography image and a segmentation image with the same name + "_seg", use these command line parameters:
    python src/VESSELANALYSIS/vesselanalysis.py --input [folder] --file_type tif --visualize True
    
    To find the ostium based on a segmentation file, use these command line parameters:
    python src/VESSELANALYSIS/vesselanalysis.py --input [file]
    
    To extract the graph of a skeleton made from a segmentation file, use these command line parameters (add --oriented for the oriented graph):
    python src/VESSELANALYSIS/vesselanalysis.py --input [file] --extract_graph True
    
    To evaluate the ostium detection on all segmented files, use these command line parameters:
    python src/VESSELANALYSIS/vesselanalysis.py --input [folder] --file_type tif --evaluate True
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='input file or folder in which to search recursively for images of segmented frames to skeletonize')
    parser.add_argument('--file_type', default='png', help='type of images to skeletonize (png, jpg, tif, etc)')
    parser.add_argument('--contains', default='', help='filters out files that do not contain that string (i.e. "_seg")')
    parser.add_argument('--visualize', default=False, help='mode where it saves overlapped images of the skeleton over the original image')
    parser.add_argument('--evaluate', default=False, help='mode where the ostium detection is compared with the ground truth for all images')
    parser.add_argument('--extract_graph', default=False, help='extracts the oriented graph after finding the ostium')
    parser.add_argument('--oriented', default=False, help='used only with `--extract_graph True` and defines if the graph should be oriented or not')
    parser.add_argument('--greedy_crossing', default=False, help='if True, the first valid crossing will be used instead of searching for every valid crossing and keeping the best one')
    parser.add_argument('--debug', default=False, help='will output debug logs if True (only used for parsing single files, not folders)')
    args = parser.parse_args()

    input_path = args.input
    file_type = args.file_type
    contains = args.contains
    visualize = args.visualize == "True"
    evaluate = args.evaluate == "True"
    extract_graph = args.extract_graph == "True"
    oriented = args.oriented == "True"
    greedy_crossing = args.greedy_crossing == "True"
    debug = args.debug == "True"

    search_best_crossing = not greedy_crossing

    if os.path.isfile(input_path):
        print("Input is a file")
        file_type = input_path.split('.')[-1]
        segmented_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        file_name = input_path.split(f".{file_type}")[0]
        file_name = file_name.split("_seg")[0]
        raw_image_file_name = file_name + f".{file_type}"
        raw_image = cv2.imread(raw_image_file_name, cv2.IMREAD_GRAYSCALE)
        segmented_image = fill_missing_pixels(segmented_image)
        skeleton, bifurcations = vesselanalysis(segmented_image, f"{file_name}_skeleton")
        ostium, ostium_parent = find_ostium(raw_image, skeleton, bifurcations, search_best_crossing=search_best_crossing, debug=debug)
        if ostium is not None:
            print(f"Ostium found at location {ostium}, parent is {ostium_parent}")
            if extract_graph:
                adjacency_matrix, node_features, edge_features = extract_graph_from_skeleton(raw_image, segmented_image, skeleton, bifurcations, ostium, ostium_parent, oriented, search_best_crossing=search_best_crossing, show_final_graph=True, debug=debug)
        else:
            print("Ostium not found")
        skeleton[skeleton > 0] = 1
        visualization = np.repeat(skeleton[:, :, np.newaxis], 3, 2)
        visualization[:, :, 1] -= bifurcations
        if ostium is not None:
            visualization[ostium[1], ostium[0], 0] -= 1
        plt.imshow(visualization)
        plt.show()
    else:
        print("Input is a folder")
        evaluation_results = {}  # folder: distance
        for path, subfolders, files in os.walk(input_path):
            print(path)

            output_folder = f'{path}'

            for file in files:
                image_type = f".{file_type}"
                if not file.endswith(image_type):
                    continue

                if contains != "" and contains not in file:
                    continue

                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                if (visualize or evaluate) and "_seg" in file:
                    continue

                image_path = f'{path}\\{file}'
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if visualize or evaluate:
                    file_name = file.split(f".{file_type}")[0]
                    segmentation_file_name = f"{file_name}_seg"
                    segmentation_file = f"{path}\\{segmentation_file_name}.{file_type}"
                    seg_image = cv2.imread(segmentation_file, cv2.IMREAD_GRAYSCALE)
                    seg_image = fill_missing_pixels(seg_image)

                    if visualize:
                        cv2.imwrite(f"{path}\\{segmentation_file_name}_filled.png", seg_image)
                        output = f'{output_folder}\\{file_name}_skeleton'
                        skeleton, components = vesselanalysis(seg_image, output)
                        output = f'{output_folder}\\{file_name}_overlap.png'
                        overlap(image, seg_image, skeleton, components, output)

                    else:
                        skeleton, bifurcations = vesselanalysis(seg_image, f"{output_folder}\\{file_name}_skeleton")
                        if skeleton.shape[0] != 1024:
                            print("Image has shape of", skeleton.shape)
                            exit(0)
                        ostium, ostium_parent = find_ostium(image, skeleton, bifurcations)
                        if ostium is not None:
                            ostium_gt_file_name = f"{file_name}_ostium.txt"
                            f = open(f"{path}\\{ostium_gt_file_name}", 'r')
                            ostium_gt = f.readline().split(" ")
                            ostium_gt = (int(ostium_gt[0]), int(ostium_gt[1]))
                            distance = np.sqrt((ostium[0] - ostium_gt[0]) ** 2 + (ostium[1] - ostium_gt[1]) ** 2)
                            max_distance = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
                            relative_distance = distance / max_distance
                            print(f"distance: {int(round(distance))}px or {int(relative_distance * 1000) / 10}%")
                            evaluation_results[file_name] = distance
                        else:
                            evaluation_results[file_name] = float('inf')

                        # TODO do not keep
                        # skeleton[skeleton > 0] = 1
                        # visualization = np.repeat(skeleton[:, :, np.newaxis], 3, 2)
                        # visualization[:, :, 1] -= bifurcations
                        # if ostium is not None:
                        #     visualization[ostium[1], ostium[0], 0] -= 1
                        # else:
                        #     print("Ostium not found")
                        # plt.imshow(visualization)
                        # plt.show()

                else:
                    image = fill_missing_pixels(image)
                    # file_name = file.split("segmented")[0]
                    file_name = file.split("seg")[0]
                    output = f'{output_folder}\\{file_name}skeletonized'
                    cv2.imwrite(f'{output_folder}\\{file_name}seg_filled.png', image)
                    vesselanalysis(image, output)

        if evaluate:
            sorted_evaluation_results = sorted(evaluation_results.items(), key=operator.itemgetter(1))
            print(sorted_evaluation_results)
            print(f"average: {np.mean(np.array(list(evaluation_results.values())))}px")
