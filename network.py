from link import Link
from node import Node
from path import Path
from od import OD

import sys
import traceback
import utils

import pdb
import numpy as np
import time

FRANK_WOLFE_STEPSIZE_PRECISION = 1e-4


class BadNetworkOperationException(Exception):
    """
    You can raise this exception if you try a network action which is invalid
    (e.g., trying to find a topological order on a network with cycles.)
    """
    pass


class Network:
    """
    This is the class used for transportation networks.  It uses the following
    dictionaries to store the network; the keys are IDs for the network elements,
    and the values are objects of the relevant type:
       node -- network nodes; see node.py for description of this class
       link -- network links; see link.py for description of this class
       ODpair -- origin-destination pairs; see od.py
       path -- network paths; see path.py.  Paths are NOT automatically generated
               when the network is initialized (you probably wouldn't want this,
               the number of paths is exponential in network size.)

       The network topology is expressed both in links (through the tail and head
       nodes) and in nodes (forwardStar and reverseStar are Node attributes storing
       the IDs of entering and leaving links in a list).

       numNodes, numLinks, numZones -- self-explanatory
       firstThroughNode -- in the TNTP data format, transiting through nodes with
                           low IDs can be prohibited (typically for centroids; you
                           may not want vehicles to use these as "shortcuts").
                           When implementing shortest path or other routefinding,
                           you should prevent trips from using nodes with lower
                           IDs than firstThroughNode, unless it is the destination.
    """

    def __init__(self, networkFile="", demandFile=""):
        """
        Class initializer; if both a network file and demand file are specified,
        will read these files to fill the network data structure.
        """
        self.numNodes = 0
        self.numLinks = 0
        self.numZones = 0
        self.firstThroughNode = 0

        self.node = dict()
        self.link = dict()
        self.ODpair = dict()
        self.path = dict()

        if len(networkFile) > 0 and len(demandFile) > 0:
            self.readFromFiles(networkFile, demandFile)

    def relativeGap(self):
        """
        This method should calculate the relative gap (as defined in the course text)
        based on the current link flows, and return this value.

        To do this, you will need to calculate both the total system travel time, and
        the shortest path travel time (you will find it useful to call some of the
        methods implemented in earlier assignments).
        """
        # raise utils.NotYetAttemptedException
        nom = 0
        denom = 0

        for ij in self.link:
            nom += self.link[ij].cost * self.link[ij].flow

        for od in self.ODpair.values():
            backlink, cost = self.shortestPath(od.origin)
            denom += cost[od.destination] * od.demand

        gamma = nom / denom - 1
        return gamma

    def averageExcessCost(self):
        """
        This method should calculate the average excess cost
        based on the current link flows, and return this value.

        To do this, you will need to calculate both the total system travel time, and
        the shortest path travel time (you will find it useful to call some of the
        methods implemented in earlier assignments).
        """
        # raise utils.NotYetAttemptedException
        tx = 0
        kd = 0
        td = 0

        for ij in self.link:
            tx += self.link[ij].cost * self.link[ij].flow

        for od in self.ODpair.values():
            backlink, cost = self.shortestPath(od.origin)
            kd += cost[od.destination] * od.demand
            td += od.demand

        aec = (tx - kd) / float(td)
        return aec

    def shiftFlows(self, targetFlows, stepSize):
        """
        This method should update the flow on each link, by taking a weighted
        average of the current link flows (self.link[ij].flow) and the flows
        given in the targetFlows dictionary (targetFlows[ij]).  stepSize indicates
        the weight to place on the target flows (so the weight on the current
        flows is 1 - stepSize).

        *** IMPORTANT: After updating the flow on a link, you should call its
        updateCost method, so that the travel time is updated to reflect
        the new flow value. ***

        This method does not need to return a value.
        """
        for key, link in self.link.items():
            link.flow = stepSize * \
                targetFlows[key] + (1 - stepSize) * link.flow
            link.updateCost()

    def FrankWolfeStepSize(self, targetFlows, precision=FRANK_WOLFE_STEPSIZE_PRECISION, rootfinding='NR'):
        """
        This method returns the step size lambda used by the Frank-Wolfe algorithm.

        The current link flows are given in the self.link[ij].flow attributes, and the
        target flows are given in the targetFlows dictionary.

        The precision argument dictates how close your method needs to come to finding
        the exact Frank-Wolfe step size: you are fine if the absolute difference
        between the true value, and the value returned by your method, is less than
        precision.
        """
        # raise utils.NotYetAttemptedException
        high = 1.0
        low = 0
        calc = 1e6
        intervalEnd = False
        # pdb.set_trace()
        if rootfinding == 'bisection':

            while abs(calc) >= precision and not intervalEnd:
                lam = (high + low) / 2.0
                calc = 0
                for key, link in self.link.items():
                    rflow = link.flow
                    link.flow = lam * targetFlows[key] + (1 - lam) * link.flow
                    cost = link.calculateCost()
                    link.flow = rflow
                    calc += cost * (targetFlows[key] - link.flow)

                if calc > 0:
                    high = lam
                else:
                    low = lam

                if abs(lam) < 1e-6:
                  lam = 0
                  intervalEnd = True
                elif lam > 1 - 1e-6:
                  lam = 1
                  intervalEnd = True

        if rootfinding == 'NR':

            lam = high

            while abs(calc) > precision:

                try:
                    lam = lam - calc / calc_df
                except:
                    pass

                calc = 0
                calc_df = 0

                for key, link in self.link.items():
                    rflow = link.flow
                    link.flow = lam * targetFlows[key] + (1 - lam) * link.flow
                    cost = link.calculateCost()
                    cost_df = link.calculateCost_df()
                    link.flow = rflow
                    calc += (cost * (targetFlows[key] - link.flow))
                    calc_df += (cost_df * (targetFlows[key] - link.flow)**2)

            # print(calc)
            if abs(lam) < 1e-6:
                lam = 0
                # intervalEnd = True
            elif lam > 1 - 1e-6:
                lam = 1

        return lam

    def userEquilibrium_old(self, stepSizeRule='MSA',
                        maxIterations=10,
                        targetGap=1e-6,
                        gapFunction=relativeGap):
        """
        This method uses the (link-based) convex combinations algorithm to solve
        for user equilibrium.  Arguments are the following:
           stepSizeRule -- a string specifying how the step size lambda is
                           to be chosen.  Currently 'FW' and 'MSA' are the
                           available choices, but you can implement more if you
                           want.
           maxIterations -- stop after this many iterations have been performed
           targetGap     -- stop once the gap is below this level
           gapFunction   -- pointer to the function used to calculate gap.  After
                            finishing this assignment, you should be able to
                            choose either relativeGap or averageExcessCost.
        """
        initialFlows = self.allOrNothing()
        for ij in self.link:
            self.link[ij].flow = initialFlows[ij]
            self.link[ij].updateCost()

        iteration = 0
        while iteration < maxIterations:
            iteration += 1
            gap = gapFunction()
            print("Iteration %d: gap %f" % (iteration, gap))
            if gap < targetGap:
                break
            targetFlows = self.allOrNothing()
            if stepSizeRule == 'FW':
                stepSize = self.FrankWolfeStepSize(targetFlows)
            elif stepSizeRule == 'MSA':
                stepSize = 1 / (iteration + 1)
            else:
                raise BadNetworkOperationException(
                    "Unknown step size rule " + str(stepSizeRule))
            self.shiftFlows(targetFlows, stepSize)

    def userEquilibriumTest(self, stepSizeRule='MSA',
                            maxIterations=10,
                            targetGap=1e-6,
                            gapFunction=relativeGap,
                            rootfinding='bisection',
                            MSAstepSizeChoice='vanilla'):
        """
        This method uses the (link-based) convex combinations algorithm to solve
        for user equilibrium.  Arguments are the following:
           stepSizeRule -- a string specifying how the step size lambda is
                           to be chosen.  Currently 'FW' and 'MSA' are the
                           available choices, but you can implement more if you
                           want.
           maxIterations -- stop after this many iterations have been performed
           targetGap     -- stop once the gap is below this level
           gapFunction   -- pointer to the function used to calculate gap.  After
                            finishing this assignment, you should be able to
                            choose either relativeGap or averageExcessCost.
        """
        gapList = []
        iterationList = []
        timeList = []
        
        st = time.time()
        initialFlows = self.allOrNothing()
        for ij in self.link:
            self.link[ij].flow = initialFlows[ij]
            self.link[ij].updateCost()
        print('stepSizeRule', stepSizeRule)
        if stepSizeRule == 'FW':
            print('rootfinding', rootfinding)
        else: 
            print('MSAstepSizeChoice', MSAstepSizeChoice)
        iteration = 0
        while iteration < maxIterations:
            iteration += 1
            gap = gapFunction()
            iterationList.append(iteration)
            gapList.append(gap)
            timeList.append(time.time() - st)
            print("Iteration %d: gap %f: time %f" %
                  (iteration, gap, time.time() - st))
            if gap < targetGap:
                break
            targetFlows = self.allOrNothing()
            if stepSizeRule == 'FW':
                stepSize = self.FrankWolfeStepSize(
                    targetFlows, rootfinding=rootfinding)
            elif stepSizeRule == 'MSA':
                if MSAstepSizeChoice == 'vanilla':
                    stepSize = 1 / (iteration + 1)
                if MSAstepSizeChoice == 'diff':
                    # stepSize = 1 / (iteration**(2.0/3.0))
                    stepSize = 1 / (iteration + 2)

            else:
                raise BadNetworkOperationException(
                    "Unknown step size rule " + str(stepSizeRule))

            self.shiftFlows(targetFlows, stepSize)

        return iterationList, gapList, timeList

    def userEquilibrium(self, stepSizeRule='MSA',
                        maxIterations=10,
                        targetGap=1e-6,
                        gapFunction=relativeGap):
        """
        This method uses the (link-based) convex combinations algorithm to solve
        for user equilibrium.  Arguments are the following:
           stepSizeRule -- a string specifying how the step size lambda is
                           to be chosen.  Currently 'FW' and 'MSA' are the
                           available choices, but you can implement more if you
                           want.
           maxIterations -- stop after this many iterations have been performed
           targetGap     -- stop once the gap is below this level
           gapFunction   -- pointer to the function used to calculate gap.  After
                            finishing this assignment, you should be able to
                            choose either relativeGap or averageExcessCost.
        """

        # gapList = []
        gap = 1e6
        st = time.time()

        # g = {}
        # destlist = []
        # for od in self.ODpair.values():
        #     destlist.append(od.destination)
        # destlist = list(set(destlist))
        # for dest in destlist:
        #     g[dest] = self.find_g(dest)
        # timeList = []
        # iters = 0
        while gap > targetGap:
            # iters += 1 
            for od in self.ODpair.values():
                origin = od.origin
                curnode = od.destination

                if od.demand == 0:
                    continue

                (backlink, cost) = self.shortestPath(origin)
                # (backlink, cost) = self.shortestPath(origin, od.destination, destonly=True)
                # backnode, cost = self.a_star(origin, od.destination, destonly=True)
                # backnode, cost = self.a_star(origin, od.destination, g[curnode])

                curpath = []
                curpath.append(curnode)

                while curnode != origin:
                    curnode = self.link[backlink[curnode]].tail
                    # curnode = backnode[curnode]
                    curpath.append(curnode)

                curpath = curpath[::-1]
                curpathStr = ','
                for n in curpath:
                    curpathStr += str(n) + ','

                if str(curpath) not in od.pi_rs:
                    pathTuple = utils.path2linkTuple(curpathStr)

                    if len(od.pi_rs) == 0:
                        self.path[str(curpath)] = Path(pathTuple, self, od.demand)
                    else:
                        self.path[str(curpath)] = Path(pathTuple, self, 0)

                    od.pi_rs.append(str(curpath))

                if len(od.pi_rs) > 1:
                    minv = np.inf
                    basic_p = None
                    basic_v = None
                    for k in od.pi_rs:
                        v = self.path[k].cost
                        if v <= minv:
                            basic_p = k
                            basic_v = v
                            minv = v
                    b_links = self.path[basic_p].links
                    for k in od.pi_rs:
                        if k != basic_p:
                            v = self.path[k].cost
                            diff = v-basic_v
                            nb_links = self.path[k].links
                            u = set(nb_links).union(set(b_links))
                            i = set(nb_links).intersection(set(b_links))
                            diff_links = tuple(u.difference(i))
                            denom = 0
                            for lnk in diff_links:
                                denom += self.link[lnk].calculateCost_df() 

                            shift = min(diff/denom, self.path[k].flow)
                            self.path[k].flow -= shift
                            self.path[basic_p].flow += shift
                            
                            ## delete unused paths
                            if self.path[k].flow == 0:
                                del self.path[k]
                                od.pi_rs.remove(k)
                self.loadPaths()

            gap = gapFunction()
            # gapList.append(gap)
            # timeList.append(time.time()-st)
            print("gap %f: time %f" % (gap, time.time()-st))

        # return gapList, timeList

    def userEquilibriumImprovement2(self, stepSizeRule='MSA',
                            maxIterations=10,
                            targetGap=1e-6,
                            gapFunction=relativeGap,
                            rootfinding='bisection',
                            MSAstepSizeChoice='vanilla'):
        """
        This method uses the (link-based) convex combinations algorithm to solve
        for user equilibrium.  Arguments are the following:
           stepSizeRule -- a string specifying how the step size lambda is
                           to be chosen.  Currently 'FW' and 'MSA' are the
                           available choices, but you can implement more if you
                           want.
           maxIterations -- stop after this many iterations have been performed
           targetGap     -- stop once the gap is below this level
           gapFunction   -- pointer to the function used to calculate gap.  After
                            finishing this assignment, you should be able to
                            choose either relativeGap or averageExcessCost.
        """


            # def find_g(self, destination):
        #for everynode find sp to s
        g = {}
        for i in self.node:
            if i != destination:
                bnode, cost = self.shortestPath(ori, freeflow=True)
                ### test a_star code with older hw assignment
                g[i] = cost[destination]
        g[destination] = 0
        return g

        gapList = []
        gap = 1e6
        st = time.time()

        backlink_o = {} 
        cost_o = {}

        destlist = []
        for od in self.ODpair.values():
            orlist.append(od.origin)
        orlist = list(set(orlist))
        for ori in orlist:
            backlink, cost = self.shortestPath(ori, freeflow=True)
            backlink_o[ori] = backlink
            cost_o[ori] = cost

            curnode = od.destination
            while curnode != origin:
                curnode = self.link[backlink[curnode]].tail
                curpath.append(curnode)

            curpath = curpath[::-1]
            curpathStr = ','
            for n in curpath:
                curpathStr += str(n) + ','

            pathTuple = utils.path2linkTuple(curpathStr)
            self.bush[ori] = Bush(pathTuple, ori, self, curpath)

            self.bush[ori].findTopologicalOrder()

            for topoNode in range(self.numNodes + 1, self.node[origin].order + 1):
                if i == ori:
                    break
                i = self.topologicalList[topoNode]

                maxorder = -np.inf
                maxorderNode = None
                for n in self.bush[ori].node:
                    curorder = self.bush[ori].node[n].order
                    if curorder > maxorder:
                        maxorder = curorder
                        maxorderNode = n

                self.bush[ori].calculateBushLabels()
                backlinkL = self.bush[ori].backlinkL[maxorderNode]
                piL = []
                curnode = maxorderNode
                piL.append(curnode)

                while curnode != ori:
                    curnode = self.bush[ori].link[backlinkL[curnode]].tail
                    piL.append(curnode)

                backlinkU = self.bush[ori].backlinkU[maxorderNode]
                piU = []
                curnode = maxorderNode
                piU.append(curnode)

                while curnode != ori:
                    curnode = self.bush[ori].link[backlinkU[curnode]].tail
                    piU.append(curnode)

                # lastcommonNode = None
                # decoypiU = piU[::-1]
                # decoypiL = piL[::-1]
                # for j in range(1, len(decoypiL)):
                #     if decoypiL[j] == decoypiU[j]:
                #         lastcommonNode = decoypiL[j]
                #         break

                lastcommonNode = None
                decoypiU = piU
                decoypiL = piL
                for j in range(1, len(decoypiL)):
                    if decoypiL[j] == decoypiU[j]:
                        lastcommonNode = decoypiL[j]
                        break

                piL = piL[::-1]
                curpathStr = ','
                for n in piL:
                    curpathStr += str(n) + ','
                piL = utils.path2linkTuple(curpathStr)

                piU = piU[::-1]
                curpathStr = ','
                for n in piU:
                    curpathStr += str(n) + ','
                piU = utils.path2linkTuple(curpathStr)

                nom = (self.bush[ori].U[i] - self.bush[ori].U[a]) - (self.bush[ori].L[i] - self.bush[ori].L[a])
                denom = 0
                cmn_links = list(set(piL).union(set(piU)))
                for lnk in cmn_links:
                    denom += self.link[lnk].calculateCost_df()

                minUij = np.inf
                for ij in piU:
                    curflow = self.bush[ori].link[ij].flow
                    if curflow < minUij:
                        minUij = curflow

                deltah = min(nom/denom, minUij)

            updatealltijs


        return None




    #     # def loadBush:


    #     iters = 0
    #     while gap > targetGap:
    #         iters += 1 
    #         for od in self.ODpair.values():
    #             origin = od.origin
    #             curnode = od.destination

    #             if od.demand == 0:
    #                 continue

    #             (backlink, cost) = self.shortestPath(origin)
    #             # (backlink, cost) = self.shortestPath(origin, od.destination, destonly=True)
    #             # backnode, cost = self.a_star(origin, od.destination, destonly=True)
    #             # backnode, cost = self.a_star(origin, od.destination, g[curnode])

    #             curpath = []
    #             curpath.append(curnode)

    #             while curnode != origin:
    #                 curnode = self.link[backlink[curnode]].tail
    #                 # curnode = backnode[curnode]
    #                 curpath.append(curnode)

    #             curpath = curpath[::-1]
    #             curpathStr = ','
    #             for n in curpath:
    #                 curpathStr += str(n) + ','

    #             if str(curpath) not in od.pi_rs:
    #                 pathTuple = utils.path2linkTuple(curpathStr)

    #                 if len(od.pi_rs) == 0:
    #                     self.path[str(curpath)] = Path(pathTuple, self, od.demand)
    #                 else:
    #                     self.path[str(curpath)] = Path(pathTuple, self, 0)

    #                 od.pi_rs.append(str(curpath))

    #             if len(od.pi_rs) > 1:
    #                 minv = np.inf
    #                 basic_p = None
    #                 basic_v = None
    #                 for k in od.pi_rs:
    #                     v = self.path[k].cost
    #                     if v <= minv:
    #                         basic_p = k
    #                         basic_v = v
    #                         minv = v
    #                 b_links = self.path[basic_p].links
    #                 for k in od.pi_rs:
    #                     if k != basic_p:
    #                         v = self.path[k].cost
    #                         diff = v-basic_v
    #                         nb_links = self.path[k].links
    #                         u = set(nb_links).union(set(b_links))
    #                         i = set(nb_links).intersection(set(b_links))
    #                         diff_links = tuple(u.difference(i))
    #                         denom = 0
    #                         for lnk in diff_links:
    #                             denom += self.link[lnk].calculateCost_df() 

    #                         shift = min(diff/denom, self.path[k].flow)
    #                         self.path[k].flow -= shift
    #                         self.path[basic_p].flow += shift
                            
    #                         ## delete unused paths
    #                         if self.path[k].flow == 0:
    #                             del self.path[k]
    #                             od.pi_rs.remove(k)
    #             self.loadPaths()

    #         gap = gapFunction()
    #         gapList.append(gap)
    #         print("gap %f: time %f" % (gap, time.time()-st))



    #     return gapList, time.time()-st

    def beckmannFunction(self):
        """
        This method evaluates the Beckmann function at the current link
        flows.
        """
        beckmann = 0
        for ij in self.link:
            beckmann += self.link[ij].calculateBeckmannComponent()
        return beckmann

    def acyclicShortestPath(self, origin):
        """
        This method finds the shortest path in an acyclic network, from the stated
        origin.  You can assume that a topological order has already been found,
        and referred to in the 'order' attributes of network Nodes.  You can also
        find a list of nodes in topological order in self.topologicalList.  (See the
        method createTopologicalList below.)

        Use the 'cost' attribute of the Links to calculate travel times.  These values
        are given -- do not try to recalculate them based on flows, BPR functions, etc.

        Be aware that both the order Node attribute and topologicalList respect the usual
        convention in network modeling that the topological order starts at 1, whereas
        Python starts numbering at 0.  

        The implementation in the text uses a vector of backnode labels.  In this
        assignment, you should use back-LINK labels instead.  The idea is exactly
        the same, except you are storing the ID of the last *link* in a shortest
        path to each node.

        The backlink and cost labels are both stored in dict's, whose keys are
        node IDs.

        *** BE SURE YOUR IMPLEMENTATION RESPECTS THE FIRST THROUGH NODE!
        *** Travelers should not be able to use "centroid connectors" as shortcuts,
        *** and the shortest path tree should reflect this.

        You should use the macro utils.NO_PATH_EXISTS to initialize backlink labels,
        and utils.INFINITY to initialize cost labels.
        """
        backlink = dict()
        cost = dict()

        for i in self.node:
            backlink[i] = utils.NO_PATH_EXISTS
            cost[i] = utils.INFINITY
        cost[origin] = 0

        for topoNode in range(self.node[origin].order + 1, self.numNodes + 1):
            i = self.topologicalList[topoNode]
            for hi in self.node[i].reverseStar:
                h = self.link[hi].tail
                if h < self.firstThroughNode and h != origin:
                    continue

                tempCost = cost[h] + self.link[hi].cost
                if tempCost < cost[i]:
                    cost[i] = tempCost
                    backlink[i] = hi

        return (backlink, cost)

    def shortestPath(self, origin, destination=None, destonly=False, freeflow=False):

        """
        This method finds the shortest path in a network which may or may not have
        cycles; thus you cannot assume that a topological order exists.

        The implementation in the text uses a vector of backnode labels.  In this
        assignment, you should use back-LINK labels instead.  The idea is exactly
        the same, except you are storing the ID of the last *link* in a shortest
        path to each node.

        Use the 'cost' attribute of the Links to calculate travel times.  These values
        are given -- do not try to recalculate them based on flows, BPR functions, etc.

        The backlink and cost labels are both stored in dict's, whose keys are
        node IDs.

        *** BE SURE YOUR IMPLEMENTATION RESPECTS THE FIRST THROUGH NODE!
        *** Travelers should not be able to use "centroid connectors" as shortcuts,
        *** and the shortest path tree should reflect this.

        You should use the macro utils.NO_PATH_EXISTS to initialize backlink labels,
        and utils.INFINITY to initialize cost labels.
        """
        backlink = dict()
        cost = dict()

        for i in self.node:
            backlink[i] = utils.NO_PATH_EXISTS
            cost[i] = utils.INFINITY

        cost[origin] = 0

        scanList = [self.link[ij].head for ij in self.node[origin].forwardStar]

        while len(scanList) > 0:
            i = scanList[0]
            scanList.remove(i)
            labelChanged = False
            for hi in self.node[i].reverseStar:
                h = self.link[hi].tail
                if h < self.firstThroughNode and h != origin:
                    continue
                if freeflow:
                    tempCost = cost[h] + self.link[hi].freeFlowTime
                
                else:
                    tempCost = cost[h] + self.link[hi].cost
                if tempCost < cost[i]:
                    cost[i] = tempCost
                    backlink[i] = hi
                    labelChanged = True
            if labelChanged == True:
                scanList.extend([self.link[ij].head for ij in self.node[i].forwardStar
                                 if self.link[ij].head not in scanList])
            if destonly:
                if not labelChanged and i==destination:
                    break

        return (backlink, cost)

    def a_star(self, origin, destination, g=None, freeflow=False, destonly=False):

        """
        This method finds the shortest path in a network which may or may not have
        cycles; thus you cannot assume that a topological order exists.

        The implementation in the text uses a vector of backnode labels.  In this
        assignment, you should use back-LINK labels instead.  The idea is exactly
        the same, except you are storing the ID of the last *link* in a shortest
        path to each node.

        Use the 'cost' attribute of the Links to calculate travel times.  These values
        are given -- do not try to recalculate them based on flows, BPR functions, etc.

        The backlink and cost labels are both stored in dict's, whose keys are
        node IDs.

        *** BE SURE YOUR IMPLEMENTATION RESPECTS THE FIRST THROUGH NODE!
        *** Travelers should not be able to use "centroid connectors" as shortcuts,
        *** and the shortest path tree should reflect this.

        You should use the macro utils.NO_PATH_EXISTS to initialize backlink labels,
        and utils.INFINITY to initialize cost labels.
        """
        backnode = dict()
        L = dict()

        allnodes = []
        for i in self.node:
            backnode[i] = utils.NO_PATH_EXISTS
            L[i] = utils.INFINITY
            allnodes.append(i)

        L[origin] = 0

        F = []
        E = [origin]

        while True:
            try:
                pick = None
                mincost = np.inf
                for n in E:
                    if g==None:
                        curcost = L[n]
                    else:
                        curcost = L[n] + g[n]
                    if curcost <= mincost:
                        mincost = curcost
                        pick = n
                i = pick
                
                F.append(i)
                E.remove(i)

                if len(F) == len(allnodes):
                    break

                if g!=None and destination in F:
                    break
                if destonly and destination in F:
                    break


                jlist = []
                for ij in self.node[i].forwardStar:
                    j = self.link[ij].head
                    if i < self.firstThroughNode and i != origin:
                        continue
                    if freeflow:
                        tempCost = L[i] + self.link[ij].freeFlowTime
                    else:
                        tempCost = L[i] + self.link[ij].cost
                    if tempCost < L[j]:
                        L[j] = tempCost
                        labelChanged = True
                        backnode[j] = i
                    jlist.append(j)

                for j in jlist:
                    if (j not in F) and (i in F):
                        E.append(j)
                        E = list(set(E))
            except:
                pdb.set_trace()

        return (backnode, L)

    def find_g(self, destination):
        #for everynode find sp to s
        g = {}
        for i in self.node:
            if i != destination:
                bnode, cost = self.a_star(i, destination, freeflow=True)
                ### test a_star code with older hw assignment
                g[i] = cost[destination]
        g[destination] = 0
        return g

    def allOrNothing(self):
        """
        This method generates an all-or-nothing assignment using the current link
        cost values.  It must do the following:
           1. Find shortest paths from all origins to all destinations
           2. For each OD pairs in the network, load its demand onto the shortest
              path found above.  (Ties can be broken arbitrarily.)
        The resulting link flows should be returned in the allOrNothing dict, whose
        keys are the link IDs.

        Be aware that the network files are in the TNTP format, where nodes are numbered
        starting at 1, whereas Python starts numbering at 0.  

        Your code will not be scored based on efficiency, but you should think about
        different ways of finding an all-or-nothing loading, and how this might
        best be done.
        """
        allOrNothing = dict()
        for ij in self.link:
            allOrNothing[ij] = 0

        for origin in range(1, self.numZones + 1):
            (backlink, cost) = self.shortestPath(origin)
            for OD in [OD for OD in self.ODpair if self.ODpair[OD].origin == origin]:
                curnode = self.ODpair[OD].destination
                while curnode != self.ODpair[OD].origin:
                    allOrNothing[backlink[curnode]] += self.ODpair[OD].demand
                    curnode = self.link[backlink[curnode]].tail

        return allOrNothing

    def findLeastEnteringLinks(self):
        """
        This method should return the ID of the node with the *least* number
        of links entering the node.  Ties can be broken arbitrarily.
        """
        leastEnteringLinks = self.numLinks + 1
        leastEnteringNode = None
        for i in self.node:
            if len(self.node[i].reverseStar) < leastEnteringLinks:
                leastEnteringLinks = len(self.node[i].reverseStar)
                leastEnteringNode = i
        return leastEnteringNode

    def formAdjacencyMatrix(self):
        """
        This method should produce an adjacency matrix, with rows and columns
        corresponding to each node, and entries of 1 if there is a link connecting
        the row node to the column node, and 0 otherwise.  This matrix should
        be stored in self.adjacencyMatrix, which is a dictionary of dictionaries:
        the first key is the "row" (tail) node, and the second key is the "column"
        (head) node.
        """
        self.adjacencyMatrix = dict()
        for i in self.node:
            self.adjacencyMatrix[i] = dict()
            for j in self.node:
                self.adjacencyMatrix[i][j] = 0

        for ij in self.link:
            self.adjacencyMatrix[self.link[ij].tail][self.link[ij].head] = 1

    def findTopologicalOrder(self):
        """
        This method should find a topological order for the network, storing
        the order in the 'order' attribute of the nodes, i.e.:
           self.node[5].order 
        should store the topological label for node 5.

        The topological order is generally not unique, this method can return any
        valid order.  The nodes should be labeled 1, 2, 3, ... up through numNodes.

        If the network has cycles, a topological order does not exist.  The presence
        of cycles can be detected in the algorithm for finding a topological order,
        and you should raise an exception if this is detected.
        """
        # This implementation temporarily messes with reverse stars, must fix
        # at end
        numOrderedNodes = 0
        while numOrderedNodes < self.numNodes:
            nextNode = self.findLeastEnteringLinks()
            if len(self.node[nextNode].reverseStar) > 0:
                print("Error: Network given to findTopologicalOrder contains a cycle.")
                raise BadNetworkOperationException
            numOrderedNodes += 1
            self.node[nextNode].order = numOrderedNodes
            self.node[nextNode].reverseStar = [0] * self.numLinks
            for ij in self.node[nextNode].forwardStar:
                self.node[self.link[ij].head].reverseStar.remove(ij)

        # Repopulate reverse star list
        for i in self.node:
            self.node[i].reverseStar = list()
        for ij in self.link:
            self.node[self.link[ij].head].reverseStar.append(ij)


    def createTopologicalList(self):
        """
        Takes a topological ordering of the nodes, expressed by the 'order'
        attribute of the Node objects, and creates a single list which stores
        the IDs of the nodes in topological order.  This is essentially the
        inverse function of the topological order, the k-th element of this list
        gives you the ID of the node whose order value is k.  
        """
        sortedList = list(self.node.items())
        sortedList.sort(key=lambda item: item[1].order)
        self.topologicalList = [i[0] for i in sortedList]

        # Add dummy element, since topological order starts at 1.
        self.topologicalList = [utils.NO_PATH_EXISTS] + self.topologicalList

    def loadPaths(self):
        """
        This method should take given values of path flows (stored in the
        self.path[].flow attributes), and do the following:
           1. Set link flows to correspond to these values (self.link[].flow)
           2. Set link costs based on new flows (self.link[].cost), see link.py
           3. Set path costs based on new link costs (self.path[].cost), see path.py
        """
        for ij in self.link:
            self.link[ij].flow = 0
        for p in self.path:
            for ij in self.path[p].links:
                self.link[ij].flow += self.path[p].flow
        for ij in self.link:
            self.link[ij].updateCost()
        for p in self.path:
            self.path[p].updateCost()

    def __str__(self, printODData=False):
        """
        Output network data; by default prints link flows and costs.
        If printODData == True, will also print OD pair demand and equilibrium costs.
        """
        networkStr = "Link\tFlow\tCost\n"
        for ij in sorted(self.link, key=lambda ij: self.link[ij].sortKey):
            networkStr += "%s\t%f\t%f\n" % (ij,
                                            self.link[ij].flow, self.link[ij].cost)
        if printODData == True:
            networkStr += "\n"
            networkStr += "OD pair\tDemand\tLeastCost\n"
            for ODpair in self.ODpair:
                networkStr += "%s\t%f\t%f\n" % (
                    ODpair, self.ODpair[ODpair].demand, self.ODpair[ODpair].leastCost)
        return networkStr

    def readFromFiles(self, networkFile, demandFile):
        """
        Reads network data from a pair of files (networkFile, containing the topology,
        and demandFile, containing the OD matrix), then do some basic checks on
        the input data (validate) and build necessary data structures (finalize).
        """
        self.readNetworkFile(networkFile)
        if type(demandFile) == list:
            self.readMultDemandFile(demandFile)
        else:
            self.readDemandFile(demandFile)
        self.validate()
        self.finalize()

    def readNetworkFile(self, networkFileName):
        """
        Reads network topology data from the TNTP data format.  In keeping with
        this format, the zones/centroids are assumed to have the lowest node
        IDs (1, 2, ..., numZones).
        """
        try:
            with open(networkFileName, "r") as networkFile:
                fileLines = networkFile.read().splitlines()

                # Set default parameters for metadata, then read
                self.numNodes = None
                self.numLinks = None
                self.numZones = None
                self.firstThroughNode = 0
                metadata = utils.readMetadata(fileLines)

                try:
                    self.numNodes = int(metadata['NUMBER OF NODES'])
                    self.numLinks = int(metadata['NUMBER OF LINKS'])
                    if self.numZones != None:
                        if self.numZones != int(metadata['NUMBER OF ZONES']):
                            print(
                                "Error: Number of zones does not match in network/demand files.")
                            raise utils.BadFileFormatException
                    else:
                        self.numZones = int(metadata['NUMBER OF ZONES'])
                    self.firstThroughNode = int(metadata['FIRST THRU NODE'])
                except KeyError:  # KeyError
                    print(
                        "Warning: Not all metadata present, error checking will be limited and code will proceed as though all nodes are through nodes.")
                self.tollFactor = float(metadata.setdefault('TOLL FACTOR', 0))
                self.distanceFactor = float(
                    metadata.setdefault('DISTANCE FACTOR', 0))

                for line in fileLines[metadata['END OF METADATA']:]:
                    # Ignore comments and blank lines
                    line = line.strip()
                    commentPos = line.find("~")
                    if commentPos >= 0:  # strip comments
                        line = line[:commentPos]

                    if len(line) == 0:
                        continue

                    data = line.split()
                    if len(data) < 11 or data[10] != ';':
                        print("Link data line not formatted properly:\n '%s'" % line)
                        raise utils.BadFileFormatException

                    # Create link
                    linkID = '(' + str(data[0]).strip() + \
                        "," + str(data[1]).strip() + ')'

                    self.link[linkID] = Link(self,
                                             # head and tail
                                             int(data[0]), int(data[1]),
                                             float(data[2]),   # capacity
                                             float(data[3]),   # length
                                             float(data[4]),   # free-flow time
                                             float(data[5]),   # BPR alpha
                                             float(data[6]),   # BPR beta
                                             float(data[7]),   # Speed limit
                                             float(data[8]),   # Toll
                                             data[9])          # Link type

                    # Create nodes if necessary
                    if data[0] not in self.node:  # tail
                        self.node[int(data[0])] = Node(
                            True if int(data[0]) <= self.numZones else False)
                    if data[1] not in self.node:  # head
                        self.node[int(data[1])] = Node(
                            True if int(data[1]) <= self.numZones else False)

        except IOError:
            print("\nError reading network file %s" % networkFile)
            traceback.print_exc(file=sys.stdout)

    def readDemandFile(self, demandFileName):
        """
        Reads demand (OD matrix) data from a file in the TNTP format.
        """
        try:
            with open(demandFileName, "r") as demandFile:
                fileLines = demandFile.read().splitlines()
                self.totalDemand = 0

                # Set default parameters for metadata, then read
                self.totalDemandCheck = None

                metadata = utils.readMetadata(fileLines)
                try:
                    self.totalDemandCheck = float(metadata['TOTAL OD FLOW'])
                    if self.numZones != None:
                        if self.numZones != int(metadata['NUMBER OF ZONES']):
                            print(
                                "Error: Number of zones does not match in network/demand files.")
                            raise utils.BadFileFormatException
                    else:
                        self.numZones = int(metadata['NUMBER OF ZONES'])

                except KeyError:  # KeyError
                    print(
                        "Warning: Not all metadata present in demand file, error checking will be limited.")

                for line in fileLines[metadata['END OF METADATA']:]:
                    # Ignore comments and blank lines
                    line = line.strip()
                    commentPos = line.find("~")
                    if commentPos >= 0:  # strip comments
                        line = line[:commentPos]
                    if len(line) == 0:
                        continue

                    data = line.split()

                    if data[0] == 'Origin':
                        origin = int(data[1])
                        continue

                    # Two possibilities, either semicolons are directly after
                    # values or there is an intervening space
                    if len(data) % 3 != 0 and len(data) % 4 != 0:
                        print("Demand data line not formatted properly:\n %s" % line)
                        raise utils.BadFileFormatException

                    for i in range(int(len(data) // 3)):
                        destination = int(data[i * 3])
                        check = data[i * 3 + 1]
                        demand = data[i * 3 + 2]
                        demand = float(demand[:len(demand) - 1])
                        if check != ':':
                            print(
                                "Demand data line not formatted properly:\n %s" % line)
                            raise utils.BadFileFormatException
                        ODID = str(origin) + '->' + str(destination)
                        self.ODpair[ODID] = OD(origin, destination, demand)
                        self.totalDemand += demand

        except IOError:
            print("\nError reading network file %s" % networkFile)
            traceback.print_exc(file=sys.stdout)

    def readMultDemandFile(self, demandFileName):
        """
        Reads demand (OD matrix) data from multiple files in the TNTP format.
        """
        try:
            self.totalDemand = 0
            for item in demandFileName:
                with open(item, "r") as demandFile:
                    fileLines = demandFile.read().splitlines()

                    # Set default parameters for metadata, then read
                    self.totalDemandCheck = None

                    metadata = utils.readMetadata(fileLines)
                    try:
                        self.totalDemandCheck = float(metadata['TOTAL OD FLOW'])
                        if self.numZones != None:
                            if self.numZones != int(metadata['NUMBER OF ZONES']):
                                print(
                                    "Error: Number of zones does not match in network/demand files.")
                                raise utils.BadFileFormatException
                        else:
                            self.numZones = int(metadata['NUMBER OF ZONES'])

                    except KeyError:  # KeyError
                        print(
                            "Warning: Not all metadata present in demand file, error checking will be limited.")

                    for line in fileLines[metadata['END OF METADATA']:]:
                        # Ignore comments and blank lines
                        line = line.strip()
                        commentPos = line.find("~")
                        if commentPos >= 0:  # strip comments
                            line = line[:commentPos]
                        if len(line) == 0:
                            continue

                        data = line.split()

                        if data[0] == 'Origin':
                            origin = int(data[1])
                            continue

                        # Two possibilities, either semicolons are directly after
                        # values or there is an intervening space
                        if len(data) % 3 != 0 and len(data) % 4 != 0:
                            print("Demand data line not formatted properly:\n %s" % line)
                            raise utils.BadFileFormatException

                        for i in range(int(len(data) // 3)):
                            destination = int(data[i * 3])
                            check = data[i * 3 + 1]
                            ODID = str(origin) + '->' + str(destination)
                            demand = data[i * 3 + 2]
                            demand = float(demand[:len(demand) - 1])
                            if check != ':':
                                print(
                                    "Demand data line not formatted properly:\n %s" % line)
                                raise utils.BadFileFormatException
                            try:
                                self.ODpair[ODID].demand += demand
                            except KeyError:
                                self.ODpair[ODID] = OD(origin, destination, demand)
                                self.totalDemand += demand

        except IOError:
            print("\nError reading network file %s" % networkFile)
            traceback.print_exc(file=sys.stdout)

    def validate(self):
        """
        Perform some basic validation checking of network, link, and node
        data to ensure reasonableness and consistency.
        """
        valid = True

        # Check that link information is valid
        for ij in self.link:
            valid = valid and self.link[ij].head in self.node
            valid = valid and self.link[ij].tail in self.node
            if not valid:
                print("Error: Link tail/head not found: %s %s" %
                      (self.link[ij].tail, self.link[ij].head))
                raise utils.BadFileFormatException
            valid = valid and self.link[ij].capacity >= 0
            valid = valid and self.link[ij].length >= 0
            valid = valid and self.link[ij].freeFlowTime >= 0
            valid = valid and self.link[ij].alpha >= 0
            valid = valid and self.link[ij].beta >= 0
            valid = valid and self.link[ij].speedLimit >= 0
            valid = valid and self.link[ij].toll >= 0
            if not valid:
                print("Link %s has negative parameters." % ij)

        # Then check that all OD pairs are in range
        for ODpair in self.ODpair:
            (origin, destination) = (self.ODpair[
                ODpair].origin, self.ODpair[ODpair].destination)
            valid = valid and origin in self.node
            valid = valid and destination in self.node
            if not valid:
                print("Error: Origin/destination %s not found" % ODpair)
                raise utils.BadFileFormatException
            valid = valid and self.node[origin].isZone == True
            valid = valid and self.node[destination].isZone == True
            if not valid:
                print(
                    "Error: Origin/destination %s does not connect two zones" % str(ODpair))
                raise utils.BadFileFormatException
            valid = valid and self.ODpair[ODpair].demand >= 0
            if not valid:
                print("Error: OD pair %s has negative demand" % ODpair)
                raise utils.BadFileFormatException

        # Now error-check using metadata
        if self.numNodes != None and len(self.node) != self.numNodes:
            print("Warning: Number of nodes implied by network file %d different than metadata value %d" % (
                len(self.node), self.numNodes))
            self.numNodes = len(self.node)
        if self.numLinks != None and len(self.link) != self.numLinks:
            print("Warning: Number of links given in network file %d different than metadata value %d" % (
                len(self.link), self.numLinks))
            self.numLinks = len(self.link)
        if self.numZones != None and len([i for i in self.node if self.node[i].isZone == True]) != self.numZones:
            print("Warning: Number of zones given in network file %d different than metadata value %d" % (
                len([i for i in self.node if self.node[i].isZone == True]), self.numZones))
            self.numLinks = len(self.link)
        if self.totalDemandCheck != None:
            if self.totalDemand != self.totalDemandCheck:
                print("Warning: Total demand is %f compared to metadata value %f" % (
                    self.totalDemand, self.totalDemandCheck))

    def finalize(self):
        """
        Establish the forward and reverse star lists for nodes, initialize flows and
        costs for links and OD pairs.
        """
        # Establish forward/reverse star lists, set travel times to free-flow
        for i in self.node:
            self.node[i].forwardStar = list()
            self.node[i].reverseStar = list()

        for ij in self.link:
            self.node[self.link[ij].tail].forwardStar.append(ij)
            self.node[self.link[ij].head].reverseStar.append(ij)
            self.link[ij].cost = self.link[ij].freeFlowTime + self.link[ij].length * \
                self.distanceFactor + self.link[ij].toll * self.tollFactor
            self.link[ij].flow = 0

        for OD in self.ODpair:
            self.ODpair[OD].leastCost = 0
