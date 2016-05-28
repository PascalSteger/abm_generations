#!/usr/bin/env python3

##
# @file
# generations.py: Simulate and analyze a Bounded Confidence Model with finite Ages
# dependency: cluster, install via sudo pip install cluster
# the program will store output files into folders data/ and fig/
#
# run as
# # mkdir data fig
# # python3 generations.py
#
# (c) 2013 ETHZ Pascal Steger, psteger@phys.ethz.ch

import ipdb
import numpy as np
import numpy.random as npr
npr.seed(1989) # for random events that are reproducible
import cluster, pickle, datetime

from copy import copy, deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

numagents = 50        # number of agents
epsilon = 0.3         # Epsilon: threshold for interaction based on opinion difference

smoothedge = 0.       # future investigation: smooth the decline from 100% to 0% influence over this distance

zeta    = 0.1         # Zeta: convergence speed parameter
updatesPerStep = 100  # Number of updates per step to execute

maxage  = 40          # new variable: age
maxagespread = 0      # future investigation: with Gaussian distribution of this width

Tmax = 50            # Number of steps to simulate

# grouping characteristics
minNumberAgentsInGroup = max(2, 0.1*(numagents*(2*(epsilon+smoothedge))))

# store intermediate steps
histOpinions, histGroups, histAges = [], [], []

class Agent():
    def __init__(self, ID, date, opinion):
        self.ID = ID
        if date < 0:
            self.Birthday = 0-npr.randint(0, -date) # uniform random integer
        else:
            self.Birthday = date
            
        if opinion < 0:
            self.Opinion = npr.rand()
        else:
            self.Opinion = opinion

        self.histOpinion = []
        self.histOpinion.append(self.Opinion)

        self.Pos     = [npr.rand(), npr.rand()]
    ## \fn __init__(self)
    # birth, with zero age, and random opinion

    def __repr__(self):
        return "Agent "+str(self.ID)+', Birthday '+str(self.Birthday)+', opinion history: '+str(self.histOpinion)
    ## \fn __repr__(self)
    # return string for representation in ipython


    def age(self, date):
        return date-self.Birthday
    ## \fn age(self, date)
    # return age of an agent
    # @param date current time


class Population():
    def __init__(self, N):
        self.N = 0
        self.runningAgentID = 0
        self.runningGroupID = 0
        self.rate = 1./maxage # probability to spawn bunch of new agents at any timestep
        self.Agents = []
        self.Elders = []
        self.setup_uniform_age(N)
    ## \fn __init__(self, N)
    # set up population of agents

    
    def __repr__(self):
        return "Population with "+str(self.N)+" Agents and "+str(len(self.Elders))+" Elders."
    ## \fn __repr__(self)
    # return string for representation in ipython


    def setup_uniform_age(self, N):
        self.add_agents(N, date=-maxage, opinion=-1)
        return
    ## \fn setup_uniform_age(self, N)
    # create agents with age uniformly distributed between 0 and maxage
    # @param N number of agents


    def get_opinions(self):
        opinionlist = []
        for ind in range(self.N):
            opinionlist = np.append(opinionlist, self.Agents[ind].Opinion)
        return opinionlist
    ## \fn get_opinions(self)
    # return opinions of all agents


    def find_agents(self, mean, std):
        step = 1.
        ids = []
        for ag in self.Agents:
            if mean-step*std <= ag.Opinion <= mean+step*std:
                ids.append(ag.ID)
        return np.array(ids)
    ## \fn find_agents(self, mean, std)
    # locate all agents within bracket of the group, return their IDs
    # @param mean mean opinion
    # @param std standard deviation of opinions in group
    # @return ID list of all Agents in bracket [mean-std, mean+std]


    def get_groups(self, time, spread=smoothedge):
        ids = []
        for ind in range(self.N):
            local = self.Agents[ind]
            #ids = np.append(ids, [local.ID(), 1])
            ids = np.append(ids, local.Opinion)

        # find hierarchical clustering scheme
        o = ids.argsort()
        ids = ids[o]

        cl = cluster.HierarchicalClustering(ids, lambda x,y: (abs(x-y))**0.5) # penalize agents with big distance
        #clevel = cl.getlevel(1./np.sqrt(self.N))     # get clusters of items closer than 1/sqrt(N)
        # or better, use cutting criterium based on interaction strength (interaction strength reduced to 10%)
        if spread >= 0:
            clevel = cl.getlevel(self.distofinfluence(spread/10))
        else:
            clevel = cl.getlevel(-spread)
        numcluster = len(clevel)

        groups = []
        for i in range(numcluster):
            clr = np.ravel(clevel[i])
            mi = np.mean(clr)
            st = np.std(clr)
            agents = self.find_agents(mi, st)
            # a group has to have at least three agents to be considered
            if len(agents) < minNumberAgentsInGroup:
                continue
            self.runningGroupID += 1
            groups.append(OpinionGroup(self.runningGroupID, time, agents, mi, st))
        print('     ', len(groups),' groups')
        return groups
    ## \fn get_groups(self, spread=smoothedge)
    # identify groups in opinion space via hierarchical clustering
    # @param spread minimum distance between agents to classify as separate groups


    def get_ages(self, date):
        agelist = []
        for i in range(self.N):
            agelist = np.append(agelist, self.Agents[i].age(date))
        return agelist
    ## \fn get_ages(self, date)
    # get ages of all previously live agents at a given time
    # @param date current time


    def add_agents(self, n, date=-maxage, opinion=-1):
        for i in range(n):
            self.runningAgentID += 1
            self.Agents = np.append(self.Agents, Agent(self.runningAgentID, date, opinion))
            self.N += 1
        print(n, "     agents were born")
        return
    ## \fn add_agents(self, n, date=-maxage, opinion=-1)
    # spawn N agents to population
    # @param n number
    # @param date current date, if not given: distribute ages uniformly within maxage
    # @param opinion fixed opinion; if not given: distribute opinions uniformly within [0,1]


    def clear_old(self, date):
        mark_dead = []
        for i in range(self.N):
            if self.Agents[i].age(date) >= maxage:
                mark_dead.append(i)
        if (len(mark_dead)==0):
            return

        print(date,': ', len(mark_dead),'agents died')
        self.Elders = np.append(self.Elders, self.Agents[mark_dead])
        self.Agents = np.delete(self.Agents, mark_dead)
        self.N -= len(mark_dead)

        # update with the same number of new agents to keep the population constant
        self.add_agents(len(mark_dead), date, -1)
        return
    ## \fn clear_old(self, date)
    # remove agents that reached end of their lifetime
    # @param date current date, needed for age calculation

    
    def distance(self, agent1, agent2, date, alpha=1, beta=0, gamma=0):
        dist = 0
        dist += alpha * np.abs(self.Agents[agent2].Opinion -\
                               self.Agents[agent1].Opinion)
        dist += beta  * np.abs(self.Agents[agent2].age(date) -\
                               self.Agents[agent1].age(date))
        ### TODO: dist += gamma*EulerDistance(self.Agents[agent2].Pos, \
        #                                     self.Agents[agent1].Pos)
        return dist
    ## \fn distance(self, agent1, agent2, date, alpha=1, beta=0, gamma=0)
    # find social distance between agents
    # @param agent1 first agent
    # @param agent2 second agent
    # @param date current date
    # @param alpha weight of opinion difference, default is 1
    # @param beta  weight of age difference, default is 0 (neglect)
    # @param gamma weight of position difference, default is 0 (neglect)

    
    def opinion_diff(self, agent1, agent2):
        return self.Agents[agent2].Opinion - self.Agents[agent1].Opinion
    ## \fn opinion_diff(self, agent1, agent2)
    # get difference in opinion space only, with sign. Needed for strength determination
    # @param agent1 first agent
    # @param agent2 second agent
    

    def strength(self, distance, epsilon, spread=0.00):
        s = 1./(1.+np.exp((distance-epsilon)/(spread*epsilon)))
        return s
    ## \fn strength(distance, epsilon, spread=0.01)
    # return strength of interaction (bounded by 0 and 1)
    # attention: no normalization to 1 is done for high spreads,
    # so interaction strength at small distances compared to epsilon might be
    # quite weak as well
    # @param distance between to agents, float
    # @param epsilon relevant distance scale: below: preferring interaction
    # @param spread  allowance region for smoothing, in percent of epsilon


    def distofinfluence(self, spread=0.0):
        if spread<1.e-30:
            return epsilon
        return epsilon + (spread*epsilon) * np.log(1/(0.1*(     1./(1.+np.exp(-1./spread)          ))))
    ## \fn distofinfluence(self, spread=0.0)
    # determine interaction kernel size
    # @param spread = 0.0, Gaussian width of change of interaction strength


    def remember(self):
        for agent in self.Agents:
            agent.histOpinion.append(agent.Opinion)
        return
    ## \fn remember(self)
    # add Agent's opinions to global history variable for later plotting

    
    def add_new_boom(self, date):
        NumberNewAgents = self.rate/npr.rand() # TODO: want bunch of new agents?
        if (NumberNewAgents > 1):
            self.add_agents(int(NumberNewAgents), date, -1)
        return
    ## \fn add_new_boom(self, date)
    # alternative spawning recipe for new agents; baby-boom-like

    
    def update(self, date):
        self.clear_old(date)
        if (self.N <= 1):
            # population just died out
            return
        agent1 = npr.randint(1, self.N)-1
        agent2 = npr.randint(1, self.N)-1

        dist = self.distance(agent1, agent2, date)
        OpinionDiff = self.opinion_diff(agent1, agent2)
        s = self.strength(dist, epsilon, spread=smoothedge)
        
        self.Agents[agent1].Opinion += s * zeta * OpinionDiff
        self.Agents[agent2].Opinion -= s * zeta * OpinionDiff
        return
    ## \fn update(self, date)
    # cell state update function, interactions between random agents
    # @param date current simulation date

    
    def update_group(self, date):
        self.clear_old(date)
        if (self.N <= 1):
            return
        updatedAgents = np.copy(self.Agents)
        for i in range(self.N):
            change = 0.
            for j in range(self.N):
                distance    = self.distance(i, j, 1, 0, 0)
                OpinionDiff = self.opinion_diff(i, j)
                s = self.strength(OpinionDiff, epsilon)
                change += s * zeta * OpinionDiff
            updatedAgents[i].Opinion = self.Agents.Opinion[i] + change
        self.Agents = np.copy(updatedAgents)
        return
    ## \fn update_group(self, date)
    # alternative update function
    # using interactions between agent and all other agents
    # @param date current simulation date

    
class OpinionGroup():
    def __init__(self, ID, time, agents, meanOpinion, stdOpinion):
        self.ID = ID
        self.time = time
        self.Agents = agents
        self.meanOpinion = meanOpinion
        self.stdOpinion = stdOpinion
        self.successor = None
    ## \fn __init__(self, meanOpinion, stdOpinion)
    # set up OpinionGroup of like-minded agents
    # @param meanOpinion float, mean opinion as reported by cluster.getlevel
    # @param stdOpinion float, opinion standard deviation as reported by cluster.getlevel


    def __repr__(self):
        return "OpinionGroup "+str(self.ID)+", "+str(len(self.Agents))+\
            " Agents, mean="+str(self.meanOpinion)+\
                            " std="+str(self.stdOpinion)
    ## \fn __repr__(self)
    # return string for representation in ipython


    def find_successor(self, newgrouplist):
        N = len(newgrouplist)
        commonagents = np.zeros(N)
        for k in range(N):
            ID1 = self.Agents
            ID2 = newgrouplist[k].Agents
            commonagents[k] = sum(np.bincount(np.hstack([ID1, ID2]))>1)
        # successor is closest to current Opinion if no common agents
        if len(commonagents) == 0:
            self.successor = None
            return
        if max(commonagents) == 0:
            dist = [abs(g.meanOpinion-self.meanOpinion) for g in newgrouplist]
            if min(dist) < epsilon:
                self.successor = np.argmin(dist)
            else:
                self.successor = None # or none if too far away
        else: # else where most agents go into
            self.successor = np.argmax(commonagents)
    ## \fn find_successor(self, newgrouplist)
    # find successor given the new groups
    # successor is defined as group with the highest number of common agents
    # if no common agents found, generate a new group
    # each group gets a timeline, up to consumption
    # timelines will be plotted as before
    # @param newgrouplist array of new groups


class BCGenerations():
    def __init__(self):
        self.N = numagents                             # number of agents [1]
        self.pop = Population(self.N)
    ## \fn __init__(self)
    # setup simulation framework for a bounded confidence model with generations

    def __repr__(self):
        return "simulation with "+str(self.N)+" Agents"
    ## \fn __repr__(self)
    # return string for representation in ipython

    
    def simulate(self):
        for time in range(Tmax):
            self.update(time)
    ## \fn simulate(self)
    # and run simulation through all Tmax coarse timesteps


    def update(self, time):
        for j in range(updatesPerStep):
            #self.pop.add_new_boom(time)
            self.pop.remember()
            self.pop.update(time)          # 1:1 interactions between agents
            # self.update_group()  # interactions with all the members
        histOpinions.append(self.pop.get_opinions())
        histAges.append(self.pop.get_ages(time))
        newgroups = self.pop.get_groups(time, spread=-epsilon)
        if time > 0:
            for gp in histGroups[-1]:
                gp.find_successor(newgroups)
        histGroups.append(newgroups)
        return
    ## \fn update(self, time)
    # update ensemble of Agents for 1 timestep, with updatesPerStep interactions
    # @param time current simulation time


    def get_group_variability(self):
        workingGroups = deepcopy(histGroups)
        variability = []
        weights = []
        for k in range(len(workingGroups)):
            for ll in range(len(workingGroups[k])):
                startgroup = workingGroups[k][ll]
                if startgroup.meanOpinion < 0:
                    continue

                gr = startgroup
                ind = k
                DeltaT = Tmax*1.1 # default value, to be shortened if needed
                weight = 0

                while gr.successor is not None:
                    weight += len(gr.Agents)
                    dist = np.abs(startgroup.meanOpinion - (workingGroups[ind+1][gr.successor]).meanOpinion)
                    # compare distance to sigma of *last*
                    if dist >= 2*(workingGroups[ind+1][gr.successor]).stdOpinion:
                        DeltaT = (workingGroups[ind+1][gr.successor]).time-startgroup.time
                        break

                    # tag that opinionGroup is part of a chain and not to be investigated later
                    (workingGroups[ind+1][gr.successor]).meanOpinion = -1

                    # get new group to look at
                    gr = workingGroups[ind+1][gr.successor]
                    ind += 1

                variability.append(DeltaT)
                weights.append(weight/DeltaT)

        return variability, weights
    ## \fn get_group_variability(self)
    # return all time_of_changes as a list, with number of group members as weight function


    def show_evolution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_title("Evolution of Opinions in Generations")
        ax.axis([0, Tmax, 0, 1])
        for agent in self.pop.Elders:
            y = agent.histOpinion
            x = (np.arange(len(y))+agent.Birthday*updatesPerStep)/updatesPerStep
            ax.plot(x, y, color='black', alpha=0.2)

        for agent in self.pop.Agents:
            y = agent.histOpinion
            x = (np.arange(len(y))+agent.Birthday*updatesPerStep)/updatesPerStep
            ax.plot(x, y, color='black', alpha=0.2)
            
        # plot the opinion group merger tree
        for k in range(len(histGroups)-1):
            oldgroups = histGroups[k]
            newgroups = histGroups[k+1]
            for j in range(len(oldgroups)):
                oldgroup = oldgroups[j]
                if oldgroup.successor is None:
                    continue
                else:
                    newgroup = newgroups[oldgroup.successor]
                x = np.array([oldgroup.time, newgroup.time])
                y = np.array([oldgroup.meanOpinion, newgroup.meanOpinion])
                yerr = np.array([oldgroup.stdOpinion, newgroup.stdOpinion])
                ax.fill_between(x, y-yerr, y+yerr, color='blue', alpha=0.2, lw=1)
                ax.plot(x, y, '-o', color='blue', alpha=1, lw=2)
            
        ax.set_xlabel('Time [timesteps]')
        ax.set_ylabel('Opinion [a.u.]')
        plt.draw()
        filename = 'fig/g'+str(numagents)+'_'+str(maxage)+'_'+str(Tmax)+\
                   '_'+str(epsilon)+'_'+str(smoothedge)+'_'+str(zeta)+\
                   datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig( filename+'.png')
        pp = PdfPages( filename+'.pdf')
        pp.savefig(fig)
        pp.close()
        plt.show()#block=True)
    ## \fn show(self)
    # graphical representation of the opinion evolution history,
    # from each agent's individual timeline


    def show_variability(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_title("Variability of Opinion Groups")
        DeltaT, wei = self.get_group_variability()
        ax.hist(DeltaT, int(Tmax/3), weights=wei, alpha=0.7, color='black', normed=True)#2*np.sqrt(self.pop.runningGroupID))
        ax.axvline(x=maxage, lw=2, color='red')

        analyticdistro = [np.pi*epsilon*w/(4*zeta) for w in wei]
        #ax.hist(analyticdistro, lw=1, alpha=0.7, color='green', normed=True)
        ax.set_xlim([0, 1.2*Tmax])
        ax.set_xlabel('group coherence [timesteps]')
        ax.set_ylabel('Frequency [1]')
        plt.draw()
        filename = 'fig/g'+str(numagents)+'_'+str(maxage)+'_'+str(Tmax)+\
                   '_'+str(epsilon)+'_'+str(smoothedge)+'_'+str(zeta)+\
                   datetime.datetime.now().strftime("%Y%m%d%H%M")+'_var'
        plt.savefig( filename+'.png')
        pp = PdfPages( filename+'.pdf')
        pp.savefig(fig)
        pp.close()
        plt.show()#block=True)
    ## \fn show_variability(self)
    # show the time needed to change meanOpinion by more than 2 sigma


    def save(self):
        filename = 'data/g'+str(numagents)+'_'+str(maxage)+'_'+str(Tmax)+\
                   '_'+str(epsilon)+'_'+str(smoothedge)+'_'+str(zeta)+\
                   '_'+str(updatesPerStep)+'_'+\
                   datetime.datetime.now().strftime("%Y%m%d%H%M")+'_var'
        pickle.dump(self, open( filename, "wb" ) )
    ## \fn save(self)
    # save all variables in serialized form for later reference


if __name__ ==  "__main__":
    # first step: simulate anew
    a = BCGenerations()
    a.simulate()

    a.show_evolution()
    a.show_variability() # including calculation thereof
    a.save()

    # second step, analysis after the fact: use previous results
    # a = pickle.load( open('save/g'+str(numagents)+'_'+str(maxage)+'_'+str(Tmax)+\
    #               '_'+str(epsilon)+'_'+str(smoothedge)+'_'+str(zeta)+\
    #               '_'+str(updatesPerStep)+'_'+\
    #               datetime.datetime.now().strftime("%Y%m%d%H%M")+'_var'))
