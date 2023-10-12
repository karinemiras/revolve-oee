"""Optimizer for finding a good modular robot body and brain using CPPNWIN genotypes and simulation using mujoco."""

import math
import pickle
from random import Random, randint
from typing import List, Tuple

import multineat
import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
import sqlalchemy
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
import quaternion as qt
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import DbId
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
#from revolve2.core.physics.environment_actor_controller import (
#    EnvironmentActorController,
#)
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    ActorState,
    ActorControl,
    Batch,
    Environment,
    PosedActor,
    EnvironmentController,
    Runner,
)
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select


import numpy as np
from datetime import datetime
from scipy.stats import qmc
import csv
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
import pandas as pd

#Some warning show up repeatedly, this ensures we aren't spammed in the terminal
import warnings
with warnings.catch_warnings():
    warnings.warn("Let this be your last warning")
    warnings.simplefilter("ignore")

import random


# This is not exactly the same as the revolve class `revolve2.core.physics.environment_actor_controller.EnvironmentActorController`
class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controllerList: List[ActorController]

    def __init__(self, actor_controllerList: List[ActorController]) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for multiple actors in the environment.
        """

        #This is to make sure we have a different rng each time
        #IMPORTANT: if you do repeat experiments, dont forget to change the seed
        random.seed(9)
        np.random.seed(9)
        self.actor_controllerList = actor_controllerList

        #Global Level Variables
        self.actorCount = 0
        self.cognitiveList = {}
        self.modelList = []
        self.configuration = [3,3,2]

        #Experiment Time
        self.maxSimTime = 6500


        #DEPRECATED
        self.preyImm = 10

        #This list is for accessing all the actors in a dataframe format
        self.actFrame = pd.DataFrame(columns=['id', 'actor', 'preyPred','timeBorn','lifeTime','gridID','lastKiller'])
        self.actFrame.set_index('id')

        #Initializing Time Variables
        self.lastTime = (datetime.now().timestamp())
        self.currTime = (datetime.now().timestamp())
        self.simStartTime = (datetime.now().timestamp())
        self.predDeathTime = (datetime.now().timestamp())
        self.preyDeathTime = (datetime.now().timestamp())
        self.lastTagTime = (datetime.now().timestamp())

        #We try to make roughly half of the robots prey and half predators
        cutIndex = math.ceil(len(self.actor_controllerList) / 2)
        
        #Initialize each actor_controller with a Neural Network (NN)
        for ind,actor in enumerate(self.actor_controllerList):
            actor.controllerInit(ind,
                                 self.new_denseWeights(self.configuration),
                                 ("prey" if ind <= cutIndex else "pred"),
                                 )
            #actFrame is a controller database that stores all the important values that we want to compare
            #between individuals
            list_row = [ind,actor,actor.preyPred,actor.timeBorn,actor.lifeTime,(0,0),actor.lastKiller]
            self.actFrame.loc[len(self.actFrame)] = list_row

            self.actorCount += 1

        self.updPreyPred()

        #Each string represents a column of the data we collect
        header = ['id', 'simTime', 'position','angle','predprey', 'tag',"otherID","immCheck","RanW",'closestAlly']


        #Writes the header for our positional data
        with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)


        #Makes the header for our death & birth data
        headerDeath = ['id', 'simTime','predprey', 'lifespan','RanW','caughtBy','byRanW']
        with open('deathBorn.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(headerDeath)

        self.pushCollectData = []

    ###
    # Neural Network Functions
    ###

    #Returns new random weights if given the configuration we want, i.e. subsequent layer sizes
    def new_denseWeights(self,config):
        weights = []
        biases = []
        #All weights are chosen to be between -1 and 1
        for ind in range(len(config)-1):
            weights.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind],config[ind+1])) )
            biases.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind+1],)) )
        return list([ weights, biases])
    
    #Allows us to make new mutated weight matrices from parents
    #alpha controls how harsh the mutations are
    def mutateWeights(self,weights,config,alpha=0.05):
        #Technically you could find the matrix size implicitly (and would be better for general use), but this is easier for us
        mutWeights = weights.copy()

        for ind in range(len(config)-1):
            # a represents the main weights
            a = np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind],config[ind+1])) 
            mutWeights[0][ind] += a
            mutWeights[0][ind] = np.clip(mutWeights[0][ind],-1.0,1.0)
            # b is for the bias weights
            b = np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind+1],)) 
            mutWeights[1][ind] += b
            mutWeights[1][ind] = np.clip(mutWeights[1][ind],-1.0,1.0)
        return mutWeights
    
    #DEPRECATED: Combines two parents' genotpye to make a child genotype
    def myCrossover(self,parent1,parent2):
        #To make sure we don't modify the original arrays, we use the copy() function
        parent1W = np.copy(parent1)
        parent2W = np.copy(parent2)
        crossWeights = []
        crossBiases = []

        for ind in range(len(self.configuration)-1):
            #Crossover Point at a random interval in the next layer
            cutIndex = randint(1,self.configuration[ind+1]-1) 

            #Not every layer may have the same size of weight arrays
            crossWeights.append(np.concatenate((parent1W[0][ind][:cutIndex],parent2W[0][ind][cutIndex:]))    )
            crossBiases.append(np.concatenate((parent1W[1][ind][:cutIndex],parent2W[1][ind][cutIndex:])))
        return list([ crossWeights, crossBiases])

    ###
    #Control Section: This is the place where the cake is put together
    ###
    def control(self, dt: float, actor_control: ActorControl, argList: List) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        """

        #We collect data about the actors from the Mujoco environment
        self.actorStates = argList

        #We only pass info here that is needed on every tick, so not the inputs to the NN
        #It needs to know its current angle, position, and the time step in order to move
        for ind, actor in enumerate(self.actor_controllerList):

            actor.passInfo(self.actorStates[ind],
                           self.get_grid_Tup(ind),
                           )
            actor.step(dt)
            actor_control.set_dof_targets(ind, actor.get_dof_targets())

        

        #Here we retrieve the current time, useful for comparison to past times
        self.currTime = (datetime.now().timestamp())

        #The main loop for most of our mechanisms, updates every 2 seconds
        if float(self.currTime - self.lastTime) > 2:
            
            #Checks if newborn prey are far away enough (5 units) to predators such that they can activate
            for actor in self.actor_controllerList:
                if actor.immCheck == False and actor.preyPred == "prey":
                    posList = [other.bodyPos for other in self.predList["actor"]]
                    distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
                    smallest = min(distList)
                    if smallest > 5:
                        actor.immCheck = True
                    else:
                        actor.timeBorn = self.currTime


            self.updateGrid()
            
            #Make all the actors decide their target angle and tag
            self.cognitiveActors(self.actorStates)

            #Mapping positional data is all done inside the dedicated function
            self.positionMap()

            #Resets the time loop
            self.lastTime = (self.currTime)   
            


    ###
    #Mechanics: this is where actors have their states changed according to the 
    #experiment setup ideas
    ###
        
    #Makes the brain switch from predator to prey and vice versa, i.e. the death function
    def switchBrain(self,id):
        actor = self.actor_controllerList[id]
        self.deathBornCSV(id,actor.preyPred,actor.timeBorn,actor.lastKiller)

        #Our selection process as a new predator says we should take the weight of whoever killed us
        if actor.preyPred == "prey":
            if actor.lastPredWeights != None:
                bestGeno = actor.lastPredWeights
            else:
                bestGeno = self.new_denseWeights(self.configuration)
            actor.preyPred = "pred"
            secondBest = self.bestGenotype("pred")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "pred"
            #Predators are not immune
            actor.immCheck = True
        else:

            actor.preyPred = "prey"

            #Our selection process as a new prey states that we get the weights from the closest prey
            if actor.closestPreyW != None:
                bestGeno = actor.closestPreyW
            else:
                bestGeno = self.new_denseWeights(self.configuration)


            self.actFrame.loc[id,"preyPred"] = "prey"
            #Prey are immune at birth
            actor.immCheck = False

        reproChance = np.random.uniform(0.0,1.0)

        #With 66% chance, we follow our open-ended selection process
        if reproChance < 0.66:
            #We also mutate weights by adding a scaled down version to the existing weights
            actor.weights = self.mutateWeights(bestGeno,self.configuration)
            actor.hasRanW = False
        #with 34% chance, we select new random weights
        else:
            actor.weights = self.new_denseWeights(self.configuration)
            actor.hasRanW = True

        #Time initializations
        itsNow = datetime.now().timestamp()
        actor.timeBorn = itsNow
        actor.lifeTime = itsNow
        actor.lastTag = itsNow

        #Update the actor's birth time
        self.actFrame.loc[id,"timeBorn"] = itsNow
        self.actFrame.loc[id,"lifeTime"] = itsNow
        self.updateActFrame()
        self.updPreyPred()

    #Handles the mechanics of who gets caught and who dies out
    def updateGrid(self):

        #If the simulation reaches its end time, it will stop by manually giving it an error
        if self.currTime - self.simStartTime > self.maxSimTime:
            iWillStop

        #All information retrieval needs to happen before changes are made
        self.updateActFrame()
        self.updPreyPred()
        
        caught = None
        pyL = self.preyList

        #DEPRECATED: finds the closest fellow prey for each prey
        for prey in self.preyList["actor"]:
            preyList = [actor for actor in self.preyList["actor"] if actor.immCheck ]
            posList = [other.bodyPos for other in preyList]
            distList = [self.actorDist(prey.bodyPos,pos) for pos in posList]
            if len(distList) > 0:
                prey.smallAllyD = min(distList)
                prey.smallAllyID =(preyList[distList.index(prey.smallAllyD)]).id

        #To avoid ordering bias in the predator IDs, we shuffle the list everytime
        pdIndexes = [i for i in range(len(self.predList["actor"]))]
        l_shuffled = random.sample(pdIndexes, len(pdIndexes))

        for predIND in l_shuffled:
            pred = self.predList["actor"].iloc[predIND]

            #Young predators, i.e. less than 20 seconds old, cannot hunt just yet
            if self.currTime - pred.timeBorn < 20:
                continue

            #caught will capture the actor ID of a prey that got caught
            caught = None

            #DEPRECATED: finds the closest fellow predator
            predList = [i for i in self.predList["actor"] if self.currTime - i.timeBorn > 30 ]
            posList = [other.bodyPos for other in predList]
            distList = [self.actorDist(pred.bodyPos,pos) for pos in posList]
            if len(distList) > 0:
                pred.smallAllyD = min(distList)
                pred.smallAllyID =(predList[distList.index(pred.smallAllyD)]).id

            #In case there are not enough prey, we put a halt on hunting
            if len(self.preyList.index) < 7:
                break

            #Scans the list of prey for the closest one to the current actor
            preyList = [actor for actor in self.preyList["actor"] if actor.immCheck ]
            posList = [other.bodyPos for other in preyList]
            distList = [self.actorDist(pred.bodyPos,pos) for pos in posList]
            
            #We find the distance of the closest prey
            #Seting the smallest distance to 1000 ensures it is never used
            if len(distList) > 0:
                smallest = min(distList)
            else:
                smallest = 1000

            if smallest < 1:
                #If the prey is within 1 unit of distance to the predator, it may be caught
                caught = (preyList[distList.index(smallest)]).id

            if caught != None:
                pred.closestPrey = None
                pred.closestPreyW = None


                #Updates regarding the time value when the predator is born
                itsNow = (datetime.now().timestamp())
                pred.lifeTime = itsNow
                self.actFrame.loc[pred.id,"lifeTime"] = itsNow
                (self.actor_controllerList[caught]).lastPredWeights = pred.weights
                
                #Updates the database and individual class object with what predator killed them
                self.actFrame.loc[caught,"lastKiller"] = pred.id
                self.actor_controllerList[caught].lastKiller = pred.id
                
                #Kills the Prey
                self.switchBrain(caught)
                
                self.updPreyPred()

        #Handles Death of Predator: those who don't see a prey within 5 units are eligible for death
        #The one who has spent the most time since last catching a prey dies first
        if len(self.predList) > 7 and (self.currTime - self.predatorlifeSpan() > self.predDeathTime):
            #This is always a true conditional
            if np.random.uniform(0.0,1.0) < 1.0:
                oldest = self.currTime
                predID = None
                for pred in self.predList["actor"]:
                    #If someone is close to a prey, they feel hope and dont die just yet
                    if pred.smallDist < 5:
                        continue
                    if oldest > pred.lifeTime:
                        oldest = pred.lifeTime
                        predID = pred.id
                if predID == None:
                    predID = self.predList["lifeTime"].idxmin()
            #DEPRECATED
            else:
                smallDistG = 10
                for predIND in l_shuffled:
                    pred = self.predList["actor"].iloc[predIND]    

                    if pred.smallAllyD <= smallDistG:
                        smallDistG = pred.smallAllyD
                        predID = pred.smallAllyID
            self.switchBrain(predID)
            self.predDeathTime = (datetime.now().timestamp())
        
        #A Random Prey May Die if there are not enough predators
        if len(self.predList) <= 7:
            preyID = random.choice(self.preyList.index)
            #The switchBrain function is how the individual "dies"
            self.switchBrain(preyID)
    
    #Signals our robots to cognitively determine the next target angle
    def cognitiveActors(self,actorStates):
        #Input 3: Tag ratio is the same for a all actors, so we calculate it once here
        tagRatio = self.getTagRatio()

        for ind,actor in enumerate(self.actor_controllerList):
            
            #This block scans all the viable actors, gets their distance to our current subject, and find the 
            #one with the minimal distance
            if actor.preyPred == "prey":
                #Viable actors means that we need of the opposite species, same tag, not immune
                viableOther = [pred for pred in self.predList["actor"] if (actor.tag == pred.tag) ]
            else:
                viableOther = list(filter(lambda other: ((actor.tag == other.tag) and ((other.immCheck) and (other.preyPred == 'prey'))), self.actor_controllerList))
            posList = [other.bodyPos for other in viableOther]
            distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
            if len(distList) > 0:
                smallest = min(distList)
            else:
                continue

            closestActor = self.actor_controllerList[(viableOther[distList.index(smallest)]).id]
            actor.closestID = closestActor.id
            actor.smallDist = smallest

            #To find the vector to the closest adversary, we can simply take the difference in position
            closestVector =  np.array(closestActor.bodyPos[:2]) - np.array(actor.bodyPos[:2])

            #Converting the vector into an angle (starting from  the negative x axis)
            standardAngle = self.angleBetween(closestVector,[-1.0,-0.0])
        
            #We need to find the difference angle between the rotation of the actor now, and its target
            angle = self.goodAngle(actor.bodyA,standardAngle)
            
            #Input 1: Finds wether the closest adversary is on the right or left of the current actor
            if actor.preyPred == "prey":
                angle = self.modusAng(angle+math.pi)

            if angle > 0:
                LeftR = 1
            else:
                LeftR = -1

            #DEPRECATED: Normalizes the angle to fit in between -1 and 1
            angleNorm = angle / math.pi

            #Input 2: Distance to the closest adversary
            inDist = np.clip(smallest/20,0.0,1.0)


            #Updates which prey are closest at any time
            if actor.preyPred == "pred":
                if actor.closestPrey != closestActor.id:
                    actor.closestPrey = closestActor.id
                    actor.closestPreyW = closestActor.weights

            #DEPRECATED: Finds the angle of the closest Ally
            closestAlly = self.actor_controllerList[actor.smallAllyID]
            closestVector =  np.array(closestAlly.bodyPos[:2]) - np.array(actor.bodyPos[:2])
            standardAngle = self.angleBetween(closestVector,[-1.0,-0.0])
            angleAlly = (self.goodAngle(actor.bodyA,standardAngle)) / math.pi

            #Passing along time information that the actor cannot access itself
            actor.currTime = self.currTime

            #We here feed our inputs to the cognitive brain so it can make decisions on the angle and the tag it chooses
            actor.makeCognitiveOutput(LeftR,inDist,tagRatio)


    #The lifespan of predators gets shorter the more that there are of them
    def predatorlifeSpan(self):
        predsLeft = len(self.predList)
        
        if predsLeft > 2:
            #currently set to a linear scale
            return 50 - predsLeft*2
        else:
            return 1000000000

    #DEPRECATED    
    def preylifeSpan(self):
        preysLeft = len(self.preyList)
        
        if preysLeft > 3:
            #currently set to a linear scale
            return 90 - 2*preysLeft
        else:
            return 1000000000

    ###
    #Informational Functions
    ###

    #Returns a tuple for where the actor is on the grid : DEPRECATED
    def get_grid_Tup(self, id):
        position = (self.actorStates[id].position)
        x = round(position[0] * 0.2)
        y = round(position[1] * 0.2)
        return (x, y)
    
    #Get the oldest genotypes
    def bestGenotype(self,preyPred):
        #Updates the predator and prey lists before checking, its probably uneccessary though
        self.updPreyPred()
        
        #Depending on the species, the best genotype is the closest prey, or a random predator
        if preyPred == "prey":
            bestDist = 0
            for prey in self.preyList['actor']:
                if prey.immCheck == False:
                    continue
                closestPos = (self.actor_controllerList[prey.closestID]).bodyPos
                closestDist = (prey.bodyPos[0] - closestPos[0])**2 + (prey.bodyPos[1] - closestPos[1])**2  
                if closestDist > bestDist:
                    bestDist = closestDist
                    genoID = prey.closestID      
        else:
            genoID = list(random.choices(self.predList.index, k=1, weights=(self.currTime - self.predList['timeBorn']) ) )[0]
        return (self.actor_controllerList[genoID]).weights

    #Updates which are prey and which are predators
    def updPreyPred(self):
        self.preyList = self.actFrame.query("preyPred=='prey'")
        self.predList = self.actFrame.query("preyPred=='pred'")

    #Will be passed as one of the NN inputs, the more positive the ratio, the more +1 tags there are
    def getTagRatio(self):
        count = 0
        plusTag = 0
        for actor in self.actor_controllerList:
            if actor.immCheck == False:
                continue
            count += 1
            if actor.tag == 1:
                plusTag += 1

        half = count / 2
        return (plusTag - half) / count


    #Finds the distance between two actors, return a super large distance if same position
    #so that an actor "ignores" itself in terms of distance
    def actorDist(self,pos1,pos2):
        y = pos1[0] - pos2[0]
        x = pos1[1] - pos2[1]
        dist = math.sqrt( (x**2 + y**2) )
        if dist > 0.01:
            return dist
        else:
            return 10000000
        
    #This is the way to calculate angle WITH direction
    def angleBetween(self,v1,v2):
        dot = np.dot(v1,v2)                     #Dot Product
        det = (v1[0]*v2[1] - v2[0]*v1[1])       # Determinant
        angle = math.atan2(det, dot)            # atan2(y, x) or atan2(sin, cos)
        return angle

    #DEPRECATED
    def modusAngle(self,ang1,ang2):
        diff = (ang2+math.pi) - (ang1+math.pi)
        modused = (diff % (2*math.pi)) - math.pi
        return modused
    
    #We cant just subtract two angles to get a target angle, it causes some phase issues
    #This function takes care of the phase such that we get the smallest representative angle
    def goodAngle(self,ang1,ang2):
        modus = ang2 - ang1
        if abs(modus) > math.pi:
            modus += -2*math.pi*np.sign(modus)
        return modus

    #Sometimes angles exceed pi to the way we get our angles from the Quaternion
    #This function translates such extreme angles to be absolutely less than pi
    def modusAng(self,ang):
        phase = ang + math.pi
        modused = (phase % (2*math.pi)) - math.pi
        return modused


    ###
    # Utility Functions
    ###

    #DEPRECATED
    def writeMyCSV(self):
        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write multiple rows
            writer.writerows(self.pushCollectData)

    #Here is all the output data needed for positional data
    def positionMap(self):
        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            simTime = self.currTime - self.simStartTime

            # write multiple rows
            for actor in self.actor_controllerList:
                newDataLine = [actor.id,simTime,actor.bodyPos,actor.bodyA,actor.preyPred,actor.tag,actor.closestID,actor.immCheck,actor.hasRanW,actor.smallAllyID] 
                writer.writerow(newDataLine)

    #Handles data regarding when and how predators/prey die
    def deathBornCSV(self,id,preyPred,timeBorn,caughtBy):
        with open('deathBorn.csv', 'a', encoding='UTF8', newline='') as f:
            #Some variables are retrieved on the spot
            myActor = self.actor_controllerList[id]
            myBy = self.actor_controllerList[caughtBy]
            writer = csv.writer(f)
            lifespan = self.currTime - timeBorn
            simTimeNow = self.currTime - self.simStartTime

            writer.writerow([id,simTimeNow,preyPred,lifespan,myActor.hasRanW,caughtBy,myBy.hasRanW])



    #The grid needs to be updated so that we know where all the robots are
    def updateActFrame(self):
        self.actFrame['gridID'] = [actor.gridID for actor in self.actor_controllerList]


    

        
###     ###     ###     ###     ###     ###     ###     ###     ###
###     ###     ###     ###     ###     ###     ###     ###     ###
###################################################################
###################################################################
######################THE GREAT WALL OF CODE#######################
###################################################################
###################################################################
###################################################################

    




class Optimizer(EAOptimizer[Genotype, float]):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    #_TERRAIN = terrains.flat()
    #_TERRAIN = terrains.crater((20,20),1,1)
    _TERRAIN = terrains.jail()


    _db_id: DbId

    _runner: Runner

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        initial_population: List[Genotype],
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param initial_population: List of genotypes forming generation 0.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :param offspring_size: Number of offspring made by the population each generation.
        """
        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._db_id = db_id
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :returns: True if this complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        if not await super().ainit_from_database(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
        ):
            return False

        self._db_id = db_id
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.db_id == self._db_id.fullname)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=True)

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                2,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        new_individuals: List[Genotype],
        new_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda n, genotypes, fitnesses: selection.multiple_unique(
                n,
                genotypes,
                fitnesses,
                lambda genotypes, fitnesses: selection.tournament(
                    self._rng, fitnesses, k=2
                ),
            ),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId,
    ) -> List[float]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        for genotype in genotypes:


            db = open_async_database_sqlite("./walkDatabase")
            async with AsyncSession(db) as session:
                best_individual = (
                    await session.execute(
                        select(DbEAOptimizerIndividual, DbFloat)
                        .filter(DbEAOptimizerIndividual.fitness_id == DbFloat.id)
                        .order_by(DbFloat.value.desc()))
                ).first()

                assert best_individual is not None

                #print(f"fitness: {best_individual[1].value}")

                genotype = (
                    await GenotypeSerializer.from_database(
                        session, [best_individual[0].genotype_id]
                    )           
                )[0]

            actor, controller = develop(genotype).make_actor_and_controller()
            
            #IMPORTANT: you can add or subtract the number of actors in the experiment by
            #simply changing numberAgents
            numberAGENTS = 30
            controllerList = []
            for i in range(numberAGENTS):
                actor, controller = develop(genotype).make_actor_and_controller()
                controllerList.append(controller)
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controllerList))
            env.static_geometries.extend(self._TERRAIN.static_geometry)

            #DEPRECATED: poisson positioning to make sure actors arent too close to each other
            radius = 0.05
            engine = qmc.PoissonDisk(d=2, radius=radius)
            sample = engine.random(numberAGENTS)

            

            #each actor is placed at a completely random position within the terrain
            for i in range(len(controllerList)):
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3(
                            [
                                np.random.uniform(-1.0,1.0)*9*1,
                                np.random.uniform(-1.0,1.0)*9*1,
                                bounding_box.size.z / 2.0 - bounding_box.offset.z + i*0,
                            ]
                        ),
                        Quaternion(),
                        [0.0 for _ in controller.get_dof_targets()],
                    )
                )    
            batch.environments.append(env)
        batch_results = await self._runner.run_batch(batch)

        return [
            self._calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        """ return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        ) """
        print(f"Fitness: %s " % float(end_state.position[0]*-1))
        return float(end_state.position[0]*-1)

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                db_id=self._db_id.fullname,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    """Optimizer state."""

    __tablename__ = "optimizer"

    db_id = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
