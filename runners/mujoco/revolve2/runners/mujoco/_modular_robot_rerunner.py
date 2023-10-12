"""Rerun(watch) a modular robot in Mujoco."""

from typing import List, Optional, Union, Tuple
import math
import pickle
from random import Random, randint

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics import Terrain
"""from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)"""
"""
from revolve2.examples.sanity3P.optimizer import (
    EnvironmentActorController,
)
"""
from revolve2.core.physics.running import Batch, Environment, PosedActor, RecordSettings
from revolve2.runners.mujoco import LocalRunner


class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(
        self,
        robots: Union[ModularRobot, List[ModularRobot]],
        control_frequency: float,
        terrain: Terrain,
        simulation_time: int = 1000000,
        start_paused: bool = False,
        record_settings: Optional[RecordSettings] = None,
    ) -> None:
        """
        Rerun a single robot.

        :param robots: One or more robots to simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param terrain: The terrain to use.
        :param simulation_time: How long to rerun each robot for.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param record_settings: Optional settings for recording the runnings. If None, no recording is made.
        """
        if isinstance(robots, ModularRobot):
            robots = [robots]

        batch = Batch(
            simulation_time=simulation_time,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
        )

        for robot in robots:
            actor, controller = robot.make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController([controller]))
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            env.static_geometries.extend(terrain.static_geometry)
            batch.environments.append(env)

        runner = LocalRunner(headless=False, start_paused=start_paused)
        await runner.run_batch(batch, record_settings=record_settings)


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )

### This is a bit messy, beware

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

#Dimitri Imports
#from tensorflow import keras
import numpy as np
from datetime import datetime
from scipy.stats import qmc
import csv
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
import pandas as pd


# This is not exactly the same as the revolve class `revolve2.core.physics.environment_actor_controller.EnvironmentActorController`
class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controllerList: List[ActorController]

    def __init__(self, actor_controllerList: List[ActorController]) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for multiple actors in the environment.
        """
        self.actor_controllerList = actor_controllerList

        self.actorCount = 0
        self.cognitiveList = {}
        self.modelList = []
        self.configuration = [4,3,2]

        #This list is for accessing all the actors in a dataframe
        self.actFrame = pd.DataFrame(columns=['id', 'actor', 'preyPred','timeBorn','gridID','lastKiller'])
        self.actFrame.set_index('id')

        self.lastTime = (datetime.now().timestamp())

        #cutIndex = math.ceil(len(self.actor_controllerList) / 2)
        cutIndex = 1
        #Initialize each actor_controller with a NN:
        for ind,actor in enumerate(self.actor_controllerList):
            actor.controllerInit(ind,
                                 self.new_denseWeights(self.configuration),
                                 ("prey" if ind <= cutIndex else "pred"),
                                 )
            list_row = [ind,actor,actor.preyPred,actor.timeBorn,(0,0),actor.lastKiller]
            self.actFrame.loc[len(self.actFrame)] = list_row

            self.actorCount += 1

        #print(self.actFrame)


        self.updPreyPred()

        header = ['id', 'predprey', 'tag', 'position']
        data = [
            ['Albania', 28748, 'AL', 'ALB'],
            ['Algeria', 2381741, 'DZ', 'DZA'],
            ['American Samoa', 199, 'AS', 'ASM'],
            ['Andorra', 468, 'AD', 'AND'],
            ['Angola', 1246700, 'AO', 'AGO']
                ]

        with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(data)

        self.pushCollectData = []

    ###
    # Neural Network Functions
    ###

    #Importing existing libraries was buggy, so I'm making my own neural net infrastructure
    #TO-DO: make smart NN initalization choices
    def new_denseWeights(self,config):
        weights = []
        biases = []
        for ind in range(len(config)-1):
            weights.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind],config[ind+1])) )
            biases.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind+1],)) )
        return [ np.array(weights), np.array(biases)]
    
    #Allows us to make new mutated weight matrices from parents
    #alpha controls how harsh the mutations are
    def mutateWeights(self,weights,config,alpha=0.1):
        #Technically you could find the matrix size implicitly (and would be better design)
        mutWeights = weights.copy()
        for ind in range(len(config)-1):
            mutWeights[0][ind] += np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind],config[ind+1])) 
            mutWeights[1][ind] += np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind+1],)) 
        return mutWeights
    
    #Combines two parents' genotpye to make a child genotype
    #STILL HAVE TO TEST!
    def crossover(self,parent1,parent2):
        parent1W = parent1.copy()
        parent2W = parent1.copy()
        crossWeights = []
        crossBiases = []

        for ind in range(len(self.configuration)-1):
            #Crossover Point at a random interval in the next layer
            cutIndex = randint(1,self.configuration[ind+1]-1) 

            crossWeights.append(np.concatenate(parent1W[0][ind][:cutIndex],parent2W[0][ind][cutIndex:]))
            crossBiases.append(np.concatenate(parent1W[1][ind][:cutIndex],parent2W[1][ind][cutIndex:]))
        return [crossWeights, crossBiases]

    ###
    #Control Section: This is the place where the cake is put together
    ###
    def control(self, dt: float, actor_control: ActorControl, argList: List) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        """

        self.actorStates = argList

        #Passing info to the actor and asking it to control
        #Only pass info that is needed on every tick
        for ind, actor in enumerate(self.actor_controllerList):

            actor.passInfo(self.actorStates[ind],
                           self.get_grid_Tup(ind),
                           )
            actor.step(dt)
            actor_control.set_dof_targets(ind, actor.get_dof_targets())

        ##self.updateGrid()

        ## Time Based Section - doesnt update on every loop
        self.currTime = (datetime.now().timestamp())
        if float(self.currTime - self.lastTime) > 0.5:
            #print(self.currTime - self.lastTime)
            #print('predlist')
            #print(self.predList)
            #print(self.preyList)
            ##self.cognitiveActors(self.actorStates)
            ##self.writeMyCSV()
            #print(f"prey: %s" % self.preyList.index)
            #print(f"pred: %s" % self.predList.index)
            #print(self.actFrame.iloc[0])
            self.lastTime = (self.currTime)   

        #Loop for data collection
        if float(self.currTime - self.lastTime) > 0.3:
            raAct = randint(0,0)
            ##actor = self.actor_controllerList[raAct]

            ##datas = [actor.id,actor.preyPred,actor.tag,actor.bodyPos]
            #Normally, having a dynamically sized array especially with tons of data is a bad idea,
            #Luckily it gets emptied out fairly regularly so the ammortization isnt super harmful
            ##self.pushCollectData.append(datas)
            


    ###
    #Mechanics: this is where actors have their states changed according to the 
    #experiment setup ideas
    ###
        
    #Makes the brain switch from predator to prey and vice versa
    def switchBrain(self,id):
        print("change")
        actor = self.actor_controllerList[id]
        if actor.preyPred == "prey":
            #actor = self.actor_controllerList[self.preyList[id]]
            actor.preyPred = "pred"
            bestPred = self.bestGenotype("pred")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "pred"
        else:
            #actor = self.actor_controllerList[self.predList[id]]
            actor.preyPred = "prey"
            bestPred = self.bestGenotype("prey")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "prey"

        #bestPred = self.new_denseWeights(self.configuration)
        
        actor.weights = self.mutateWeights(bestPred,self.configuration)
        itsNow = datetime.now().timestamp()
        actor.timeBorn = itsNow

        #Update the actor dataframe's time
        self.actFrame.loc[id,"timeBorn"] = itsNow

        

        self.updPreyPred()

    #Handles the mechanics of who gets caught and who dies out
    def updateGrid(self):
        self.updateActFrame()
        self.updPreyPred()

        #Handles Death of Prey

        #All information retrieval needs to happen before changes are made
        #preyGrid = [(self.actor_controllerList[id]).gridID for id in self.preyList]
        #preyGrid = [actor.gridID for actor in self.preyList["actor"]]
        #predTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "pred"])
        
        caught = None
        pyL = self.preyList
        for pred in self.predList["actor"]:
            if caught != None or len(self.preyList < 2):
                break
            caughtList = pyL[(pyL.gridID == pred.gridID) & (pred.id != pyL.lastKiller)]
            caught = caughtList.index[0] if len(caughtList) > 0 else None
            #predGID = (self.actor_controllerList[pred]).gridID
            #if pred.gridID in preyGrid:
            #    caught = (self.preyList).index[preyGrid.index(predGID)]
            #else:
            #    caught = None
            if caught != None:
                #print(f"caught: %s " % caught)
                #dumbo
                #print(caught)
                self.switchBrain(caught)
                #Hopefully this fixes it
                self.actFrame.loc[caught,"lastKiller"] = pred.id
                self.updPreyPred()
                #preyGrid = [actor.gridID for actor in self.preyList["actor"]]
                #preyGrid = [(self.actor_controllerList[id]).gridID for id in self.preyList]
            else:
                lol = 0

        #Handles Death of Predator
        if len(self.predList) > 0:
            minTime = min(self.predList["timeBorn"])
            #predID = predTimes.index(minTime)
            predID = self.predList["timeBorn"].idxmin()
                #print(pred.timeBorn)
                #print(self.lastTime)
                #print(pred.timeBorn - self.lastTime)

            #I don't know why but caught seems to activate despite no prey?
            if float(self.lastTime - minTime) > self.predatorlifeSpan() and True:
                    #print(wenthere)
                    self.switchBrain(predID)
    
    #Signals our robots to cognitively determine the next target angle
    def cognitiveActors(self,actorStates):
        #actorDistList = [actor.position]
        for ind,actor in enumerate(self.actor_controllerList):
            posList = [other.bodyPos for other in self.actor_controllerList]
            distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
            smallest = min(distList)
            closestActor = self.actor_controllerList[distList.index(smallest)]


            
            angle = self.angleBetween(actor.bodyPos,closestActor.bodyPos)
            dumbo = 2
            #This is where we can pass any cognitive information, 
            # right now it is: 0-angle 1-distance, 2-tag, 3-dumbo (test variable)
            #tag = randint(0,9)
            #print(tag)
            actor.makeCognitiveOutput(angle,smallest,closestActor.tag,dumbo)

            #(self.actorStates[0].position)[:2]

    #Get the LifeSpan of 
    def predatorlifeSpan(self):
        predsLeft = len(self.predList)
        
        if predsLeft > 1:
            #currently set to a linear scale
            return 4.0  + (20-predsLeft)
        else:
            return 1000000000

    ###
    #Informational Functions
    ###

    #Returns a tuple for where the actor is on the grid
    def get_grid_Tup(self, id):
        position = self.actorStates[id].position
        #NEED FIX: I dont super understand why its messing up with values other than 10
        x = round(position[0] * 5)
        y = round(position[1] * 5)
        return (x, y)
    
    #Get the oldest genotypes
    def bestGenotype(self,preyPred):
        #Update the predator and prey lists before checking, its probably uneccessary though
        self.updPreyPred()
        
        #actorMe
        #preyTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "prey"])
        #predTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "pred"])
        #preyTimes = self.preyList["timeBorn"]
        if preyPred == "prey":
            #maxTime = min(preyTimes)
            #preyID = preyTimes.index(maxTime)
            #genoID = self.preyList[preyID]
            genoID = self.preyList['timeBorn'].idxmin()
            
        else:
            #maxTime = max(predTimes)
            #predID = predTimes.index(maxTime)
            #genoID = self.predList[predID]
            genoID = self.predList['timeBorn'].idxmin()

        return (self.actor_controllerList[genoID]).weights

    #Updates which are prey and which are predators
    def updPreyPred(self):
        #self.preyList = [ actor.id for actor in self.actor_controllerList if actor.preyPred == "prey"]
        self.preyList = self.actFrame.query("preyPred=='prey'")
        self.predList = self.actFrame.query("preyPred=='pred'")
        #print(self.predList)
        #self.predList = [ actor.id for actor in self.actor_controllerList if actor.preyPred == "pred"]

    #Finds the distance between two actors, return a super large distance if same position
    #so that an actor "ignores" itself in terms of distance
    def actorDist(self,pos1,pos2):
        x = pos1[0] - pos2[0]
        y = pos1[1] - pos2[1]
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


    ###
    # Utility Functions
    ###

    def writeMyCSV(self):
        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write multiple rows
            writer.writerows(self.pushCollectData)

    #Updates the actor dataframe
    def updateActFrame(self):
        #UDATE: gridID, 
        #for actor in self.actor_controllerList:
        #    break
        #    self.actFrame.loc[actor.id,"gridID"] = actor.gridID
        
        #Pandas doesnt like the above, not sure why
        self.actFrame['gridID'] = [actor.gridID for actor in self.actor_controllerList]

   
