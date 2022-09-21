# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food = newFood.asList()
        foodDist = [manhattanDistance(successorGameState.getPacmanPosition(), x) for x in newFood.asList()]
        ghostDist = [manhattanDistance(successorGameState.getPacmanPosition(), x.configuration.pos) for x in newGhostStates]
        closestGhost = min(ghostDist)
    
        if not food:
            return 100000
        if closestGhost == 0 or action == Directions.STOP:
            return -1000000
          
        foodWeight = int((1/min(foodDist)) * 20)
        ghostWeight = int((1/ closestGhost) * 5)
        return successorGameState.getScore() + foodWeight - ghostWeight
      
        # food_count = 0 
        #Penalize if closer ghost 1/min(ghost)
        #Find current score - Ghost Position + 1/ Closest Food 
        # for x in food:
        #     print(food)
        #Find closest food
        #Distance to ghost to successor state is not zero 
        #If it is zero 
        #If action is stop 
        # food_count += len(food)
        # food_count = (1 / food_count) * 1000
        # food_count = int(food_count)
      
        # for x in newGhostStates:
        # ghostDist.append(manhattanDistance(successorGameState.getPacmanPosition, x))

    
        #Method will be called on a state and action 
        #Variables based off new state to generate a number that represents value
        #How far is pacman from food
        #How much food is left less (design given a number of food than generate off that)
        #1/food * weight or -food * weight
        #Count distance from ghost maybe prioritize distance from ghost
        #If ghost is scared there might be enough time for Pacman to catch up and he could eat it

        #Make a variable where food is by doing CurrentGameState.getFood() if u have asList use for loop to loop over food
        #Then use manhattan distance to know food location call function legalaction for every action pacman will get a new position
        #caculate manhattan distance from new position to food poistion
        #Also keep track of ghosts check the successors negative infinity if in same boat with ghost neg float 


        # lst = successorGameState.getGhostPosition() - also use getScore distance to nearest food and distance to 0 index ghost create 
        # for x in lst: shortest distance to food 
        # var = manhattanDistance(successorGameState.getPacmanPosition(), newGhostStates[0].configuration.pos)
        
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def maxValue(self, gameState: GameState, numGhosts: int, agentIndex: int):
        v = -float('inf')
        # scoredMoves = {} # key-value pairs of scores and actions
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.getActionWithIndex(successor, agentIndex + 1))
        #     scoredMoves[v] = action
        # bestScore = max(scoredMoves.keys())
        return v
        # return scoredMoves[bestScore]
        return max(self.getActionWithIndex(
              [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)], 1, currentDepth))
        
    def minValue(self, gameState: GameState, numGhosts: int, agentIndex):
        v = float('inf')
        # complete this for numGhosts times
        # scoredMoves = {}
        while agentIndex <= numGhosts:
          for action in gameState.getLegalActions(agentIndex):
              successor = gameState.generateSuccessor(agentIndex, action)
              v = min(v, self.getActionWithIndex(successor, agentIndex))
              # scoredMoves[v] = action
          agentIndex += 1
        # print(scoredMoves)
        # worstScore = min(scoredMoves.keys())
        self.depth -= 1
        return v
    
    def getActionWithIndex(self, gameState: GameState, agentIndex: int, currentDepth: int):
        """ 
        Helper function for getAction that initializes agentIndex as 0.
        """
        # If at the maximum depth or the game is complete
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0: # if the agentIndex is 0, its the maximizing PacMan
            return self.maxValue(gameState, gameState.getNumAgents() - 1, agentIndex)
        else: # otherwise, it's a minimizing ghost (there could be many)
            agentIndex += 1
            if gameState.getNumAgents() == agentIndex: # if we already done went through 
                agentIndex = 0
            if agentIndex == 0:
                depth += 1
            return min(self.getActionWithIndex(
              [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex - 1)], agentIndex, currentDepth))
              # return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
              #                   getLegalActionsNoStop(agent, gameState))
              
          # return self.minValue(gameState, gameState.getNumAgents() - 1, agentIndex)
        
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.getActionWithIndex(gameState, agentIndex=0, currentDepth=0)
        
  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
