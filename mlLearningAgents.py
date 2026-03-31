# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.pacmanPos = state.getPacmanPosition()
        self.ghostPositions = tuple(state.getGhostPositions())
        self.foodPositions = tuple(state.getFood().asList())
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        self.legalActions = tuple(legal)

    def __hash__(self):
        return hash((self.pacmanPos, self.ghostPositions, self.foodPositions))

    def __eq__(self, other):
        return isinstance(other, GameStateFeatures) and (
            self.pacmanPos == other.pacmanPos and
            self.ghostPositions == other.ghostPositions and
            self.foodPositions == other.foodPositions
        )


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.1,
                 gamma: float = 0.9,
                 maxAttempts: int = 5,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qValues = util.Counter()
        self.actionCounts = util.Counter()
        self.lastState = None
        self.lastAction = None
        self.lastGameState = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        reward = endState.getScore() - startState.getScore()
        if endState.isWin():
            reward += 500.0
        if endState.isLose():
            reward -= 500.0
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.qValues[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        if not state.legalActions:
            return 0.0
        return max(self.getQValue(state, a) for a in state.legalActions)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        oldQ = self.getQValue(state, action)
        target = reward + (self.getGamma() * self.maxQValue(nextState))
        updatedQ = ((1 - self.getAlpha()) * oldQ) + (self.getAlpha() * target)
        self.qValues[(state, action)] = updatedQ

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.actionCounts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.actionCounts[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.getMaxAttempts():
            return 1e6
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)
        if len(legal) == 0:
            return Directions.STOP

        # Learn from the previous real transition: the current state is the
        # actual successor of whatever action was taken last step (including
        # ghost movement), so this captures the true dynamics.
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.lastGameState, state)
            self.learn(self.lastState, self.lastAction, reward, stateFeatures)
            self.updateCount(self.lastState, self.lastAction)

        # Epsilon-greedy action selection.  During training the exploration
        # function biases the agent toward under-tried actions; during test
        # time (alpha == 0) we pick the action with the best Q-value,
        # breaking ties at random so the agent does not get stuck.
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            if self.getAlpha() > 0:
                actionValues = util.Counter()
                for candidate in legal:
                    qValue = self.getQValue(stateFeatures, candidate)
                    count = self.getCount(stateFeatures, candidate)
                    actionValues[candidate] = self.explorationFn(qValue, count)
                action = actionValues.argMax()
            else:
                qValues = [(self.getQValue(stateFeatures, a), a) for a in legal]
                maxQ = max(q for q, _ in qValues)
                bestActions = [a for q, a in qValues if q == maxQ]
                action = random.choice(bestActions)

        # Store current state and action for next-step learning.
        self.lastState = stateFeatures
        self.lastAction = action
        self.lastGameState = state

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Perform the final Q-learning update for the terminal transition.
        # getAction() is not called once the episode ends, so the last
        # transition (leading to win/loss) must be learned here.
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.lastGameState, state)
            terminalFeatures = GameStateFeatures(state)
            self.learn(self.lastState, self.lastAction, reward, terminalFeatures)
            self.updateCount(self.lastState, self.lastAction)

        self.lastState = None
        self.lastAction = None
        self.lastGameState = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
