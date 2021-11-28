package maze;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import policy.DecayingEpsilonGreedy;

import java.awt.*;

public class MazeProblem {

    private static final int MAX_X = 30;
    private static final int MAX_Y = 30;

    public static void main(String[] args) {

        GridWorldDomain gw = new GridWorldDomain(MAX_X, MAX_Y);
        gw.setMap(new MazeLevelGenerator().getMaze());
        gw.setProbSucceedTransitionDynamics(1); //stochastic transitions with 0.8 success rate
        //ends when the agent reaches a location
        final TerminalFunction tf = new SinglePFTF(
                PropositionalFunction.findPF(gw.generatePfs(), GridWorldDomain.PF_AT_LOCATION));

        //reward function definition
        final RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), 100, -0.01);

        gw.setTf(tf);
        gw.setRf(rf);

        final OOSADomain domain = gw.generateDomain(); //generate the grid world domain
        //setup initial state
        GridWorldState s = new GridWorldState(new GridAgent(0, 0),
                new GridLocation(29, 29, "loc0"));
        //initial state generator
        final ConstantStateGenerator sg = new ConstantStateGenerator(s);
        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();


        valueIteration(domain, s, hashingFactory);
        policyIteration(domain, s, hashingFactory);


        /**
         * Create factory for Q-learning agent
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
            private QLearning qLearning;

            public String getAgentName() {
                return "Q-learning";
            }

            public LearningAgent generateAgent() {
                if (qLearning == null) {
                    qLearning = new QLearning(domain, 0.99, hashingFactory, 0.3, 0.99, 1000);
                    DecayingEpsilonGreedy policy = new DecayingEpsilonGreedy(qLearning, 0.15, 0.999);
                    qLearning.setLearningPolicy(policy);
                }
                return qLearning;
            }
        };
//
//        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, sg);
        Visualizer visualizer = GridWorldVisualizer.getVisualizer(gw.getMap());
//        VisualActionObserver observer = new VisualActionObserver(domain, visualizer);
//        observer.initGUI();
//        env.addObservers(observer);
        //define experiment
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                1, 500, qLearningFactory);
        exp.setUpPlottingConfiguration(600, 400, 3, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.MEDIAN_EPISODE_REWARD,
                PerformanceMetric.AVERAGE_EPISODE_REWARD,
                PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
                PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE
        );
        //start experiment
        exp.startExperiment();
    }

    private static void valueIteration(OOSADomain domain, GridWorldState s, SimpleHashableStateFactory hashingFactory) {
        ValueIteration vi = new ValueIteration(domain, 0.99, hashingFactory, 0.00015, 1000);
        vi.setDebugCode(1);
        Policy policy = vi.planFromState(s);
        PolicyUtils.rollout(policy, s, domain.getModel());

        ValueFunctionVisualizerGUI guiVI = GridWorldDomain.getGridWorldValueFunctionVisualization(vi.getAllStates(), MAX_X, MAX_Y, vi, policy);
        guiVI.setTitle("Value Iteration VF");
        guiVI.setPreferredSize(new Dimension(1200, 1200));
        guiVI.initGUI();
    }

    private static void policyIteration(OOSADomain domain, GridWorldState s, SimpleHashableStateFactory hashingFactory) {

        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.00015, 1000, 1000);
        pi.setDebugCode(1);
        Policy piPolicy = pi.planFromState(s);

        ValueFunctionVisualizerGUI guiPI = GridWorldDomain.getGridWorldValueFunctionVisualization(pi.getAllStates(), MAX_X, MAX_Y, pi, piPolicy);
        guiPI.setTitle("Policy Iteration VF");
        guiPI.setPreferredSize(new Dimension(1200, 1200));
        guiPI.initGUI();
    }

}
