import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import policy.DecayingEpsilonGreedy;

import static burlap.domain.singleagent.blockdude.BlockDude.*;

public class BlockDudeTest {

    public static void main(String[] args) {

        BlockDude bd = new BlockDude(25, 25);

//        RewardFunction rf = new UniformCostRF();
        TerminalFunction tf = new BlockDudeTF();
        RewardFunction rf = new GoalBasedRF(tf, 10, -0.01);
        bd.setRf(rf);
        bd.setTf(tf);
        OOSADomain domain = bd.generateDomain();

        State initialState = BlockDudeLevelV2.getLevel4(domain);
//        State initialState = BlockDudeLevelConstructor.getLevel3(domain);

        final ConstantStateGenerator sg = new ConstantStateGenerator(initialState);
        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();


        Visualizer visualizer = BlockDudeVisualizer.getVisualizer(25, 25);

        VisualExplorer exp = new VisualExplorer(domain, visualizer, initialState);
        exp.addKeyAction("w", ACTION_UP, "");
        exp.addKeyAction("d", ACTION_EAST, "");
        exp.addKeyAction("a", ACTION_WEST, "");
        exp.addKeyAction("s", ACTION_PICKUP, "");
        exp.addKeyAction("x", ACTION_PUT_DOWN, "");
        exp.initGUI();

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

        valueIteration(domain, initialState, hashingFactory);
//        policyIteration(domain, initialState, hashingFactory);


        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, sg);
//        env.addObservers(new VisualActionObserver(domain, visualizer));
//        LearningAgent learningAgent = qLearningFactory.generateAgent();
//        for (int i = 0; i <= 500; i++) {
////            if( i == 500-1)
////            env.addObservers(new VisualActionObserver(domain, visualizer));
//            Episode episode = learningAgent.runLearningEpisode(env, 100);
//            Optional<Double> totalReward = episode.rewardSequence.stream().reduce((r1, r2) -> r1 + r2);
//            System.out.println("totalReward: " + totalReward);
//            env.resetEnvironment();
////        }


        //define experiment
        LearningAlgorithmExperimenterV2 experiment = new LearningAlgorithmExperimenterV2(env,
                1, 500, qLearningFactory);
//        experiment.toggleTrialLengthInterpretation(false);
        experiment.debugCode = 1;
        experiment.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.MEDIAN_EPISODE_REWARD,
                PerformanceMetric.AVERAGE_EPISODE_REWARD,
                PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
                PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE
        );
        //start experiment

        experiment.startExperiment();
        experiment.writeStepAndEpisodeDataToCSV("/home/tristani/Desktop/plotter_data");
    }

    private static void valueIteration(OOSADomain domain, State initialState, SimpleHashableStateFactory hashingFactory) {
        ValueIteration vi = new ValueIteration(domain, 0.99, hashingFactory, 0.00015, 500);
        vi.setDebugCode(1);
        Policy policy = vi.planFromState(initialState);
        PolicyUtils.rollout(policy, initialState, domain.getModel());


        ValueFunctionVisualizerGUI guiVI = GridWorldDomain.getGridWorldValueFunctionVisualization(vi.getAllStates(), 25, 25, vi, policy);
        guiVI.setTitle("Value Iteration VF");
        guiVI.initGUI();
    }

    private static void policyIteration(OOSADomain domain, State initialState, SimpleHashableStateFactory hashingFactory) {
        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.0015, 500, 500);
        pi.setDebugCode(1);
        Policy piPolicy = pi.planFromState(initialState);

        ValueFunctionVisualizerGUI guiPI = GridWorldDomain.getGridWorldValueFunctionVisualization(pi.getAllStates(), 25, 25, pi, piPolicy);
        guiPI.setTitle("Policy Iteration VF");
        guiPI.initGUI();
    }

}

