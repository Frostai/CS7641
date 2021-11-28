import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;

public class LearningAlgorithmExperimenterV2 extends LearningAlgorithmExperimenter {


    /**
     * Initializes.
     * The trialLength will be interpreted as the number of episodes, but it can be reinterpreted as a total number of steps per trial using the
     * {@link #toggleTrialLengthInterpretation(boolean)}.
     *
     * @param testEnvironment the test {@link Environment} in which experiments will be performed.
     * @param nTrials         the number of trials
     * @param trialLength     the length of the trials (by default in episodes, but can be intereted as maximum step length)
     * @param agentFactories  factories to generate the agents to be tested.
     */
    public LearningAlgorithmExperimenterV2(Environment testEnvironment, int nTrials, int trialLength, LearningAgentFactory... agentFactories) {
        super(testEnvironment, nTrials, trialLength, agentFactories);
    }

    @Override
    protected void runEpisodeBoundTrial(LearningAgentFactory agentFactory){

        //temporarily disable plotter data collection to avoid possible contamination for any actions taken by the agent generation
        //(e.g., if there is pre-test training)
        this.plotter.toggleDataCollection(false);

        LearningAgent agent = agentFactory.generateAgent();

        this.plotter.toggleDataCollection(true); //turn it back on to begin

        this.plotter.startNewTrial();

        for(int i = 0; i < this.trialLength; i++){
            agent.runLearningEpisode(this.environmentSever, 300);
            this.plotter.endEpisode();
            this.environmentSever.resetEnvironment();
        }

        this.plotter.endTrial();

    }

    protected void runStepBoundTrial(LearningAgentFactory agentFactory){

        //temporarily disable plotter data collection to avoid possible contamination for any actions taken by the agent generation
        //(e.g., if there is pre-test training)
        this.plotter.toggleDataCollection(false);
        LearningAgent agent = agentFactory.generateAgent();
        this.plotter.toggleDataCollection(true); //turn it back on to begin
        this.plotter.startNewTrial();
        int stepsRemaining = this.trialLength;
        while(stepsRemaining > 0){
            Episode ea = agent.runLearningEpisode(this.environmentSever, stepsRemaining);
            stepsRemaining -= ea.numTimeSteps()-1; //-1  because we want to subtract the number of actions, not the number of states seen
//            this.plotter.endEpisode();
            this.environmentSever.resetEnvironment();
        }
        this.plotter.endTrial();

    }
}
