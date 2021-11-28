package policy;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.valuefunction.QProvider;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

public class DecayingEpsilonGreedy extends EpsilonGreedy {
    public DecayingEpsilonGreedy(QProvider planner, double epsilon, double decay) {
        super(planner, epsilon);
        this.decay = decay;
    }

    private double decay;

    public DecayingEpsilonGreedy(double epsilon, double decay) {
        super(epsilon);
        this.decay = decay;
    }

    @Override
    public Action action(State s) {
        Action action = super.action(s);
        decayEpsilon();
        return action;
    }

    private void decayEpsilon() {
        double _epsilon = this.getEpsilon() * this.decay;
//        System.out.printf("DecayEpsilon: %f \n", _epsilon);
        this.setEpsilon(_epsilon);
    }
}
