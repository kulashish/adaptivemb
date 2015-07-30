package in.ac.iitb.cse.qh.util;

import cc.mallet.util.Maths;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.CurrentStateVector;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.data.TargetStateVector;

public class KLDivergenceCalculator {

	public static double calculate(CurrentState cstate, TargetState tstate) {
		CurrentStateVector[] cstateVectors = cstate.getP();
		TargetStateVector[] tstateVectors = tstate.getS();
		double totalKL = 0.0d;
		for (int i = 0; i < cstate.getTrainingSize(); i++)
			totalKL += Maths.klDivergence(tstateVectors[i].getSi(),
					cstateVectors[i].getPi());
		return totalKL;
	}
}
